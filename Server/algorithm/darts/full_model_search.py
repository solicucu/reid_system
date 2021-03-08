# -*- coding: utf-8 -*-
# @Author: solicucu
# @E-mail: 2748847613@qq.com
# @Date:   2020-02-21 15:09:31
# @Last Modified time: 2020-02-29 12:55:41

import json
import glob 
import torch
import copy 
import numpy as np  
import torch.nn as nn 
import torch.nn.functional as F 
import torch.nn.init as init 
from torch.autograd import Variable 
from operations import * 
from genotypes import * 

# use FOPS， split the attention module
class MixedOP(nn.Module):

	def __init__(self, C_in, C_out, stride, layer_size):

		super(MixedOP, self).__init__()
		self.ops = nn.ModuleList()
		for name in fop_names:
			op = FOPS[name](C_in, C_out, stride, layer_size, False)
			self.ops.append(op)

	def forward(self, x, weights):

		return sum(w * op(x) for w, op in zip(weights,self.ops))


class Cell(nn.Module):
	
	def __init__(self, in_planes, out_planes, stride, layer_size, step = 3):
		super(Cell, self).__init__()
		self.stride = stride
		self.in_planes = in_planes
		self.out_planes = out_planes
		self.layer_size = layer_size 
		self.step = step

		# if stride is 2, double the channel firstly
		if self.stride ==2:
			self.expand = ConvBNReLU(in_planes, out_planes, stride = self.stride)
			self.in_planes = out_planes

		# use for mixed op
		planes = out_planes // 3

		assert in_planes % 3 == 0 and out_planes % 3 ==0 
		# construct ops for each step
		"""
		Step1: s0 , s1, s2 -> s3  mixedop: 3
		Step2：s0, s1, s2, s3 -> s4  mixedop:4
		Step3: s0, s1, s2, s3, s4 -> s5  mixedop:5 
		"""
		self.ops = nn.ModuleList()

		for i in range(self.step):
			for j in range(3 + i):
				# all stride is 1
				op = MixedOP(planes, planes, 1, self.layer_size)
				self.ops.append(op)

		self.gconv1x1 = conv1x1(out_planes, out_planes, groups = 3)
	# weights: [12, 10]
	def forward(self, x, weights):

		if self.stride == 2:
			x = self.expand(x)

		states = []
		splits = x.chunk(3, dim = 1)
		states.extend(splits)

		offset = 0
		for i in range(self.step):
			state = sum(self.ops[offset + j](s, weights[offset + j]) for j, s in enumerate(states))
			states.append(state)

		res = torch.cat(states[-3:], dim = 1)

		# fuse the feature 
		res = channel_shuffle(res, groups = 3)
		res = self.gconv1x1(res)

		return res 


class FSNetwork(nn.Module):

	def __init__(self, num_class, in_planes, init_size, layers, use_gpu = False, pretrained = None):

		super(FSNetwork, self).__init__()
		self.num_class = num_class
		self.input_channels = in_planes
		self.layer_size = init_size 
		self.layers = layers 
		self.gpu = use_gpu
		self.width_multiplier = 2
		self.steps = 3
		self.loss = None
		# save the init value for create a new same model
		self.in_planes = in_planes
		# yes
		self.use_bnneck = 'yes'
		self.is_before_gap = False
		self.pretrained = pretrained 
		# here we must use deepcopy, othewise, self.layer_size and self.init_size will have same memory address
		self.init_size = copy.deepcopy(init_size)

		self.conv1 = ConvBNReLU(3, in_planes, kernel_size = 3, stride = 2)
		# size -> (128, 64)
		self.layer_size[0] //= 2
		self.layer_size[1] //= 2

		# put all layers' cell in a module list, in this way we can directly pass the weight to each cells
		self.cells = nn.ModuleList()
		for num in layers:
			self.cells += self._make_layer(num, self.steps)

		if self.use_bnneck == 'yes':
			self.neck = nn.BatchNorm2d(self.final_planes)

		self.GAP = nn.AvgPool2d((8,4))
		self.classifier = nn.Linear(self.final_planes, num_class)

		# init the parameters for architecture
		self._init_alphas()

		if self.pretrained is not None:
			print("use pretrained model from latest checkpoint")
			self._load_pretrained_model(self.pretrained)
		else:
			self.kaiming_init_()

	def _make_layer(self, num_cells, step):

		self.layer_size[0] //= 2
		self.layer_size[1] //= 2
		
		size = copy.deepcopy(self.layer_size)
		cells = []
		out_channels = self.input_channels * self.width_multiplier
		cells.append(Cell(self.input_channels, out_channels, 2, size, step))
		
		for i in range(num_cells -1):
			cells.append(Cell(out_channels, out_channels, 1, size, step))
		
		self.input_channels = out_channels
		self.final_planes = out_channels

		return cells

	def _init_alphas(self):

		k = len(self.layers)
		num_ops = len(fop_names)
		edges = 0
		for i in range(self.steps):
			edges += (3 + i)

		if self.gpu:
			self.alphas = Variable(1e-3 * torch.ones(k, edges, num_ops).cuda(), requires_grad = True)
		else:
			self.alphas = Variable(1e-3 * torch.ones(k, edges, num_ops), requires_grad = True)
		
		self.arch_parameters = [self.alphas]

	def _arch_parameters(self):

		return self.arch_parameters

	def new(self):

		init_size = copy.deepcopy(self.init_size)
		model_new = FSNetwork(self.num_class, self.in_planes, init_size, self.layers, self.gpu)

		if self.gpu:
			model_new = model_new.cuda()
		for x, y in zip(model_new._arch_parameters(), self._arch_parameters()):
			x.data.copy_(y.data)

		# set loss
		model_new._set_loss(self.loss)

		return model_new


	def forward(self, x):

		x = self.conv1(x)
		pos = -1
		weights = F.softmax(self.alphas, dim = -1)
		for i, num in enumerate(self.layers):
			# one layer
			for j in range(num):
				pos += 1
				x = self.cells[pos](x, weights[i])

		
		if self.is_before_gap:
			return x
		else:
			x = self.GAP(x)
		
		if self.use_bnneck == 'yes':
			last_feat = self.neck(x)
		else:
			last_feat = x 

		feat = x.view(x.shape[0], -1)
		last_feat = last_feat.view(last_feat.shape[0], -1)
		
		if self.training:
			cls_score = self.classifier(feat)
			return cls_score, last_feat
		else:
			return last_feat

	def _set_loss(self, loss_fn):

		self.loss = loss_fn

	# compute the loss
	def _loss(self, imgs, labels):

		score, feats = self(imgs)

		return self.loss(score, feats, labels)

	def kaiming_init_(self):
		print("use kaiming init")
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				init.kaiming_normal_(m.weight)
				if m.bias is not None:
					init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				if m.weight is not None:
					init.constant_(m.weight, 1)
				if m.bias is not None:
					init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				init.normal_(m.weight, std = 1e-3)
				if m.bias is not None:
					init.constant_(m.bias, 0)

	def _parse_genotype(self, file = 'fgenotype.json'):

		weights = F.softmax(self.alphas, dim = -1)
		if self.gpu:
			weights = weights.cpu()
		weights = weights.detach().numpy()
		
		geno = {}
		# save the selceted edge and ops for load the pretrained model
		edge_ops = {}
		num_layer = len(self.layers)
		num_ops = len(fop_names) # num ops for each edge
		
		for i in range(num_layer):
			# parse a cell
			ops = []
			edge_op = []
			start = 0 # for each layer, restart
			n = 3  # initial pre-node 
			for j in range(self.steps):
				end = start + n 
				# first select weights[i] for i layer
				w = weights[i][start:end]
				
				# select 2 edges with maximum weight
				
				edges = sorted(range(j + 3), key = lambda x: max(w[x][k] for k in range(num_ops)))[-2:]
				# edges is the number id for node
				# for each edges, select the max weighted op
				for e in edges:
					# w[e] a list [10-elem]
					ind = np.argmax(w[e], axis = -1) # numpy.int64
					ops.append((fop_names[ind], e)) # e: int 
					# numpy.int64 is not json wirtable, change to int 
					edge_op.append((start + e, int(ind)))

				n += 1
				start = end 
			# end parse a cell
			key = 'layer{}'.format(i+1)
			geno[key] = ops
			edge_ops[key] = edge_op

		# edge_ops = np.array(edge_ops)
		geno['edge_ops'] = edge_ops

		# save alpha
		alphas = copy.deepcopy(self.alphas)
		if self.gpu:
			alphas = alphas.cpu()
		alphas = alphas.detach().numpy().tolist()

		geno['alphas'] = alphas


		json_data = json.dumps(geno, indent = 4)
		with open(file, 'w') as f:
			f.write(json_data)

		return geno 


	def _load_pretrained_model(self, path):

		# checkpoint 
		ckpt_list = glob.glob(path + "checkpoint_*")
		ckpt_list = sorted(ckpt_list)
		ckpt_name = ckpt_list[-1]
		num = int(ckpt_name.split("_")[1].split(".")[0])
		self.start_epoch = num

		# self.load_state_dict(torch.load(ckpt_name))
		# or need to modify key 
		self_state_dict = self.state_dict()
		state_dict = torch.load(ckpt_name)
		for key in self_state_dict:
			self_state_dict[key].data.copy_(state_dict[key])
		print("load checkpoint from {}".format(ckpt_name))

		# genotype same number as ckpt
		geno_name = path + "genotype{}.json".format(num)
		with open(geno_name, 'r') as f:
			geno = json.load(f)

		alphas = torch.tensor(geno['alphas'])
		self.alphas.data.copy_(alphas)



if __name__ == "__main__":

	imgs = torch.randn(1,3,256,128)
	# model = MixedOP(30, 60, 1, [128,64])
	# model = Cell(30, 60, 2, [64, 32])
	weights = torch.randn(12,10)
	# res = model(imgs,weights)
	# print(res.size())
	model = FSNetwork(751, 30, [256, 128], layers = [2,2,2,2])
	# print(model)
	res = model(imgs)
	print(res[0].size())
	# model_new = model.new()
	# res = model_new(imgs)
	# print(res[0].size(), res[1].size())
	# model._parse_genotype()
