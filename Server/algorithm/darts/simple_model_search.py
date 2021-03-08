# -*- coding: utf-8 -*-
# @Author: solicucu
# @E-mail: 2748847613@qq.com
# @Date:   2020-02-12 15:37:13
# @Last Modified time: 2020-03-07 12:00:20
import copy 
import glob 
import json 
import torch
import torch.nn as nn 
from operations import * 
from genotypes import * 
from torch.autograd import Variable 
import torch.nn.functional as F
import torch.nn.init as init 

# note that channel will double only when stride = 2
class MixedOp(nn.Module):

	def __init__(self, C_in, C_out, stride):

		super(MixedOp, self).__init__()
		self.ops = nn.ModuleList()
		for name in op_names:
			op = OPS[name](C_in, C_out, stride, False)
			self.ops.append(op)

	def forward(self, x, weights):
		
		# ws = weights.clone().to(x.device)

		return sum(w * op(x) for w, op in zip(weights, self.ops))



class Cell(nn.Module):

	def __init__(self, in_planes, out_planes, stride, use_attention = False, layer_size = None):
		super(Cell, self).__init__()

		self.stride = stride
		# save for later
		self.in_planes = in_planes
		self.out_planes = out_planes
		assert in_planes % 3 == 0 and out_planes % 3 ==0

		in_planes = in_planes // 3
		out_planes = out_planes // 3
		self.branch1 = MixedOp(in_planes, out_planes, stride)
		self.branch2 = MixedOp(in_planes, out_planes, stride)
		self.branch3 = MixedOp(in_planes, out_planes, stride)

		self.gconv1x1 = conv1x1(self.out_planes, self.out_planes, groups = 3)
		
		if use_attention:
			self.attention = Attention(self.out_planes, layer_size)
		else:
			self.attention = None
		

	"""
	weights: [3,8], weights for each brance
	"""
	def forward(self, x, weights):

		x1, x2, x3 = x.chunk(3, dim = 1)

		res = torch.cat([self.branch1(x1, weights[0]), self.branch2(x2, weights[1]), self.branch3(x3, weights[2])], dim = 1)
		# shuffle
		res = channel_shuffle(res, groups = 3)
		# fuse the feature 
		res = self.gconv1x1(res)
		# channel attention and spatial attention
		# channel attention
		if self.attention is not None:
			res = self.attention(res)

		return res 

class SSNetwork(nn.Module):

	def __init__(self, num_class, in_planes, init_size, layers, use_attention = True, use_gpu = False, pretrained = None):
		super(SSNetwork, self).__init__()

		self.num_class = num_class
		self.input_channels = in_planes
		self.layer_size = init_size
		self.layers = layers
		self.gpu = use_gpu

		self.attention = use_attention
		# just use for the first cell in each layer
		self.width_multiplier = 2
		# save the init value for create a new same model
		self.in_planes = in_planes
		# follow variable value will pass by param
		self.final_planes = 0
		self.use_bnneck = 'no'
		self.is_before_gap = False
		self.pretrained = pretrained

		# here we must use deepcopy, othewise, self.layer_size and self.init_size will have same memory address
		self.init_size = copy.deepcopy(init_size) 
		# print(id(self.layer_size))
		# print(id(self.init_size))

		self.conv1 = ConvBNReLU(3, in_planes, kernel_size = 3, stride = 2)
		# size -> (128, 64)
		self.layer_size[0] //= 2
		self.layer_size[1] //= 2


		"""
		self.layer1 = self._make_layer(layers[0], self.attention) 64 x 32
		self.layer2 = self._make_layer(layers[1], self.attention) 32 x 16
		self.layer3 = self._make_layer(layers[2], self.attention) 16 x 8 
		self.layer4 = self._make_layer(layers[3], self.attention) 8 x 4 
		"""
		# put all layers' cell in a module list, in this way we can directly pass the weight to each cells
		self.cells = nn.ModuleList()
		for num in layers:
			self.cells += self._make_layer(num, self.attention)
		
		# yes only when use triplet loss
		if self.use_bnneck == 'yes':
			self.neck = nn.BatchNorm2d(self.final_planes)

		self.GAP = nn.AvgPool2d((8,4))
		self.classifier = nn.Linear(self.final_planes, self.num_class)
		#init the parameters for architecture 
		self._init_alphas()
		# with different address but has the same layer_size...???
		# for cell in self.cells:
		# 	print(cell.attention.layer_size)
		# 	# print(id(cell.attention))
		if self.pretrained is not None:
			print("use pretrained model from latest model")
			self._load_pretrained_model(self.pretrained)
		else:
			self.kaiming_init_()

	def _make_layer(self, num_cells, attention):
		# layer_size // 2
		self.layer_size[0] //= 2
		self.layer_size[1] //= 2
		# copy the size, don't pass the layer_size directly, or each attention has the same layer_size as the last size
		size = [self.layer_size[0], self.layer_size[1]]
		cells = []
		out_channels = self.input_channels * self.width_multiplier
		# first cell with stride == 2
		cells.append(Cell(self.input_channels, out_channels, 2, attention, size))
		
		for i in range(num_cells -1):
			cells.append(Cell(out_channels, out_channels, 1, attention, size))

		self.input_channels = out_channels
		self.final_planes = out_channels

		return cells

	def _init_alphas(self):

		k = len(self.layers)
		num_ops = len(op_names)

		if self.gpu:
			self.alphas = Variable(1e-3 * torch.ones(k, 3, num_ops).cuda(), requires_grad = True)
		else:
			self.alphas = Variable(1e-3 * torch.ones(k, 3, num_ops), requires_grad = True)
		# optimizer need a list 
		self.arch_parameters = [self.alphas]

	def _arch_parameters(self):

		return self.arch_parameters

	def new(self):
		# we need to keep the self.init_size same, so do not directly pass the self.init_size
		# or, every time make a new model, the self.init_size continously decrease
		init_size = copy.deepcopy(self.init_size) 
		model_new = SSNetwork(self.num_class, self.in_planes, init_size, self.layers, self.attention, self.gpu)
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
		# ws = F.softmax(self.alphas, dim = -1)
		weights = F.softmax(self.alphas, dim = -1)
		# weights = ws.clone().to(x.device) # keep the same device
		# print("x_device",x.device)
		# print("weight_device", weights.device)
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


	
	def _parse_genotype(self, file = './genotype.json'):
		# change to cpu 
		
		geno = {}
		# four layers 
		for i in range(4):
			w = self.alphas[i]
			# find the maxvalue indices 
			_, indices = w.max(dim = -1)
			ops = []

			if self.gpu:
				indices = indices.cpu().numpy()
			else:
				indices = indices.numpy()

			for ind in indices:
				ops.append(op_names[ind])
			key = 'layer{}'.format(i+1)
			geno[key] = ops

		alphas = copy.deepcopy(self.alphas)
		if self.gpu:
			alphas = alphas.cpu()

		alphas = alphas.detach().numpy().tolist()
		
		geno["alphas"] = alphas

		json_data = json.dumps(geno, indent = 4)
		with open(file, 'w') as f:
			f.write(json_data)

		return geno

	def _set_loss(self, loss_fn):

		self.loss = loss_fn 

	# compute the loss 
	def _loss(self, imgs, labels):

		score, feats = self(imgs)

		return self.loss(score, feats, labels)

	def kaiming_init_(self):

		# print("use kaiming init")
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

	def _load_pretrained_model(self, path):

		# checkpoint
		ckpt_list = glob.glob(path + "checkpoint_*")
		# linux is big to small 
		ckpt_list = sorted(ckpt_list)
		
		ckpt_name = ckpt_list[-1]
		num = int(ckpt_name.split("_")[1].split(".")[0])
		self.start_epoch = num

		self_state_dict = self.state_dict()

		state_dict = torch.load(ckpt_name)
		print("load checkpoint from {}".format(ckpt_name))
		
		# because use dataparallel, has extra module.
		for key in self_state_dict:
			# new_key = 'module.'+ key
			self_state_dict[key].data.copy_(state_dict[key].data) 
		

		# self.load_state_dict(ckpt_name)
		# genotype 
		genotype_list = glob.glob(path + "genotype_*")
		genotype_list = sorted(genotype_list)
		
		geno_name = genotype_list[-1]
		with open(geno_name, 'r') as f:
			geno = json.load(f)

		alphas = torch.tensor(geno['alphas'])
		self.alphas.data.copy_(alphas)


if __name__ == "__main__":

	# weights = list(range(8))
	weights = torch.randn(3, 8).numpy()
	tensor = torch.randn(1,3,256,128)
	# model = MixedOp(2,2,1)
	# model = Cell(30,30,1, use_attention = True, layer_size = (4,4))
	model = Network(3,30,[256,128],[2,4,6,2],True)
	# print(model)
	# res = model(tensor, weights)
	# param = model._arch_parameters()
	# print(param)
	# print(res)
	genotype = model._parse_genotype()
	print(genotype)

	res = model(tensor)
	# print(res.size())
	# model_new = model.new()
	# res2 = model_new(tensor)
	# print(res2.size())
	# param is different
	# print(res[0][0])
	# print(res2[0][0])
