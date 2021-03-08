# -*- coding: utf-8 -*-
# @Author: solicucu
# @E-mail: 2748847613@qq.com
# @Date:   2020-02-16 16:19:37
# @Last Modified time: 2020-04-18 10:21:03
# root = "D:/project/Paper/papercode/myReID/HAReID" # for pc
import sys 
# sys.path.append(root)
import copy 
import json 
import os 

from darts import * 
from .basic_ops import * 
from config import cfg 
from utils.utils import * 

class Cell(nn.Module):
	def __init__(self, in_planes, out_planes, stride, op_names, use_attention = False, layer_size = None, pretrained = False, double = False):

		super(Cell, self).__init__()
		self.op_names = op_names
		self.in_planes = in_planes
		self.out_planes = out_planes
		self.stride = stride
		self.double = double

		assert in_planes % 3 == 0 and out_planes % 3 == 0
		in_planes = in_planes // 3
		out_planes = out_planes // 3

		self.branch1 = OPS[op_names[0]](in_planes, out_planes, stride, double, False)
		self.branch2 = OPS[op_names[1]](in_planes, out_planes, stride, double, False)
		self.branch3 = OPS[op_names[2]](in_planes, out_planes, stride, double, False)

		self.gconv1x1 = conv1x1(self.out_planes, self.out_planes, groups = 3)

		if use_attention:
			self.attention = Attention(self.out_planes, layer_size)
		else:
			self.attention = None

		if stride == 2 or double:
			self.downsample = nn.Sequential(
				conv1x1(self.in_planes, self.out_planes, stride = stride),
				nn.BatchNorm2d(self.out_planes)
				)
		self.relu = nn.ReLU(inplace = True)

	def forward(self, x):

		residual = x 
		x1, x2, x3 = x.chunk(3, dim = 1)

		res = torch.cat([self.branch1(x1), self.branch2(x2), self.branch3(x3)], dim = 1)
		# shuffle
		res = channel_shuffle(res, groups = 3)
		# fuse the feature
		res = self.gconv1x1(res)

		# spatial attention and channel attention
		if self.attention is not None:
			res = self.attention(res)

		if self.stride == 2 or self.double:
			residual = self.downsample(residual)

		res += residual 
		res = self.relu(res)

		return res

class SSNetwork(nn.Module):
	# in_planes, init_size, layers, use_attention = False, genotype = "" use_gpu = False
	def __init__(self, cfg, before_gap = False):
		super(SSNetwork, self).__init__()

		self.input_channels = cfg.DARTS.IN_PLANES 
		self.layer_size = cfg.DARTS.INIT_SIZE
		self.layers = cfg.DARTS.LAYERS 
		self.gpu = cfg.MODEL.DEVICE == 'cuda'
		self.attention = cfg.DARTS.USE_ATTENTION
		self.pretrained = cfg.DARTS.PRETRAINED 
		self.last_stride = cfg.TRICKS.LAST_STRIDE 
		self.extend_planes = cfg.MODEL.EXTEND_PLANES 

		self.ckpt_path = cfg.OUTPUT.ROOT_DIR + cfg.OUTPUT.CKPT_DIR
		self.genotype_file = cfg.OUTPUT.ROOT_DIR + cfg.DARTS.GENOTYPE

		with open(self.genotype_file,'r') as f:
			self.genotype = json.load(f) 

		# just use for the first cell  
		self.width_multiplier = cfg.DARTS.MULTIPLIER 
		self.is_before_gap = before_gap
		self.init_size = copy.deepcopy(cfg.DARTS.INIT_SIZE)

		self.conv1 = ConvBNReLU(3, self.input_channels, kernel_size = 3, stride = 2)
		# size -> (128, 64)
		self.layer_size[0] //= 2
		self.layer_size[1] //= 2

		self.cells = nn.ModuleList()
		layer_name = ['layer{}'.format(i) for i in range(1,5)]
		for name,num in zip(layer_name,self.layers):
			stride = 2
			double = False
			if name == 'layer4':
				stride = self.last_stride
				double = True

			self.cells += self._make_layer(num, self.genotype[name], self.attention, stride, double)
		# extend the channels
		if self.extend_planes > 0 :
			self.extend = conv1x1(self.input_channels, self.extend_planes)
			self.input_channels = self.extend_planes

		# self.GAP = nn.AvgPool2d((8,4))
		self.GAP = nn.AdaptiveAvgPool2d(1)

		if self.pretrained:
			self._load_state_dict(cfg)

	def _make_layer(self, num_cells, op_names, attention, stride = 2, double = False):

		# layer_size // 2
		self.layer_size[1] //= 2
		self.layer_size[0] //= 2

		size = copy.deepcopy(self.layer_size)
		cells = []
		out_channels = self.input_channels * self.width_multiplier
		# first cell with stride = 2

		cells.append(Cell(self.input_channels, out_channels, stride, op_names, attention, size, double = double))

		for i in range(num_cells - 1):
			# atten = False if i < num_cells - 2 else attention
			cells.append(Cell(out_channels, out_channels, 1, op_names, False, size))

		# cells.append(Cell(out_channels, out_channels, 1, op_names, attention, size, double = double))

		self.input_channels = out_channels

		return cells

	def forward(self, x):
		x = self.conv1(x)
		# x = self.cells(x)
		for cell in self.cells:
			x = cell(x)
			
		if self.extend_planes > 0 :
			x = self.extend(x)

		if self.is_before_gap:
			return x 
		else:
			return self.GAP(x)

	def _load_state_dict(self, cfg):

		print("load the pretrained model")
		
		ckpt_fit = cfg.OUTPUT.ROOT_DIR + cfg.DARTS.CKPT_NAME

		if not os.path.exists(ckpt_fit):
			# need to change the checkpoint first
			ckpt = cfg.OUTPUT.ROOT_DIR + "best_ckpt.pth"
			change_state_dict(ckpt, self.genotype_file, self.layers)
		else:
			print("best_ckpt_fit.pth is exist")

		self_state_dict = self.state_dict()
		state_dict = torch.load(ckpt_fit)
		for key in self_state_dict:
			self_state_dict[key].data.copy_(state_dict[key].data)

		print("end of loading pretrained model")





if __name__ == "__main__":
	op_names = ["avg_pool_3x3","avg_pool_3x3","skip_connect"]
	# atten = Attention(30, [256, 128]) # stride =2 , layer_size / 2 firstly 
	# model = Cell(30, 60, 2, op_names, use_attention = True, layer_size = [64, 32])
	tensor = torch.randn(4,3,256, 128)
	model = SSNetwork(cfg, before_gap = True)
	# print(model)
	res = model(tensor)
	print(res.size())
	# two problem
	# pool identity if stride == 1, no double 
	# if stride = 1, redidual no double 