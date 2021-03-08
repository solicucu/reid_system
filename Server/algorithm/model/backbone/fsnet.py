# -*- coding: utf-8 -*-
# @Author: solicucu
# @E-mail: 2748847613@qq.com
# @Date:   2020-02-25 16:19:58
# @Last Modified time: 2020-03-03 10:33:01
# root = "D:/project/Paper/papercode/myReID/MobileNetReID" # for pc
import sys 
# sys.path.append(root)
import copy 
import json 
import os 
import torch 

from darts import * 
from .basic_ops import * 
from config import cfg 
from utils.utils import * 

class Cell(nn.Module):

	def __init__(self, in_planes, out_planes, stride, op_names = None, layer_size = None, step = 3):
		super(Cell, self).__init__()
		self.stride = stride
		self.in_planes = in_planes
		self.out_planes = out_planes
		# op_names, a list of (operation_name, preprocessor) 
		self.op_names = op_names
		self.layer_size = layer_size 
		self.step = step 

		if self.stride == 2:
			self.expand = ConvBNReLU(in_planes, out_planes, stride = self.stride)
			self.in_planes = out_planes 

		assert in_planes % 3 == 0 and out_planes % 3 == 0
		planes = out_planes // 3

		# get the ops from selected op_names
		self.ops = nn.ModuleList()
		for item in self.op_names:

			self.ops.append(FOPS[item[0]](planes, planes, 1, self.layer_size, False))

	def forward(self, x):

		if self.stride == 2:
			x = self.expand(x)

		splits = x.chunk(3, dim = 1)

		states = []
		states.extend(splits)

		for i in range(self.step):
			# first and second edge for each node 
			k1, k2 = i * 2, i * 2 + 1
			
			state = sum(self.ops[k1](states[self.op_names[k1][1]]), self.ops[k2](states[self.op_names[k2][1]]))
			states.append(state)

		return torch.cat(states[-3:], dim = 1)

class FSNetwork(nn.Module):

	def __init__(self, cfg, before_gap = False):
		super(FSNetwork, self).__init__()
		
		self.input_channels = cfg.DARTS.IN_PLANES 
		self.layer_size = cfg.DARTS.INIT_SIZE
		self.layers = cfg.DARTS.LAYERS 
		self.gpu = cfg.MODEL.DEVICE == 'cuda'
		self.pretrained = cfg.DARTS.PRETRAINED 
		self.step = cfg.DARTS.STEP 

		self.ckpt_path = cfg.OUTPUT.ROOT_DIR + cfg.OUTPUT.CKPT_DIR  
		self.genotype_file = cfg.OUTPUT.ROOT_DIR + cfg.DARTS.GENOTYPE

		with open(self.genotype_file, 'r') as f:
			self.genotype = json.load(f)

		# just use for the first cell
		self.width_multiplier = cfg.DARTS.MULTIPLIER
		self.is_before_gap = before_gap 
		# no use ?
		# self.init_size = copy.deepcopy(cfg.DARTS.INIT_SIZE)

		self.conv1 = ConvBNReLU(3, self.input_channels, kernel_size = 3, stride = 2)
		# size -> (128, 64)
		self.layer_size[0] //= 2
		self.layer_size[1] //= 2

		self.cells = nn.ModuleList()
		layer_name = ['layer{}'.format(i) for i in range(1,5)]
		for name, num in zip(layer_name, self.layers):
			self.cells += self._make_layer(num, self.genotype[name], self.step)

		self.GAP = nn.AvgPool2d((8,4))

		if self.pretrained:
			self._load_state_dict(cfg)

	def _make_layer(self, num_cells, op_names, step):

		self.layer_size[0] //= 2
		self.layer_size[1] //= 2

		size = copy.deepcopy(self.layer_size)
		cells = []
		out_channels = self.input_channels * self.width_multiplier
		cells.append(Cell(self.input_channels, out_channels, 2, op_names, size, step))

		for i in range(num_cells - 1):
			cells.append(Cell(out_channels, out_channels, 1, op_names, size, step))

		self.input_channels = out_channels
		self.final_planes = out_channels 

		return cells 

	def forward(self, x):
		x = self.conv1(x)

		for cell in self.cells:
			x = cell(x)

		if self.is_before_gap:
			return x 
		else:
			return self.GAP(x)

	def _load_state_dict(self, cfg):

		print("load the pretrained search model")

		ckpt_fit = cfg.OUTPUT.ROOT_DIR + cfg.DARTS.CKPT_NAME

		if not os.path.exists(ckpt_fit):
			# need to change the checkpoint first
			ckpt = cfg.OUTPUT.ROOT_DIR + "best_ckpt.pth"
			full_change_state_dict(self, ckpt, self.genotype_file, self.layers)
		else:
			print("best_ckpt_fit.pth is existed")

		self.load_state_dict(torch.load(ckpt_fit))

		print("end of loading pretrained model")
		
	

if __name__ == "__main__":

	path = "D:\\project\\Paper\\papercode\\myReID\\MobileNetReID\\darts\\fgenotype.json"

	with open(path, 'r') as f:
		data = json.load(f)

	names = data['layer4']
	imgs = torch.randn(1,3, 256,128)
	# model = Cell(30, 30, 1, names, [256, 128])
	# print(model)
	model = FSNetwork(cfg)
	# res = model(imgs)
	# print(model)
	# print(res.size())
	# state_dict = model.state_dict()
	# for key in state_dict:
	# 	print(key)
	# ckpt_path = model.ckpt_path + 'best_ckpt.pth'
	# state_dict = torch.load(ckpt_path)
	# for key in state_dict:
	# 	print(key)
	# ckpt_path = model.ckpt_path + "best_ckpt.pth"
	# full_change_state_dict(model, ckpt_path, model.genotype_file, model.layers)


