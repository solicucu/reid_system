# -*- coding: utf-8 -*-
# @Author: solicucu
# @E-mail: 2748847613@qq.com
# @Date:   2020-01-18 15:06:14
# @Last Modified time: 2020-01-30 14:46:10

import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.autograd import Variable 
from .basic_ops import * 

def channel_shuffle(x, groups):

	batch_size, num_channels, height, width = x.size()
	channels_per_group = num_channels // groups

	# reshape
	x = x.view(batch_size, groups, channels_per_group, height, width )
	# shuffle
	x = torch.transpose(x, 1, 2).contiguous()

	x = x.view(batch_size, -1, height, width)

	return x 

class BottleNeck(nn.Module):

	def __init__(self, in_planes, out_planes, stride, groups):
		super(BottleNeck, self).__init__()
		self.stride = stride 
		mid_planes = out_planes // 4
		# the groud is 1 for the first stage,because the inplanes is two small
		self.g = 1 if in_planes == 24 else groups

		self.conv1x1_bn1 = conv1x1_bn(in_planes, mid_planes, groups = self.g, bias = False)
		self.conv_bn = conv_bn(mid_planes, mid_planes, stride = self.stride, groups = mid_planes, bias = False)
		self.conv1x1_bn2 = conv1x1_bn(mid_planes, out_planes, groups = self.g, bias = False)
		# if stride == 2 downsample 
		self.shortcut = nn.Sequential()
		if stride == 2:
			self.shortcut = nn.Sequential(nn.AvgPool2d(3, stride = 2, padding = 1))
		self.relu = nn.ReLU(inplace = True)

	def forward(self, x):

		out = self.conv1x1_bn1(x)
		out = channel_shuffle(out,self.g)
		out = self.conv_bn(out)
		out = self.conv1x1_bn2(out)
		# cat 
		if self.stride == 2:
			res = self.shortcut(x)
			out = self.relu(torch.cat((out,res), 1))
		
		return out

class ShuffleNetV1(nn.Module):

	def __init__(self, cfg, before_gap = False):

		super(ShuffleNetV1, self).__init__()
		out_planes = cfg['out_planes']
		num_blocks = cfg['num_blocks']
		groups = cfg['groups']

		self.is_before_gap = before_gap
		self.conv_bn = conv_bn(in_planes = 3, out_planes = 24, stride = 2, bias = False)
		self.in_planes = 24
		self.last_planes = out_planes[2]

		self.layer1 = self._make_layer(out_planes[0], num_blocks[0], groups)
		self.layer2 = self._make_layer(out_planes[1], num_blocks[1], groups)
		self.layer3 = self._make_layer(out_planes[2], num_blocks[2], groups)
		# self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		# for unified, we set the last channel is 1024
		self.conv1x1 = conv1x1(self.last_planes, 1024, stride = 2)

		self.GAP = nn.AvgPool2d((8, 4))


	def _make_layer(self, out_planes, num_block, groups):

		layers = []
		for i in range(num_block):
			stride = 2 if i == 0 else 1
			cat_planes = self.in_planes if i == 0 else 0

			layers.append(BottleNeck(self.in_planes, out_planes - cat_planes, stride = stride, groups = groups))
			self.in_planes = out_planes
		return nn.Sequential(*layers)

	def forward(self, x):
		# 256 x 128  5 downsmaple
		out = self.conv_bn(x)
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		# out = self.max_pool(out)
		out = self.conv1x1(out) 
		# 8 x 4

		if self.is_before_gap:
			return out 
		else:
			return self.GAP(out)


config = {
	"g1":{
		'out_planes': [144, 288, 576],
		'num_blocks': [4,8,4],
		'groups': 1
	},
	"g2":{
		'out_planes': [200, 400, 800],
		'num_blocks': [4,8,4],
		'groups': 2
	},
	"g3":{
		'out_planes': [240, 480, 960],
		'num_blocks': [4,8,4],
		'groups': 3
	},
	"g4":{
		'out_planes': [272, 544, 1088],
		'num_blocks': [4,8,4],
		'groups': 4
	},
	"g8":{
		'out_planes': [384, 768, 1536],
		'num_blocks': [4,8,4],
		'groups': 8
	},
}

def get_shufflenet(gname = 'g1', before_gap = False):

	return ShuffleNetV1(config[gname])

if __name__ == "__main__":

	model = get_shufflenet('g2')
	imgs = Variable(torch.randn(1, 3, 256, 128))
	res = model(imgs)

	print(res.size())