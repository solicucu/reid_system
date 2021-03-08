# -*- coding: utf-8 -*-
# @Author: solicucu
# @E-mail: 2748847613@qq.com
# @Date:   2020-01-19 13:28:32
# @Last Modified time: 2020-02-28 10:20:55

import torch
import torch.nn as nn  
import torch.nn.functional as F 
from .basic_ops import * 
from torch.autograd import Variable 

class  BasicBlock(nn.Module):
	"""docstring for  BasicBlock"""
	def __init__(self, in_planes, out_planes, stride = 1):
		super( BasicBlock, self).__init__()
		self.conv_bn1 = conv_bn(in_planes, out_planes, stride = stride)
		self.conv_bn2 = conv_bn(out_planes, out_planes)

		self.shortcut = nn.Sequential()
		if stride == 2 or in_planes != out_planes:

			self.shortcut = nn.Sequential(
				conv1x1(in_planes, out_planes,stride = stride),
				nn.BatchNorm2d(out_planes)
				)
		# SE layers use conv fc replace linear fc
		self.fc1 = conv1x1(out_planes, out_planes // 16)
		self.fc2 = conv1x1(out_planes // 16, out_planes)
		self.sigmoid = nn.Sigmoid()
		self.relu = nn.ReLU(True)

	def forward(self, x):

		out = self.conv_bn1(x)
		out = self.conv_bn2(out)

		# squeeze
		h, w = out.size(1), out.size(2)
		s = F.avg_pool2d(out, (h, w))
		s = self.relu(self.fc1(s))
		s = self.sigmoid(self.fc2(w))

		# excitation
		out = out * s 
		# add 
		out += self.shortcut(x)
		out = self.relu(out)

		return out 


class BottleNeck(nn.Module):
	"""docstring for BottleNeck"""
	def __init__(self, in_planes, out_planes, stride = 1 ):
		super(BottleNeck, self).__init__()
		mid_planes= out_planes // 4
		self.conv1x1_bn1 = conv1x1_bn(in_planes, mid_planes, stride = stride)
		self.conv_bn = conv_bn(mid_planes, mid_planes) 
		self.conv1x1_bn2 = conv1x1_bn(mid_planes, out_planes)

		self.shortcut = nn.Sequential()
		if stride == 2 or in_planes != out_planes:
			self.shortcut = nn.Sequential(
					conv(in_planes, out_planes, stride = stride),
					nn.BatchNorm2d(out_planes)
				)
		# SE layer
		self.fc1 = conv1x1(out_planes, out_planes // 16)
		self.fc2 = conv1x1(out_planes // 16, out_planes)
		self.sigmoid = nn.Sigmoid()
		self.relu = nn.ReLU()
	
	def forward(self, x):
		out = self.conv1x1_bn1(x)
		out = self.conv_bn(out)
		out = self.conv1x1_bn2(out)

		# squeeze
		h, w = out.size(2), out.size(3)
		s = F.avg_pool2d(out, (h, w))
		s = self.relu(self.fc1(s))
		s = self.sigmoid(self.fc2(s))

		# excitation
		out = out * s 

		# add 
		out += self.shortcut(x)
		out = self.relu(out)

		return out


class SEResNet(nn.Module):
	"""docstring for SEResNet"""
	def __init__(self, cfg, before_gap = False):
		super(SEResNet, self).__init__()
		block = cfg['block']
		out_planes = cfg['out_planes']
		num_blocks = cfg['num_blocks']

		self.is_berfore_gap = before_gap
		self.in_planes = 64
		self.final_planes = out_planes[-1]
		
		self.conv_bn = conv_bn(in_planes = 3, out_planes = 64, stride = 2)

		# self.max_pool = nn.MaxPool2d(kernel_size = 3, stride =2, padding = 1)

		self.layer1 = self._make_layer(block, out_planes[0], num_blocks[0]) 
		self.layer2 = self._make_layer(block, out_planes[1], num_blocks[1])
		self.layer3 = self._make_layer(block, out_planes[2], num_blocks[2])
		self.layer4 = self._make_layer(block, out_planes[3], num_blocks[3])	

		self.GAP = nn.AvgPool2d((8,4))

	def _make_layer(self, block, out_planes, num_block):
		
		layers = []
		for i in range(num_block):
			stride = 2 if i == 0 else 1
			layers.append(block(self.in_planes, out_planes, stride = stride))
			self.in_planes = out_planes

		return nn.Sequential(*layers)


	def forward(self, x):
		# 256 x 128
		out = self.conv_bn(x)
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		# 8 x 4

		if self.is_berfore_gap:
			return out
		else:
			return self.GAP(out)

config ={
	'seresent18':{
		"block":BasicBlock,
		'out_planes':[128, 256, 512, 1024],
		'num_blocks':[2, 2, 2, 2]
	},
	"seresnet34":{
		'block':BasicBlock,
		'out_planes':[128, 256, 512, 1024],
		'num_blocks':[3, 4, 6, 3]
	},
	'seresnet50':{
		'block':BottleNeck,
		'out_planes':[256, 512, 1024, 2048], # 
		'num_blocks':[3, 4, 6, 3] # 3, 4, 6, 3
	},
	'seresnet50_half':{
		'block':BottleNeck,
		'out_planes':[128, 256, 512, 1024], # 256，512，1024，2048
		'num_blocks':[3, 4, 6, 3] # 3, 4, 6, 3
	}
}
"""
Note: seresnet50 has different out_channels and num_blocks
for resnet is 'out_planes': [256,512, 1024, 2048]
              'num_block': [3,4,6,3]
"""
def get_seresnet(name, before_gap = False):

	return SEResNet(config[name], before_gap = before_gap)
		


if __name__ == "__main__":

	net = get_seresenet('seresnet50')
	print(net)

		
