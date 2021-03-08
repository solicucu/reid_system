# -*- coding: utf-8 -*-
# @Author: solicucu
# @E-mail: 2748847613@qq.com
# @Date:   2020-01-10 17:03:13
# @Last Modified time: 2020-01-18 15:29:55

import torch.nn as nn 
from .basic_ops import * 

class MobileNet(nn.Module):

	def __init__(self,before_gap = False):

		super(MobileNet, self).__init__()

		self.is_before_gap = before_gap
		self.model = nn.Sequential(
			# 256 x 128
			conv_bn(3, 32, 2),
			conv_dw(32, 64, 1),
			# 128 x 64
			conv_dw(64, 128, 2),
			conv_dw(128, 128, 1),
			# 64 x 32
			conv_dw(128, 256, 2),
			conv_dw(256, 256, 1),
			# 32 x 16
			conv_dw(256, 512, 2),
			conv_dw(512, 512, 1),
			conv_dw(512, 512, 1),
			conv_dw(512, 512, 1),
			conv_dw(512, 512, 1),
			conv_dw(512, 512, 1),
			# 16 x 8 
			conv_dw(512, 1024, 2),
			conv_dw(1024, 1024, 1)
			# 8 x 4
		)
		# nn.AdaptiveAvgPool2d((8,4))
		self.GAP = nn.AvgPool2d((8,4))
	
	def forward(self, x):

		fmap = self.model(x)

		x = self.GAP(fmap)

		if self.is_before_gap:
			return fmap 
		else:
			return x  


if __name__ == "__main__":

	model = MobileNet()
	print(model)