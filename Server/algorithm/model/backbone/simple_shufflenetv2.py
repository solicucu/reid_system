# -*- coding: utf-8 -*-
# @Author: solicucu
# @E-mail: 2748847613@qq.com
# @Date:   2020-02-07 12:32:18
# @Last Modified time: 2020-02-07 12:48:03

import torch
import torch.nn as nn 
from torchvision import models 

class ShuffleNetV2(nn.Module):
	"""docstring for ShuffleNetV2"""
	def __init__(self, before_gap = False, pretrained = False):
		super(ShuffleNetV2, self).__init__()
		self.is_before_gap = before_gap
		self.backbone = models.shufflenet_v2_x1_0(pretrained = pretrained)
		self.GAP = nn.AvgPool2d((8,4))

	def forward(self, x):

		for name, midlayer in self.backbone._modules.items():

			x = midlayer(x)
			if name == "conv5":
				break

		if self.is_before_gap:
			return x 
		else:
			return self.GAP(x) 

if __name__ == "__main__":

	imgs = torch.randn(1,3,256,128)
	model = ShuffleNetV2()
	# print(model)
	res = model(imgs)
	print(res.size())

