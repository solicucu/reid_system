# -*- coding: utf-8 -*-
# @Author: solicucu
# @E-mail: 2748847613@qq.com
# @Date:   2020-02-07 12:49:20
# @Last Modified time: 2020-02-07 12:57:15

import torch
import torch.nn as nn 
from torchvision import models 

class MobileNetV2(nn.Module):
	"""docstring for MobileNetV2"""
	def __init__(self, before_gap = False, pretrained = False):
		super(MobileNetV2, self).__init__()
		self.is_before_gap = before_gap
		self.backbone = models.mobilenet_v2(pretrained = pretrained)
		self.GAP = nn.AvgPool2d((8,4))

	def forward(self, x):

		for name, midlayer in self.backbone._modules.items():

			x = midlayer(x)
			if name == "features":
				break
		
		if self.is_before_gap:
			return x 
		else:
			return self.GAP(x) 

if __name__ == "__main__":

	imgs = torch.randn(1,3,256,128)
	model = MobileNetV2()
	# print(model)
	# res = model(imgs)
	# print(res.size()) torch.Size([1, 1280, 8, 4])