# -*- coding: utf-8 -*-
# @Author: solicucu
# @E-mail: 2748847613@qq.com
# @Date:   2020-01-31 16:22:47
# @Last Modified time: 2020-02-27 08:21:35
import torch 
import torch.nn as nn 
from torchvision import models 

class ResNet(nn.Module):

	def __init__(self, before_gap = False, pretrained = False):

		super(ResNet, self).__init__()

		self.is_before_gap = before_gap
		self.backbone = models.resnet50(pretrained = pretrained)
		self.GAP = nn.AvgPool2d((8,4))

		# if pretrain_path:
		# 	self.load_param(pretrain_path)

	def forward(self, x):

		# x = self.backbone(x)
		for name, midlayer in self.backbone._modules.items():

			x = midlayer(x)
			if name == "layer4":
				break 

		if self.is_before_gap:
			return x 
		else:
			return self.GAP(x)

	def load_param(self, path):
		print("load the pretrain model from {}".format(path))
		param_dict = torch.load(path)

		for name in param_dict:
			if 'fc' in name:
				continue
			key = 'backbone.' + name
			self.state_dict()[key].copy_(param_dict[name])




if __name__ == "__main__":

	imgs = torch.randn(1,3,256,128)
	path = "/home/share/solicucu/data/checkpoints/resnet50-19c8e357.pth"
	model = ResNet(pretrain_path = path)
	res = model(imgs)
	print(res.size())
