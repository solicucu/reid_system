# -*- coding: utf-8 -*-
# @Author: solicucu
# @E-mail: 2748847613@qq.com
# @Date:   2020-03-07 14:15:40
# @Last Modified time: 2020-03-09 13:16:43

import torch 
import torch.nn as nn 

class PCBNeck(nn.Module):

	def __init__(self, num_class, in_planes, mid_planes = 256, part_num = 4):

		super(PCBNeck, self).__init__()
		self.part_num = part_num

		self.GAP = nn.AdaptiveAvgPool2d(1)
		self.conv1x1_list = nn.ModuleList()
		self.fc_list = nn.ModuleList()
		# self.fc = nn.Linear(4 * mid_planes, num_class)
		# randomly replace some elem with zero
		self.dropout = nn.Dropout(0.5)
		
		for i in range(part_num):
			self.conv1x1_list.append(nn.Conv2d(in_planes, mid_planes, kernel_size = 1, bias = False))
			self.fc_list.append(nn.Linear(mid_planes, num_class))
			

	def forward(self, x):

		batch_size = x.size()[0]
		parts = x.chunk(self.part_num, dim = 2)

		vector_g = [self.dropout(self.GAP(p)) for p in parts]
		# vector_g = [self.GAP(p) for p in parts]

		vector_h = [self.dropout(op(p).view(batch_size, -1)) for op, p in zip(self.conv1x1_list, vector_g)]
		# vector_h = [op(p).view(batch_size, -1) for op, p in zip(self.conv1x1_list, vector_g)]
		
		scores = [fc(p) for fc, p in zip(self.fc_list, vector_h)]
		# average the score 
		# score = torch.cat(scores, dim = 0).mean(dim =0)
		
		last_feat = torch.cat(vector_h, dim = -1)

		# score = self.fc(last_feat)

		if self.training:
			return scores, last_feat
		else:
			return last_feat




import numpy as np 
def count_parameters(model):
	# see the param according the model 
	# print(model)
	# for name, param in model.named_parameters():
	# 	print(name, param.size())
		
	return np.sum(np.prod(param.size()) for param in model.parameters()) / 1e6

if __name__ == "__main__":

	tensor = torch.randn(1,1024, 8, 4)
	model = PCBNeck(751, 1024, 256, 4)
	# print(model)
	size = count_parameters(model)
	print(size)

	res = model(tensor)

	print(res[0].size(), res[1].size())


