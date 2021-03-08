# -*- coding: utf-8 -*-
# @Author: solicucu
# @E-mail: 2748847613@qq.com
# @Date:   2020-02-29 17:01:48
# @Last Modified time: 2020-05-03 09:07:27
import numpy as np 
import torch
import torch.nn as nn 


def conv2d(in_planes, out_planes, kernel_size = 2, stride = 1, padding = 0):

	return nn.Conv2d(in_planes, out_planes, kernel_size = kernel_size, stride = stride, padding = padding, bias = False)

class ConvBlock(nn.Module):

	def __init__(self, in_planes, out_planes, kernel_size = 2, stride = 1, padding = 0, affine = True):
		super(ConvBlock, self).__init__()

		self.op = nn.Sequential(
			# nn.Conv2d(in_planes, out_planes, kernel_size = kernel_size, stride = stride, padding = padding, bias = False),
			#dw_conv
			nn.Conv2d(in_planes, in_planes, kernel_size = kernel_size, stride = stride, padding = padding, groups = in_planes, bias = False),
			nn.Conv2d(in_planes, out_planes, kernel_size = 1, padding = 0, bias = False)
			# nn.BatchNorm2d(out_planes),
			# nn.ReLU(inplace = True)
		)
		
	def forward(self, x):

		return self.op(x)


class GlobalBranch(nn.Module):

	def __init__(self, num_class, in_planes, use_bnneck = False, dropout = 0):
		super(GlobalBranch, self).__init__()

		self.use_bnneck = use_bnneck
		self.dropout = dropout
		self.GAP = nn.AdaptiveAvgPool2d(1)

		if use_bnneck:
			self.bnneck = nn.BatchNorm2d(in_planes)
		# print("Global_Branch: final_planes is {}".format(in_planes))
		self.drop = nn.Dropout(dropout)
		self.classifier = nn.Linear(in_planes, num_class)

	def forward(self, x):

		x = self.GAP(x)

		if self.use_bnneck:
			last_feat = self.bnneck(x)
		else:
			last_feat = x 

		# flatten the feature
		feat = x.view(x.shape[0], -1)
		last_feat = last_feat.view(last_feat.shape[0], -1)
		if self.dropout > 0 :
			feat = self.drop(feat)
			last_feat = self.drop(last_feat)

		if self.training:

			cls_score = self.classifier(feat)
			return cls_score, last_feat
		else:
			return last_feat 

class PPL(nn.Module):

	def __init__(self, num_class, in_planes, mid_planes = 512, use_bnneck = False, dropout = 0):

		super(PPL, self).__init__()
		self.use_bnneck = use_bnneck
		self.dropout = dropout

		self.first_list = nn.ModuleList()
		for i in range(4):
			self.first_list.append(ConvBlock(in_planes, mid_planes, kernel_size = 2, stride = (2, 1)))
		
		self.second_list = nn.ModuleList()
		for i in range(2):
			self.second_list.append(ConvBlock(mid_planes, mid_planes, kernel_size = 2, stride = (2, 1)))

		self.GAP = nn.AdaptiveAvgPool2d(1)
		self.classifier = nn.Linear( 2 * mid_planes, num_class)
		
		if self.use_bnneck:
			self.bnneck = nn.BatchNorm2d(6 * mid_planes)
		self.drop = nn.Dropout(dropout)
		# print("PPL branch final_planes for softmax is {}, for triplet is {}".format(2 * mid_planes, 6 * mid_planes))

	def forward(self, x):
		
		first_parts = x.chunk(4, dim = 2)
		first_res = [op(p) for op, p in zip(self.first_list, first_parts)]
		first_part_feats = [self.GAP(p) for p in first_res]

		first_map = torch.cat(first_res, dim = 2)
		first_global_feat =  self.GAP(first_map)

		second_part = first_map.chunk(2, dim = 2)
		second_res = [op(p) for op, p in zip(self.second_list, second_part)]
		second_part_feats = [self.GAP(p) for p in second_res]

		second_map = torch.cat(second_res, dim = 2)
		second_global_feat = self.GAP(second_map)

		part_feats = torch.cat(first_part_feats + second_part_feats, dim = 1)
		global_feat = torch.cat([first_global_feat, second_global_feat], dim = 1)

		# print(part_feats.size()) # 3072
		# print(global_feat.size()) # 1024
		if self.use_bnneck:
			part_feats = self.bnneck(part_feats)

		batch = global_feat.size(0)
		global_feat = global_feat.view(batch, -1)
		part_feats = part_feats.view(batch, -1)

		if self.dropout > 0 :
			global_feat = self.drop(global_feat)
			part_feats = self.drop(part_feats)

		if self.training:
			cls_score = self.classifier(global_feat)
			return cls_score, part_feats
		else:
			return part_feats 



# return two pairs to calculate loss
class Neck(nn.Module):

	def __init__(self, num_class, in_planes, mid_planes = 512, use_bnneck = False, dropout = 0):

		super(Neck, self).__init__()

		self.global_branch = GlobalBranch(num_class, in_planes, use_bnneck, dropout)
		self.part_branch = PPL(num_class, in_planes, mid_planes, use_bnneck, dropout)

	def forward(self, x):
		
		global_res = self.global_branch(x)
		part_res = self.part_branch(x)

		if self.training:
			return [global_res, part_res]
		else:
			# only consider the part features for test
			# return part_res
			# consider as both global feats and local feats
			feats = torch.cat([global_res,part_res], dim = 1)
			return feats 

def count_parameters(model):
	# see the param according the model 
	# print(model)
	# for name, param in model.named_parameters():
	# 	print(name, param.size())
		
	return np.sum(np.prod(param.size()) for param in model.parameters()) / 1e6

if __name__ == "__main__":

	imgs = torch.randn(10,1024,8,4)

	# model = GlobalBranch(10, 100)
	# model = PPL(10,1024)
	model = Neck(751, 1024, 256)
	res = count_parameters(model)
	print(res)

	# print(model)

	# global_res, part_res = model(imgs)
	# print(global_res[0].size(), global_res[1].size())
	# print(part_res[0].size(), part_res[1].size())
	