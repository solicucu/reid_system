# -*- coding: utf-8 -*-
# @Author: solicucu
# @E-mail: 2748847613@qq.com
# @Date:   2020-03-15 10:53:06
# @Last Modified time: 2020-03-17 00:21:34
# root = "D:/project/Paper/papercode/myReID/HAReID"
# import sys 
# sys.path.append(root)

import torch 
import torch.nn as nn 
import torch.nn.functional as F 


# integrate the convlution, batchnorm, relu into a block
class ConvBlock(nn.Module):

	def __init__(self, in_planes, out_planes, kernel_size, stride = 1, padding = 0, bias = False):
		super(ConvBlock, self).__init__()

		self.op = nn.Sequential(
			nn.Conv2d(in_planes, out_planes, kernel_size, stride = stride, padding = padding, bias = bias),
			nn.BatchNorm2d(out_planes),
			nn.ReLU(inplace = True)
		)

	def forward(self, x):

		return self.op(x)

def channel_shuffle(x, groups):

	batchsize, num_channels, height, width = x.size()
	channel_per_group = num_channels // groups

	# reshape
	x = x.view(batchsize, groups, channel_per_group, height, width)

	x = torch.transpose(x, 1, 2).contiguous()

	# flatten
	x = x.view(batchsize, -1, height, width)

	return x 


def conv_dw(in_planes, out_planes, kernel_size = 3, stride = 1, padding = 1):

	return nn.Conv2d(in_planes, out_planes, kernel_size, stride = stride, padding = padding, bias = False)

def conv1x1(in_planes, out_planes, stride = 1, padding = 0):

	return nn.Conv2d(in_planes, out_planes, kernel_size = 1, stride = stride, padding = padding, bias = False)

# the stride only have the choice is 1
class ShuffleBlockA(nn.Module):

	def __init__(self, in_planes, out_planes, stride = 1):

		super(ShuffleBlockA, self).__init__()

		mid_planes = out_planes // 2
		# if no stride, in_planes must same as the out_planes
		assert stride == 1 

		self.branch2 = nn.Sequential(
			# conv1x1
			conv1x1(mid_planes, mid_planes), 
			nn.BatchNorm2d(mid_planes),
			nn.ReLU(inplace = True),

			# conv_dw
			conv_dw(mid_planes, mid_planes, stride = stride),
			nn.BatchNorm2d(mid_planes),

			# conv1x1 fuse
			conv1x1(mid_planes, mid_planes),
			nn.BatchNorm2d(mid_planes),
			nn.ReLU(inplace = True)
		)

	def forward(self, x):

		x1, x2 = x.chunk(2, dim = 1)

		res = torch.cat((x1, self.branch2(x2)), dim = 1)

		return res	

class ShuffleBlockB(nn.Module):

	def __init__(self, in_planes, out_planes, stride = 1):
		super(ShuffleBlockB, self).__init__()

		
		assert stride != 1 or (in_planes == out_planes)

		mid_planes = out_planes // 2
		self.branch1 = nn.Sequential(
			# conv_dw
			conv_dw(in_planes, in_planes, stride = stride),
			nn.BatchNorm2d(in_planes),
			# conv1x1
			conv1x1(in_planes, mid_planes),
			nn.BatchNorm2d(mid_planes),
			nn.ReLU(inplace = True)
		)
		self.branch2 = nn.Sequential(
			# conv1x1
			conv1x1(in_planes, in_planes),
			nn.BatchNorm2d(in_planes),

			# conv_dw
			conv_dw(in_planes, in_planes, stride = stride),
			nn.BatchNorm2d(in_planes),

			# conv1x1
			conv1x1(in_planes, mid_planes),
			nn.BatchNorm2d(mid_planes),
			nn.ReLU(inplace = True)
		)

	def forward(self, x):

		res = torch.cat((self.branch1(x), self.branch2(x)), dim = 1)

		return res 

class SpatialAtten(nn.Module):

	def __init__(self):
		super(SpatialAtten, self).__init__()

		self.conv1 = ConvBlock(1, 1, 3, stride = 2, padding = 1)
		self.conv2 = ConvBlock(1, 1, 1)

	def forward(self, x):

		size = x.size()[2:]
		# cross-channel average
		x = x.mean(1, keepdim = True)
		# conv3x3, stride = 2
		x = self.conv1(x)
		# resotre the spatial
		x = F.interpolate(x, size, mode = 'bilinear', align_corners = True)

		# scaling the value so as to fuse with channel_attention
		x = self.conv2(x)

		return x 

class ChannelAtten(nn.Module):

	def __init__(self, in_planes, reduction_rate = 16):

		super(ChannelAtten, self).__init__()
		assert in_planes % reduction_rate == 0

		self.conv1 = ConvBlock(in_planes, in_planes // reduction_rate, 1)
		self.conv2 = ConvBlock(in_planes // reduction_rate, in_planes, 1)

	def forward(self, x):
		# squeeze
		x = x.mean([2,3], keepdim = True)
		# excitation
		x = self.conv1(x)
		x = self.conv2(x)

		return x 

class SoftAtten(nn.Module):

	def __init__(self, in_planes):
		super(SoftAtten, self).__init__()
		self.spatial_atten = SpatialAtten()
		self.channel_atten = ChannelAtten(in_planes)
		self.conv1x1 = ConvBlock(in_planes, in_planes, 1)

	def forward(self, x):

		spatial_res = self.spatial_atten(x)
		channel_res = self.channel_atten(x)

		res = spatial_res * channel_res 
		res = torch.sigmoid(self.conv1x1(res))

		return res 

class HardAtten(nn.Module):

	def __init__(self, in_planes, regions = 4):
		super(HardAtten, self).__init__()
		self.regions = regions
		self.fc = nn.Linear(in_planes, regions * 2)
		self.init_params()

	def init_params(self):
		self.fc.weight.data.zero_()
		self.fc.bias.data.copy_(torch.Tensor([0, -0.75, 0, -0.25, 0, 0.25, 0, 0.75]))

	def forward(self, x):
		# squeeze 
		batch_size = x.size()[0]
		x = x.mean([2,3]).view(batch_size, -1)
		# predict the coordinate of each region
		theta = torch.tanh(self.fc(x))
		theta = theta.view(-1, self.regions, 2)

		return theta

class HarmAtten(nn.Module):

	def __init__(self, in_planes):
		super(HarmAtten, self).__init__()
		self.soft_atten = SoftAtten(in_planes)
		self.hard_atten = HardAtten(in_planes)

	def forward(self, x):

		softatten = self.soft_atten(x)
		theta = self.hard_atten(x)

		return softatten, theta 

class LocalBlock(nn.Module):

	def __init__(self, in_planes, out_planes, stride = 2, region_size = None, use_gpu = False):

		super(LocalBlock, self).__init__()
		self.size = region_size
		self.last_local_feats = None
		self.use_gpu = use_gpu 
		self.conv = ShuffleBlockB(in_planes, out_planes, stride = stride)
		self.init_scale_factors()


	def init_scale_factors(self):
		# initialize scale factor (s_w, s_h), set fixed 
		self.scale_factors = []
		for i in range(4):
			self.scale_factors.append(torch.Tensor([[1, 0], [0, 0.25]]))
	
	def set_data(self, theta, last_local_feats):
		self.theta = theta
		self.last_local_feats = last_local_feats

	def transform_theta(self, in_theta, idx):
		scale_fator = self.scale_factors[idx]
		theta = torch.zeros(in_theta.size(0), 2, 3)
		theta[:, :, :2] = scale_fator
		theta[:, :, -1] = in_theta 

		if self.use_gpu:
			theta = theta.cuda()
		return theta

	@staticmethod 
	def stn(x, theta):

		gird = F.affine_grid(theta, x.size())
		x  = F.grid_sample(x, gird)

		return x 

	def forward(self, x):
	
		feats_list = []
		for i in range(4):
			# get the theta of part i
			theta = self.theta[:, i, :]
			theta = self.transform_theta(theta, i)
			# according theta, sample the pixel from x
			# return the same size with x  
			x_sample = self.stn(x, theta)

			# resize the image
			region  = F.interpolate(x_sample, self.size, mode = 'bilinear', align_corners = True)

			if self.last_local_feats is not None:
				
				region += self.last_local_feats[i]
			
			res = self.conv(region)
	
			feats_list.append(res)

		return feats_list


class RobustNet(nn.Module):

	def __init__(self, num_class, cfg):
		super(RobustNet, self).__init__()

		self.num_class = num_class
		self.layers = cfg.MODEL.LAYERS 
		self.out_planes = cfg.MODEL.OUT_PLANES 
		self.feat_dims = cfg.MODEL.FEAT_DIMS 
		self.use_gpu = cfg.MODEL.DEVICE == 'cuda'
		self.learn_region = cfg.MODEL.LEARN_REGION 

		self.conv = ConvBlock(3, 32, 3, stride = 2, padding = 1)

		self.in_planes = 32
		# global branch
		self.layer1 = self._make_layer(self.layers[0], self.out_planes[0])
		self.ha1 = HarmAtten(self.out_planes[0])

		self.layer2 = self._make_layer(self.layers[1], self.out_planes[1])
		self.ha2 = HarmAtten(self.out_planes[1])

		self.layer3 = self._make_layer(self.layers[2], self.out_planes[2])
		self.ha3 = HarmAtten(self.out_planes[2])
		
		# also can use conv1x1
		self.GAP = nn.AdaptiveAvgPool2d(1)
		self.global_fc = nn.Sequential(
			nn.Linear(self.in_planes, self.feat_dims),
			nn.BatchNorm1d(self.feat_dims),
			nn.ReLU()
		)
		self.global_classifier = nn.Linear(self.feat_dims, num_class)

		# local branch
		if self.learn_region:
			self.local_conv1 = LocalBlock(32, self.out_planes[0], region_size = (24, 28), use_gpu = self.use_gpu)
			self.local_conv2 = LocalBlock(self.out_planes[0], self.out_planes[1], region_size = (12, 14), use_gpu = self.use_gpu)
			self.local_conv3 = LocalBlock(self.out_planes[1], self.out_planes[2], region_size = (6, 7), use_gpu = self.use_gpu)

			self.local_fc = nn.Sequential(
				nn.Linear(self.out_planes[2] * 4, self.feat_dims),
				nn.BatchNorm1d(self.feat_dims),
				nn.ReLU()
			)
			self.local_classifier = nn.Linear(self.feat_dims, num_class)
			# last feat dim for inference 
			self.feat_dims *= 2



	def _make_layer(self, num, out_planes):

		blocks = []
		blocks.append(ShuffleBlockB(self.in_planes, self.in_planes, stride = 1))
		for i in range(num):
			blocks.append(ShuffleBlockA(self.in_planes, self.in_planes))
		blocks.append(ShuffleBlockB(self.in_planes, out_planes, stride = 2 ))

		self.in_planes = out_planes

		return nn.Sequential(*blocks)


	def forward(self, x):

		assert x.size(2) == 160 and x.size(3) == 64, "the input size does not match, expected (160, 64)"

		x = self.conv(x)

		# layer 1
		# global branch 
		x1 = self.layer1(x)
		x1_atten, x1_theta = self.ha1(x1)
		x1_out = x1 * x1_atten

		if self.learn_region:
			self.local_conv1.set_data(x1_theta, None)
			# note that the input image is x 
			x1_local_feats = self.local_conv1(x)

		# layer 2
		# global branch
		x2 = self.layer2(x1_out)
		x2_atten, x2_theta = self.ha2(x2)
		x2_out = x2 * x2_atten

		if self.learn_region:
			self.local_conv2.set_data(x2_theta, x1_local_feats)
			x2_local_feat = self.local_conv2(x1_out)

		# layer3 
		# global branch 
		x3 = self.layer3(x2_out)
		x3_atten, x3_theta = self.ha3(x3)
		x3_out = x3 * x3_atten

		if self.learn_region:
			self.local_conv3.set_data(x3_theta, x2_local_feat)
			x3_local_feat = self.local_conv3(x2_out)

		# ========================feature generation=========================
		batch_size = x3_out.size(0)
		# change, the final_planes is out_channel[2]
		if self.learn_region:
			global_feat = self.GAP(x3_out).view(batch_size, -1)
			# returns [batch_size, 512]
			global_feat = self.global_fc(global_feat)
		else:
			global_feat = self.GAP(x3_out)

		if self.learn_region:
			local_feats = []

			for i in range(4):
				feat = self.GAP(x3_local_feat[i]).view(batch_size, -1)
				local_feats.append(feat)

			local_feat = torch.cat(local_feats, dim = 1)
			local_feat = self.local_fc(local_feat)

		if self.training:
			
			if self.learn_region:
				global_scores = self.global_classifier(global_feat)
				local_scores = self.local_classifier(local_feat)
				return (global_scores, global_feat), (local_scores, local_feat)

			else:
				return global_feat
		else:

			# L2 normalization before concatenation
			if self.learn_region:

				global_feat =  global_feat / global_feat.norm(p = 2, dim = 1, keepdim = True)
				local_feat = local_feat / local_feat.norm(p = 2, dim = 1, keepdim = True)

				return torch.cat([global_feat, local_feat], 1)
			else:
				# no need to normalization ???
				return global_feat


if __name__ == "__main__":

	tensor = torch.randn(2, 3, 160, 64)

	# model = ShuffleBlockA(4, 4)
	# model = ShuffleBlockB(4, 4, stride = 1)
	# model = SpatialAtten()
	# model = ChannelAtten(64)
	# model = SoftAtten(64)
	# model = HardAtten(64)
	model = RobustNet(751, cfg)
	# model = LocalBlock(64, 128)
	# print(model)
	# exit(1)
	res = model(tensor)
	print(res[0][0].size())
	print(res[0][1].size())