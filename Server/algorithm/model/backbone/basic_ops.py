# -*- coding: utf-8 -*-
# @Author: solicucu
# @E-mail: 2748847613@qq.com
# @Date:   2020-01-18 15:23:51
# @Last Modified time: 2020-02-16 17:14:39
import torch 
import torch.nn as nn 

# use_std = True
conv_type = 'conv3x3'
# this conv can construct any type kernel_size 
def conv(in_planes, out_planes, stride = 1, groups = 1, bias = False):
	# can be change any type for the kernel_size
	convs = {
		"conv3x3": nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride,
			             padding = 1, groups = groups, bias = bias),
		"other": None
	}
	
	return convs[conv_type]

def conv1x1(in_planes, out_planes, stride = 1, groups = 1, bias = False):

	return nn.Conv2d(in_planes, out_planes, kernel_size = 1, stride = stride, 
					padding = 0, groups = groups ,bias = bias)

def conv1x1_bn(in_planes, out_planes, stride = 1, groups = 1, bias = False):

	return nn.Sequential(
			conv1x1(in_planes, out_planes, stride = stride, groups = groups, bias = bias),
			nn.BatchNorm2d(out_planes),
			nn.ReLU(inplace = True)
		)
	
def conv_bn(in_planes, out_planes, stride = 1, groups = 1, bias = False):

	return nn.Sequential(
		conv(in_planes, out_planes, stride = stride, groups = groups, bias = bias),
		nn.BatchNorm2d(out_planes),
		nn.ReLU(inplace = True)
	)

def conv_dw(in_planes, out_planes, stride = 1):
	
	return nn.Sequential(
			conv_bn(in_planes, in_planes, stride = stride, groups = in_planes),
	
			conv1x1(in_planes, out_planes),
			nn.BatchNorm2d(out_planes),
			nn.ReLU(inplace = True)
		)

def channel_shuffle(x, groups):

	batch_size, num_channels, height, width = x.size()
	channels_per_group = num_channels // groups

	# reshape
	x = x.view(batch_size, groups, channels_per_group, height, width )
	# shuffle
	x = torch.transpose(x, 1, 2).contiguous()

	x = x.view(batch_size, -1, height, width)

	return x 