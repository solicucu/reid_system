# -*- coding: utf-8 -*-
# @Author: solicucu
# @E-mail: 2748847613@qq.com
# @Date:   2020-02-08 19:42:44
# @Last Modified time: 2020-03-22 22:05:29

import torch
import torch.nn as nn 
from torch.utils.model_zoo import load_url

model_urls = {
	
	'shufflenetv2_x0.5': 'https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth',
    'shufflenetv2_x1.0': 'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth',
    'shufflenetv2_x1.5': None,
    'shufflenetv2_x2.0': None,
}

def channel_shuffle(x, groups):

	batchsize, num_channels, height, width = x.size()
	channel_per_group = num_channels // groups

	# reshape
	x = x.view(batchsize, groups, channel_per_group, height, width)

	x = torch.transpose(x, 1, 2).contiguous()

	# flatten
	x = x.view(batchsize, -1, height, width)

	return x 

class InvertedResidual(nn.Module):

	def __init__(self, in_planes, out_planes, stride):

		super(InvertedResidual, self).__init__()
		if not (1 <= stride <= 3):
			raise ValueError("illegal stride value")
		self.stride = stride 
		# half of out_channels
		branch_features = out_planes // 2
		# change for last_stride, even stride is 1, still double the channel
		# if stide == 1, keep in_planes == out_planes else in_plane != out_plane
		self.hard_double = (self.stride == 1 and (in_planes != out_planes))
		# assert (self.stride != 1) or (in_planes == out_planes)
		# stride = 2, downsample
		# construct first branch
		if self.stride > 1:
			self.branch1 = nn.Sequential(
				self.depthwise_conv(in_planes, in_planes, kernel_size = 3, stride = self.stride, padding = 1),
				nn.BatchNorm2d(in_planes),
				nn.Conv2d(in_planes, branch_features, kernel_size = 1, stride = 1, padding = 0, bias = False),
				nn.BatchNorm2d(branch_features),
				nn.ReLU(inplace = True),
			)
		else:
			self.branch1 = nn.Sequential()

		self.branch2 = nn.Sequential(
			nn.Conv2d(in_planes if (self.stride > 1) else branch_features, branch_features, kernel_size = 1, stride = 1, padding = 0, bias = False),
			nn.BatchNorm2d(branch_features),
			nn.ReLU(inplace = True),

			self.depthwise_conv(branch_features, branch_features, kernel_size = 3, stride = self.stride, padding = 1),
			nn.BatchNorm2d(branch_features),
			nn.Conv2d(branch_features, branch_features, kernel_size = 1, stride = 1, padding = 0, bias = False),
			nn.BatchNorm2d(branch_features),
			nn.ReLU(inplace = True)
		)

	def forward(self, x):
		if self.stride == 1:
			x1, x2 = x.chunk(2, dim = 1)
			if self.hard_double:
				out = torch.cat((x, self.branch2(x)), dim = 1)
			else:
				out = torch.cat((x1, self.branch2(x2)), dim = 1)
		else:
			out = torch.cat((self.branch1(x), self.branch2(x)), dim = 1)

		return out

	@staticmethod
	def depthwise_conv(i, o, kernel_size, stride = 1, padding = 0, bias = False):

		return nn.Conv2d(i, o, kernel_size, stride, padding, bias = bias, groups = i)

class ShuffleNetV2(nn.Module):

	def __init__(self, 
				 stages_repeates, 
				 stages_out_channel, 
				 inverted_residual = InvertedResidual,
				 before_gap = False,
				 last_stride = 2):
		super(ShuffleNetV2, self).__init__()

		self.is_before_gap = before_gap

		if len(stages_repeates) != 3:
			raise ValueError("expected stages_repeated to be a list with 3 ints")
		if len(stages_out_channel) != 5:
			raise ValueError("expected stages_out_channle to be a list with 5 ints")
		self._stage_out_channels = stages_out_channel

		input_channels = 3
		output_channels = self._stage_out_channels[0]

		self.conv1 = nn.Sequential(
			nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias = False),
			nn.BatchNorm2d(output_channels),
			nn.ReLU(inplace = True)
		)
		input_channels = output_channels

		self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

		stage_name = [ 'stage{}'.format(i) for i in [2,3,4]]

		for name, repeats, output_channels in zip(stage_name, stages_repeates, self._stage_out_channels[1:]):
			stride = 2
			# last stride may change as 1
			if name == 'stage4':
				stride = last_stride

			seq = [inverted_residual(input_channels, output_channels, stride)]
			for i in range(repeats -1):
				seq.append(inverted_residual(output_channels, output_channels, 1))

			# add the attribute
			setattr(self, name, nn.Sequential(*seq))
			input_channels = output_channels

		output_channels = self._stage_out_channels[-1]
		self.final_planes = output_channels 
		
		self.conv5 = nn.Sequential(
			# pw
			nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias = False),
			nn.BatchNorm2d(output_channels),
			nn.ReLU(inplace = True)
		)
		# self.GAP = nn.AvgPool2d((8,4))
		self.GAP = nn.AdaptiveAvgPool2d(1)

	def _forward_impl(self, x):

		x = self.conv1(x)
		x = self.maxpool(x)
		x = self.stage2(x)
		x = self.stage3(x)
		x = self.stage4(x)
		x = self.conv5(x)

		if self.is_before_gap:
			return x 
		else:
			return self.GAP(x)

	def forward(self, x):
		return self._forward_impl(x)

def _shufflenetv2(name, *args, **kwargs):

	pretrained = kwargs['pretrained']
	
	kwargs.pop('pretrained')

	model = ShuffleNetV2(*args, **kwargs)

	if pretrained:
		model_url = model_urls[name]
		if model_url is None:
			raise NotImplementedError("pretrained model is not existed")
		else:
			print("load pretraied model from imagenet")
			state_dits = load_url(model_url, progress = True)
			self_state_dicts = model.state_dict()
			# for key1,key2 in zip(self_state_dicts.keys(),state_dits.keys()):
			# 	print(key1 == key2, key1, key2)
				# self_state_dicts[key] = state_dits[key]
			# notice: imagenet checkpoint does not save the param *.num_batches_tracked
			# we just ignore it 
			for key in self_state_dicts:
				if "num_batches_tracked" in key:
					continue
				self_state_dicts[key] = state_dits[key]

	return model

def shufflenet_v2(**kwargs):

	multiplier = kwargs['width_mult']
	kwargs.pop('width_mult')

	if multiplier == 0.5:

		return _shufflenetv2('shufflenetv2_x0.5', [4, 8, 4], [24, 48, 96, 192, 1024],**kwargs)
	
	elif multiplier == 1.0:
		
		return _shufflenetv2('shufflenetv2_x1.0', [4, 8, 4], [24, 116, 232, 464, 1024], **kwargs)

	elif multiplier == 1.5:

		return _shufflenetv2('shufflenetv2_x1.5', [4, 8, 4], [24, 176, 352, 704, 1024], **kwargs)

	elif multiplier == 2.0:

		# return _shufflenetv2('shufflenetv2_x2.0', [4, 8, 4], [24, 244, 488, 976, 2048], **kwargs)
		return _shufflenetv2('shufflenetv2_x2.0', [4, 8, 4], [24, 244, 488, 976, 1024], **kwargs)



"""
In my code, I jsut choose shufflenetv_v2_x1_0 as ShuffleNetV2
"""
if __name__ == "__main__":

	# model = InvertedResidual(1,2,2)
	model = shufflenet_v2_x1_0(before_gap = False, pretrained = True)
	# model = ShuffleNetV2([4, 8, 4], [24, 116, 232, 464, 1024], before_gap = True, pretrained = False)
	# print(model)
	imgs = torch.randn(1, 3, 256, 128)
	res = model(imgs)
	print(res.size()) # torch.Size([1, 1024, 8, 4])