# -*- coding: utf-8 -*-
# @Author: solicucu
# @E-mail: 2748847613@qq.com
# @Date:   2020-02-11 19:39:28
# @Last Modified time: 2020-03-19 21:15:17
import torch 
import torch.nn as nn 
import torch.nn.functional as F 


def channel_shuffle(x, groups):

	batch_size, num_channels, height, width = x.size()
	channels_per_group = num_channels // groups

	# reshape
	x = x.view(batch_size, groups, channels_per_group, height, width )
	# shuffle
	x = torch.transpose(x, 1, 2).contiguous()

	x = x.view(batch_size, -1, height, width)

	return x 

class ConvBNReLU(nn.Module):

	def __init__(self, in_planes, out_planes, kernel_size = 3, stride = 1, padding = 1, affine = True):
		super(ConvBNReLU, self).__init__()

		self.op = nn.Sequential(
			nn.Conv2d(in_planes, out_planes, kernel_size = kernel_size, stride = stride, padding = padding, bias = False),
			nn.BatchNorm2d(out_planes),
			nn.ReLU(inplace = True)
		)

	def forward(self, x):

		return self.op(x)

class SepConv(nn.Module):

	def __init__(self, in_planes, out_planes, kernel_size, stride, padding, double = False, affine = True):
		super(SepConv, self).__init__()
		self.op = nn.Sequential(
			nn.Conv2d(in_planes, in_planes, kernel_size = kernel_size, stride = stride, padding = padding, groups= in_planes, bias = False),
			nn.Conv2d(in_planes, out_planes, kernel_size = 1, padding = 0, bias = False),
			nn.BatchNorm2d(out_planes),
			nn.ReLU(inplace = True)
		)

	def forward(self, x):

		return self.op(x)

class DilConv(nn.Module):
	"""docstring for DilConv"""
	def __init__(self, in_planes, out_planes, kernel_size, stride, padding, dilation, double = False, affine = True ):
		super(DilConv, self).__init__()

		self.op = nn.Sequential(
			nn.Conv2d(in_planes, in_planes, kernel_size = kernel_size, stride = stride, padding = padding, groups = in_planes, dilation = dilation, bias = False),
			nn.Conv2d(in_planes, out_planes, kernel_size = 1, padding = 0, bias = False),
			nn.BatchNorm2d(out_planes),
			nn.ReLU(inplace = True)
		)

	def forward(self, x):

		return self.op(x)

class Identity(nn.Module):
	def __init__(self, in_planes, out_planes, double = False):
		super(Identity,self).__init__()
		self.double = double

		if double:
			self.conv = conv1x1(in_planes, out_planes)

	def forward(self, x):
		if self.double:
			x = self.conv(x)

		return x 

class Zero(nn.Module):
	def __init__(self, stride, out_planes):

		super(Zero, self).__init__()
		self.stride = stride
		self.out_planes = out_planes

	def forward(self, x):

		b, C_in, h, w = x.size()
		if self.stride == 1:
			return torch.zeros(b, self.out_planes, h, w)
		else:
			return torch.zeros(b, self.out_planes, h //self.stride, w // self.stride)

class FactorizedReduce(nn.Module):
	def __init__(self, in_planes, out_planes, stride, affine = True):
		super(FactorizedReduce, self).__init__()

		assert out_planes % 2 == 0
		self.conv1 = nn.Conv2d(in_planes, out_planes // 2, 1, stride = stride, padding = 0, bias = False)
		self.conv2 = nn.Conv2d(in_planes, out_planes // 2, 1, stride = stride, padding = 0, bias = False)
		self.bn = nn.BatchNorm2d(out_planes, affine = affine)
		self.relu = nn.ReLU(inplace = True)

	def forward(self, x):
		#for example: a tensor [1,2,3,4]
		#conv1: process 1,3
		#conv2: process 2,4 
		#fully use all elements
		out = torch.cat([self.conv1(x), self.conv2(x[:,:,1:,1:])], dim = 1)
		out = self.bn(out)
		out = self.relu(out)

		return out

class AvgPool(nn.Module):
	def __init__(self, in_planes, out_planes, stride, double = False):

		super(AvgPool, self).__init__()
		self.stride = stride
		self.double = double
		self.avg_pool = nn.AvgPool2d(3, stride = stride, padding = 1, count_include_pad = False)
		if self.stride == 2 or double:
			self.conv1x1 = nn.Conv2d(in_planes, out_planes, kernel_size = 1, padding = 0, bias = False)

	def forward(self, x):
		x = self.avg_pool(x)
		if self.stride == 2 or self.double:
			x = self.conv1x1(x)
		return x

class MaxPool(nn.Module):
	def __init__(self, in_planes, out_planes, stride, double = False):

		super(MaxPool, self).__init__()
		self.stride = stride
		self.double = double
		self.max_pool = nn.MaxPool2d(3, stride = stride, padding = 1)
		if stride == 2 or double:
			self.conv1x1 = nn.Conv2d(in_planes, out_planes, kernel_size = 1, padding = 0, bias = False)

	def forward(self, x):
		x = self.max_pool(x)
		if self.stride == 2 or self.double:
			x = self.conv1x1(x)
		return x

def conv1x1(in_planes, out_planes, stride = 1, groups = 1):

	return nn.Conv2d(in_planes, out_planes, kernel_size = 1, stride = stride, padding = 0, groups = groups, bias = False)


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

# implement like HA_CNN 
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

# implement like HA-CNN
class ChannelAtten(nn.Module):

	def __init__(self, in_planes, reduction_rate = 15):

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

class Attention(nn.Module):
	# note that layer_size actually is no need 
	def __init__(self, in_planes, layer_size):
		super(Attention, self).__init__()
		self.spatial_atten = SpatialAtten()
		self.channel_atten = ChannelAtten(in_planes)
		self.conv1x1 = ConvBlock(in_planes, in_planes, 1)

	def forward(self, x):

		spatial_res = self.spatial_atten(x)
		channel_res = self.channel_atten(x)

		res = spatial_res * channel_res 
		res = torch.sigmoid(self.conv1x1(res))
		res = x * res

		return res 

class Attention_mine(nn.Module):

	def __init__(self, in_planes, layer_size):
		super(Attention, self).__init__()
	
		self.in_size = layer_size[0] * layer_size[1]
		self.layer_size = layer_size 
		self.in_planes = in_planes
		# for channel attention
		self.fc1 = conv1x1(in_planes, in_planes // 15)
		self.fc2 = conv1x1(in_planes // 15, in_planes)
		# for spatial attention
		self.fc3 = nn.Linear(self.in_size, self.in_size // 16)
		self.fc4 = nn.Linear(self.in_size // 16, self.in_size)

		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		## channel attention
		# squeeze
		s = x.mean([2,3], keepdim = True)
		s = self.relu(self.fc1(s))
		s = self.sigmoid(self.fc2(s))
		# excitation
		out1 = x * s 

		## spatial attention
		# squeeze
		s = x.mean([1], keepdim = True)
		# reshape
		s = s.view(-1, self.in_size)
		s = self.relu(self.fc3(s))
		s = self.sigmoid(self.fc4(s))
		# restore
		s = s.view(-1, 1, self.layer_size[0], self.layer_size[1])
		
		# excitation
		out2 = x * s 


		return out1 + out2

# seqeeze channel a little different from Attention module 
class ChannelAttention(nn.Module):

	def __init__(self, in_planes, out_planes,stride):
		super(ChannelAttention, self).__init__()

		self.stride = stride
		self.fc1 = conv1x1(in_planes, in_planes // 5)
		self.fc2 = conv1x1(in_planes // 5, in_planes)
		self.relu = nn.ReLU(inplace = True)
		self.sigmoid = nn.Sigmoid()
		# actually, we can directly cat the input to double channel
		if stride == 2:
			self.out = nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = self.stride, padding = 1, bias = False)

	def forward(self, x):

		# squeeze
		s = x.mean([2,3], keepdim = True)
		s = self.relu(self.fc1(s))
		s = self.sigmoid(self.fc2(s))
		# excitation
		out = x * s 
		if self.stride == 2:
			out = self.out(out)

		return out

class SpatialAttention(nn.Module):

	def __init__(self, in_planes, out_planes, init_size, stride):
		super(SpatialAttention, self).__init__()

		self.stride = stride
		self.in_size = init_size[0] * init_size[1]
		self.layer_size = init_size

		self.fc1 = nn.Linear(self.in_size, self.in_size // 16)
		self.fc2 = nn.Linear(self.in_size // 16, self.in_size)

		self.relu = nn.ReLU(inplace = True)
		self.sigmoid = nn.Sigmoid()

		if stride == 2:
			self.out = nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride, padding = 1, bias = False)

	def forward(self, x):

		# squeeze
		s = x.mean([1], keepdim = True)
		# reshape
		s = s.view(-1, self.in_size)
		s = self.relu(self.fc1(s))
		s = self.sigmoid(self.fc2(s))
		# restore
		s = s.view(-1, 1, self.layer_size[0], self.layer_size[1])
		
		out = x * s 

		if self.stride == 2:
			out = self.out(out)

		return out

class StdConv3x3(nn.Module):
	#self, in_planes, out_planes, kernel_size, stride, padding, affine = True
	def __init__(self, in_planes, out_planes, stride, padding = 1, affine = True):
		super(StdConv3x3, self).__init__()

		self.op = nn.Sequential(
			nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride, padding = padding),
			nn.BatchNorm2d(out_planes, affine = affine),
			nn.ReLU(inplace = True)
		)

	def forward(self, x):

		return self.op(x) 


# just like a key - function
# padding = (kernel_size - 1) / 2
# dil_5x5 = -> kernel_size = 5 x 2 -1 = 9 , thus padding = (9-1) / 2 = 4 
# affine 仿射，如果为true， 那个batchnorm 的 两个参数可以学习，否则为默认值
# note that each op with the same output channel as input channel, double channel operation was in preprocess 
OPS = {
	'sep_conv_3x3':lambda C_in, C_out, stride, double, affine: SepConv(C_in, C_out, 3, stride, 1, double, affine = affine),
	'sep_conv_5x5':lambda C_in, C_out, stride, double, affine: SepConv(C_in, C_out, 5, stride, 2, double, affine = affine),
	'dil_conv_3x3':lambda C_in, C_out, stride, double, affine: DilConv(C_in, C_out, 3, stride, 2, 2, double, affine = affine),
	'dil_conv_5x5':lambda C_in, C_out, stride, double, affine: DilConv(C_in, C_out, 5, stride, 4, 2, double, affine = affine),
	'avg_pool_3x3':lambda C_in, C_out, stride, double, affine: AvgPool(C_in, C_out, stride, double),
	'max_pool_3x3':lambda C_in, C_out, stride, double, affine: MaxPool(C_in, C_out, stride, double),
	'skip_connect':lambda C_in, C_out, stride, double, affine: Identity(C_in, C_out, double) if stride == 1 else FactorizedReduce(C_in, C_out, stride, affine = affine),
	# 'zeros':lambda C_in, C_out, stride, affine: Zero(stride, C_out),
	'conv_3x3': lambda C_in, C_out, stride, affine: StdConv3x3(C_in, C_out, stride, affine = affine)
}

FOPS = {
	'sep_conv_3x3':lambda C_in, C_out, stride, layer_size, affine: SepConv(C_in, C_out, 3, stride, 1, affine = affine),
	'sep_conv_5x5':lambda C_in, C_out, stride, layer_size, affine: SepConv(C_in, C_out, 5, stride, 2, affine = affine),
	'dil_conv_3x3':lambda C_in, C_out, stride, layer_size, affine: DilConv(C_in, C_out, 3, stride, 2, 2, affine = affine),
	'dil_conv_5x5':lambda C_in, C_out, stride, layer_size, affine: DilConv(C_in, C_out, 5, stride, 4, 2, affine = affine),
	'avg_pool_3x3':lambda C_in, C_out, stride, layer_size, affine: AvgPool(C_in, C_out, stride),
	'max_pool_3x3':lambda C_in, C_out, stride, layer_size, affine: MaxPool(C_in, C_out, stride),
	'skip_connect':lambda C_in, C_out, stride, layer_size, affine: Identity() if stride == 1 else FactorizedReduce(C_in, C_out, affine = affine),
	# 'zeros':lambda C_in, C_out, stride, layer_size, affine: Zero(stride, C_out),
	'channel_atten': lambda C_in, C_out, stride, layer_size, affine: ChannelAttention(C_in, C_out, stride),
	"spacial_atten": lambda C_in, C_out, stride, layer_size, affine: SpatialAttention(C_in, C_out, layer_size, stride),
	'conv_3x3': lambda C_in, C_out, stride, affine: StdConv3x3(C_in, C_out, stride, affine = affine)
}

if __name__ == "__main__":

	# model = ConvBNReLU(1,3)
	# model = SepConv(1,2,1,1,1)
	# model = DilConv(1,2,1,1,1,2)
	# model = Zero(2)
	# model = FactorizedReduce(2,4)
	# model = OPS['zeros'](2,4,1,False)
	# model = ChannelAttention(30,2)
	# model = SpatialAttention(30, 60, [128,64], 2)

	tensor = torch.randn(1,30,128,64)
	model = StdConv3x3(30,60,2, affine = True)
	x = model(tensor)
	print(x.size())