# -*- coding: utf-8 -*-
# @Author: solicucu
# @E-mail: 2748847613@qq.com
# @Date:   2020-02-18 16:55:49
# @Last Modified time: 2020-02-20 16:25:10

import torch 
import torch.nn as nn 
import os 

device = 'cuda'
device_ids = '0' # str must in range of true ids
# note devices: CUDA_VISIBLE_DEVICES must has S , or all device are visible
os.environ['CUDA_VISIBLE_DEVICES'] = device_ids

# network.cuda do not place its variable on cuda
class Network(nn.Module):
	def __init__(self,in_planes, out_planes, use_gpu):
		super(Network, self).__init__()

		self.conv = nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = 1, padding = 1)
		self._init_alpha(use_gpu)

	def _init_alpha(self, gpu):

		if gpu:
			self.alpha = torch.randn(1).cuda().requires_grad_()

		else:
			self.alpha = torch.randn(1).requires_grad_()

	def forward(self, x):
		print("x",x.device)
		# print("self_alpha",id(self.alpha))
		# change the alpha.device same as x.device
		alpha = self.alpha.clone().to(x.device)

		# alpha = self.alpha.new_tensor(self.alpha.data, device = x.device)
		# print("copy_alpha",id(alpha))
		print("copy_alpha",alpha.device)
		print(alpha.requires_grad)
		x = self.conv(x)

		return x * alpha


def test_cuda():

	tensor = torch.randn(2,2)
	print(tensor.device)
	# tensor.to(device) # must reassign or is cpu
	tensor_cuda = tensor.to(device) # to cuda:0
	print(tensor_cuda.device)
	tensor_cuda = tensor.cuda() # the result is the index of device_ids
	# tensor.cuda() # muslt reassign or is cpu 
	print(tensor_cuda.device)

def test_network_cuda():

	model = Network(32,32, device == "cuda")
	if device == 'cuda':
		model = nn.DataParallel(model)
		model = model.cuda()
	imgs = torch.randn(32,32,5,5)

	if device == 'cuda':
		imgs = imgs.cuda()
	print("imgs",imgs.device)
	res = model(imgs)

test_network_cuda()
# test_cuda()
