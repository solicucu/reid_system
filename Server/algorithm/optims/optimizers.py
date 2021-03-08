# -*- coding: utf-8 -*-
# @Author: solicucu
# @E-mail: 2748847613@qq.com
# @Date:   2020-01-12 20:42:22
# @Last Modified time: 2020-03-21 00:35:03
import numpy as np 
import torch
import torch.nn as nn 
from torch.optim import lr_scheduler

def make_optimizer(cfg, model):

	if cfg.SOLVER.OPTIMIZER_NAME == "SGD":

		optimizer = getattr(torch.optim, "SGD")(model.parameters(), lr = cfg.SOLVER.BASE_LR, momentum = cfg.SOLVER.MOMENTUM, weight_decay = cfg.SOLVER.WEIGHT_DECAY)
	else:

		optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(model.parameters(), lr = cfg.SOLVER.BASE_LR, weight_decay = cfg.SOLVER.WEIGHT_DECAY)

	return optimizer

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):

	def __init__(self, optimizer, milestones = [40, 70], lr_list = None, gama = 0.1 , warmup_factor = 0.1,
				warmup_iters = 10, warmup_method = 'linear', last_epoch = -1):

		self.milestones = milestones 
		self.lr_list = lr_list
		self.gama = gama 
		self.warmup_factor = warmup_factor  
		self.warmup_iters = warmup_iters 
		self.warmup_method = warmup_method

		super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

	def get_lr(self):
		# self.last_epoch is current epoch
		if self.last_epoch < self.warmup_iters:
			if self.warmup_method == 'linear':
				cur_iter = self.last_epoch + 1
				# lr = init_lr * gama * t, where init_lr = base_lr * warmup_factor
				lr = self.base_lrs[0] * self.warmup_factor * self.gama * cur_iter
				return [lr]
			else:
				raise NotImplementedError("not know such warmup method {}".format(self.warmup_method) )

		elif self.last_epoch < self.milestones[0]:
			return [self.lr_list[0]]
		elif self.last_epoch < self.milestones[1]:
			return [self.lr_list[1]]
		else:
			return [self.lr_list[2]]


# torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=-1)
class WarmupCosAnnLR(torch.optim.lr_scheduler._LRScheduler):
	# T_max is END_EPOCH
	def __init__(self, optimizer, T_max, eta_min, last_epoch = -1, gama = 0.1, warmup_factor = 0.1,
				warmup_iters = 10, warmup_method = 'linear'):
		
		self.eta_min = eta_min
		self.T_max = T_max - warmup_iters
		self.gama = gama 
		self.warmup_factor = warmup_factor  
		self.warmup_iters = warmup_iters 
		self.warmup_method = warmup_method
		
		super(WarmupCosAnnLR, self).__init__(optimizer, last_epoch)
		

	def get_lr(self):

		eta_max_sub_min = self.base_lrs[0] - self.eta_min

		if self.last_epoch < self.warmup_iters:
			if self.warmup_method == 'linear':

				cur_iter = self.last_epoch + 1
				# lr = init_lr * gama * t, where init_lr = base_lr * warmup_factor
				lr = self.base_lrs[0] * self.warmup_factor * self.gama * cur_iter
				return [lr] 
			else:
				raise NotImplementedError("not know warmup method {}".format(self.warmup_method))
				               # (END_EPOCH)
		elif self.last_epoch <= (self.T_max + self.warmup_iters):
			cur_iter = self.last_epoch - self.warmup_iters
			lr = self.eta_min + 0.5 * eta_max_sub_min * (1 + np.cos((cur_iter / self.T_max) * np.pi))
			return [lr]
		# rest epoch 
		else:

			return [self.eta_min * 0.1]


def make_lr_scheduler(cfg, optimizer):

	name = cfg.SOLVER.LR_SCHEDULER_NAME
	if  name == "StepLR":

		scheduler = lr_scheduler.StepLR(optimizer, step_size = cfg.SOLVER.LR_DECAY_PERIOD, gamma = cfg.SOLVER.LR_DECAY_FACTOR)
	
	elif name == "CosineAnnealingLR":

		scheduler = lr_scheduler.CosineAnnealingLR(optimizer, float(cfg.SOLVER.MAX_EPOCHS), eta_min = cfg.SOLVER.LR_MIN)

	elif name == "WarmupMultiStepLR":

		scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.MILESTONES, cfg.SOLVER.LR_LIST, cfg.SOLVER.GAMA, cfg.SOLVER.WARMUP_FACTOR, cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)

	elif name == "WarmupCosAnnLR":

		scheduler = WarmupCosAnnLR(optimizer, cfg.SOLVER.END_EPOCH, cfg.SOLVER.LR_MIN, gama = cfg.SOLVER.GAMA, warmup_factor = cfg.SOLVER.WARMUP_FACTOR,
					warmup_iters = cfg.SOLVER.WARMUP_ITERS, warmup_method = cfg.SOLVER.WARMUP_METHOD)
		
	else:

		raise RuntimeError(" name {} is not know".format(name))

	return scheduler 	

# use for testing
class Network(nn.Module):

	def __init__(self, in_planes, out_planes):

		super(Network, self).__init__()
		self.conv = nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = 2, padding = 1, bias = False)

	def forward(self, x):

		return self.conv(x)


if __name__ == "__main__":

	model = Network(3, 30)
	tensor = torch.randn(1,3,10,10)
	optimizer = torch.optim.SGD(model.parameters(), lr = 3e-4, momentum = 0.9)

	res = model(tensor)
	print(res.size())
	scheduler = WarmupMultiStepLR(optimizer, warmup_iters = 0)

	for i in range(120):
		lr = scheduler.get_lr()[0]
		#print(i, lr)
		scheduler.step()
	