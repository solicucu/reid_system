# -*- coding: utf-8 -*-
# @Author: solicucu
# @E-mail: 2748847613@qq.com
# @Date:   2020-02-14 13:40:00
# @Last Modified time: 2020-02-14 14:00:48

import torch
from torch.optim import lr_scheduler

def darts_make_optimizer(cfg, model):

	if cfg.optimizer_name == 'SGD':

		optimizer = getattr(torch.optim, 'SGD')(model.parameters(), lr = cfg.base_lr, momentum = cfg.momentum, weight_decay = cfg.weight_decay )
	
	else:

		raise NotImplementedError("no such optimizer: {}".format(cfg.optimizer_name))

	return optimizer

def darts_make_lr_scheduler(cfg, optimizer):

	name = cfg.lr_scheduler_name

	if name == 'StepLR':

		scheduler = lr_scheduler.StepLR(optimizer, step_size = cfg.lr_decay_period, gamma = cfg.lr_decay_factor)

	elif name == "CosineAnnealingLR":
		scheduler = lr_scheduler.CosineAnnealingLR(optimizer, float(cfg.max_epochs), eta_min = cfg.lr_min )

	else:
		raise NotImplementedError("no such lr scheduler:{}".format(name))

	return scheduler