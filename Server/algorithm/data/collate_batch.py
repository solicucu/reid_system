# -*- coding: utf-8 -*-
# @Author: solicucu
# @E-mail: 2748847613@qq.com
# @Date:   2020-01-10 16:10:49
# @Last Modified time: 2020-01-10 16:15:29

import torch

# recieve a batch list, change it to torch tensor
def train_collate_fn(batch):

	imgs, pids, _, _ = zip(*batch)
	pids = torch.tensor(pids, dtype = torch.int64)
	return torch.stack(imgs, dim = 0), pids 

def val_collate_fn(batch):

	imgs, pids, camids, _ = zip(*batch)
	return torch.stack(imgs, dim = 0), pids, camids