# -*- coding: utf-8 -*-
# @Author: solicucu
# @E-mail: 2748847613@qq.com
# @Date:   2020-02-03 22:50:23
# @Last Modified time: 2020-02-26 18:12:34
import numpy as np
import json
import torch 
import copy 
import re 

# from darts import *  # op_names 

"""
BatchNorm2d: has weight and bias -> default the affine = True
"""
def count_parameters(model):
	# see the param according the model 
	# print(model)
	# for name, param in model.named_parameters():
	# 	print(name, param.size())
		
	return np.sum(np.prod(param.size()) for param in model.parameters()) / 1e6

op_names = [
	'sep_conv_3x3',
	'sep_conv_5x5',
	'dil_conv_3x3',
	'dil_conv_5x5',
	'avg_pool_3x3',
	'max_pool_3x3',
	'skip_connect',
	'zeros'
]
# for simple_attensnet
def change_state_dict(ckpt, genotype, layers):
	print("change the state_dict")
	with open(genotype, 'r') as f:
		geno = json.load(f)

	state_dict = torch.load(ckpt)
	# print(len(geno))
	ops_names = {}
	#cell.1.branch1 : ops.2  when select the op dil_conv_3x3
	k = -1
	for i, num in enumerate(layers):
		name = 'layer{}'.format(i+1)
		ops = geno[name]
		# cell 
		for _ in range(num):
			k += 1
			# branch 
			for j in range(1,4):
				key = 'cells.{}.branch{}'.format(k, j)
				value = 'ops.{}'.format(op_names.index(ops[j-1]))
				ops_names[key] = value
	# cell.0.branch1 ops.3
	# for key in ops_names:
	# 	print(key, ops_names[key])
	# -> cells.0.branch1.ops.3.op.1.weight remove .ops.3
	new_state_dict = {}
	for name, value in state_dict.items():
		# process name with branch
		if 'branch' in name:
			pos = name.find('.ops')
			key = name[:pos]
			op = ops_names[key] # get the select operation index
			if op in name:
				new_key = key + name[pos + 6:]
				new_state_dict[new_key] = value

		else:
			if 'classifier' in name:
				continue
			new_state_dict[name] = value

	s1, s2 = ckpt.split('.')
	s1 += '_fit.'
	save_path = s1 + s2 
	torch.save(new_state_dict, save_path)
	
"""
cells.0.ops.1.op.0.weight == cells.0.ops.0.ops.2.op.0.weight
 
from cells.y.ops.0.x to cells.y.ops.5.x 
	construct the latter key cells.y.ops.edge.ops.op_order.x 
append the x 

"""

def full_change_state_dict(model, ckpt, genotype, layers):
	print("change the state dict for full search network")
	
	state_dict = torch.load(ckpt) 
	new_state_dict = model.state_dict()
	with open(genotype, 'r') as f:
		geno = json.load(f)

	num_layer = {}
	k = -1
	for i, num in enumerate(layers):
		layer = 'layer' + str(i+1)
		for _ in range(num):
			k += 1 # cell number
			num_layer[k] = layer

	# strs = "cells.1.ops.0.op.0.weight"
	#                             cell_num,op_order, rest str
	pattern = re.compile(r'cells.(\d+).ops.(\d+).(.*)')
	
	# consturct a dict cells.x. : layeri
	for new_key in new_state_dict:
		if 'ops' in new_key:
			cell_num, op_order, rest = pattern.search(new_key).groups()
			# get the edge and op_order
			layer = num_layer[int(cell_num)]
			edge, order = geno['edge_ops'][layer][int(op_order)]

			# cells.y.ops.edge.ops.op_order.x
			key = 'cells.{}.ops.{}.ops.{}.{}'.format(cell_num, edge, order, rest)

			new_state_dict[new_key].data.copy_(state_dict[key].data)

		else:
			# directly copy the data 
			# print(key)
			new_state_dict[new_key].data.copy_(state_dict[new_key].data)	

	s1, s2 = ckpt.split('.')
	s1 += '_fit.'
	save_path = s1 + s2 
	torch.save(new_state_dict, save_path)

if __name__ == "__main__":

	ckpt_path = 'D:/project/data/ReID/MobileNetReID/darts/checkpoints/darts2/fsnet/best_ckpt.pth'
	genotype_path = 'D:/project/data/ReID/MobileNetReID/darts/checkpoints/darts2/fsnet/best_genotype.json'
	layers = [2,2,2,2]
	full_change_state_dict(ckpt_path, genotype_path, layers)