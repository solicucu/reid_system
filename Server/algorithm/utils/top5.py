# -*- coding: utf-8 -*-
# @Author: solicucu
# @E-mail: 2748847613@qq.com
# @Date:   2020-05-09 21:55:45
# @Last Modified time: 2020-05-10 14:37:54

import torch 
import numpy as np 
import os 
from main import produce_features 

class Top5(object):
	"""
	query: a 4-elem list, [imgs, pid, camid, imgpath]
	username: the name of current user, used to position datasets and features
	"""
	def __init__(self, username, data_root):

		self.username = username
		self.data_root = data_root 

	
	def set_gallery(self, dataset, model_name):
		# if name was change, update 
		self.dataset = dataset 
		self.model_name = model_name

		# load the gallery features 
		feats_path = self.data_root + "{}/features/{}_{}.feats".format(self.username, model_name, dataset)

		if not os.path.exists(feats_path):
			print("{} is not found".format(feats_path))
			# if file is not existed, produce now
			produce_features(self.username, dataset)

		data = torch.load(feats_path)
		self.gf = data["feats"]
		self.gpid = np.asarray(data["pids"])
		self.gcamid = np.asarray(data["camids"])
		self.gpath = np.asarray(data["paths"])
		# normalize the feature 
		self.gf = torch.nn.functional.normalize(self.gf, dim = 1, p = 2)


	def set_query(self, query):

		# extract query image info
		self.qf = query[0]
		self.qpid = np.asarray(query[1]) 
		self.qcamid = np.asarray(query[2])
		self.qpath = np.asarray(query[3])
		# normalize the feature 
		self.qf = torch.nn.functional.normalize(self.qf, dim = 1, p = 2)

	# compute the euclidiean distance
	def compute(self):
		# whether ignore the image catch by the same camid
		diffcamid = True
		gf = self.gf 
		qf = self.qf 
		m,n = qf.shape[0], gf.shape[0]

		# compute the distance x^2 + y^2
		dist_mat = torch.pow(qf, 2).sum(dim = 1, keepdim = True).expand(m, n) + \
				   torch.pow(gf, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
		# gf-> m * n
		# qf-> n * m -> t()

		# - 2 x y
		dist_mat.addmm_(1, -2, qf, gf.t()) 
		dist_mat = dist_mat.detach().numpy()

		# sort by the distance
		indices = np.argsort(dist_mat, axis = 1)[0]
		matches = (self.gpid[indices] == self.qpid[:, np.newaxis]).astype(np.int32)[0]
		
		paths = self.gpath[indices]
		# because indices is 2-dims, so path with becomes 2-dims
		# set the result
		if diffcamid:
			remove = (self.gpid[indices] == self.qpid[0]) & (self.gcamid[indices] == self.qcamid[0])
			keep = np.invert(remove)
			matches = matches[keep]
			paths = paths[keep]
			

		result = {}
		result["paths"] = paths[:5]
		result["matches"] = matches[:5]

		return result 




