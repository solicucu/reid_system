# -*- coding: utf-8 -*-
# @Author: solicucu
# @E-mail: 2748847613@qq.com
# @Date:   2020-01-09 21:06:27
# @Last Modified time: 2020-03-28 00:06:18

from .market1501 import Market1501 
from .dukemtcm import DukeMTMC
from .imageDataset import ImageDataset 

datasets = {
	
	'market1501': Market1501,
	'dukemtmc': DukeMTMC
}

def get_dataset_names():

	return datasets.keys()


def init_dataset(name, *args, **kwargs):

	if name not in datasets.keys():
		raise KeyError("Unknow dataset：{}".format(name))

	return datasets[name](*args, **kwargs)
