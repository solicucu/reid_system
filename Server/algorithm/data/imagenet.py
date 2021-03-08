# -*- coding: utf-8 -*-
# @Author: solicucu
# @E-mail: 2748847613@qq.com
# @Date:   2020-03-21 23:20:30
# @Last Modified time: 2020-03-23 23:30:45

import glob 
import os 
import torch 
import torch.nn as nn 
from torchvision import datasets, transforms 
from torch.utils.data import DataLoader

types = ["ILSVRC2012_train", "ILSVRC2012_val", "ILSVRC2012_test"]

path = "/home/share/tanli/imagenet/"
save_path = "/home/share/solicucu/data/ReID/MobileNetReID/imagenet/"

# data = "/home/share/solicucu/data/hymenoptera/"
data = '/home/share/tanli/imagenet/train_val/'

if not os.path.exists(save_path):
	os.makedirs(save_path)

def get_name_list():

	for name in types:
		dirs = path + name 
		# img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
		# name_list = glob.glob(".*")
		name_list = os.listdir(dirs)

		lens = len(name_list)
		with open(save_path + name+'.txt', 'w') as f:
			for n in name_list:
				f.write(n + "\n")

			f.write(str(lens))

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
# batch_size:64 ~20000 batch  need 3 mins to load the data 
def make_data_loader(cfg):

	batch_size = cfg.SOLVER.IMGS_PER_BATCH 
	num_workers = cfg.DATALOADER.NUM_WORKERS 

	train_set = datasets.ImageFolder(os.path.join(data, 'train'), data_transforms['train'])
	val_set = datasets.ImageFolder(os.path.join(data, 'val'), data_transforms['val'])
	
	train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = num_workers)
	val_loader = DataLoader(val_set, batch_size = batch_size, shuffle = True, num_workers = num_workers)

	
	return train_loader, val_loader, 1000


if __name__ == "__main__":

	# get_name_list()
	make_data_loader()

