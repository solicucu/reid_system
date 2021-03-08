# -*- coding: utf-8 -*-
# @Author: solicucu
# @E-mail: 2748847613@qq.com
# @Date:   2020-03-22 22:59:49
# @Last Modified time: 2020-03-23 23:23:49

# specify the path to the code
# root = "D:/project/Paper/papercode/myReID/MobileNetReID" # for pc
root = "/home/hanjun/solicucu/ReID/HAReID"  # for server
import os
import sys 
import argparse
import logging
import time
import openpyxl as xl 
import torch
import torch.nn as nn  
import torch.nn.functional as F 

sys.path.append(root)

from config import cfg  
from utils import setup_logger
from torch.backends import cudnn
from data import imagenet_make_data_loader 
from model import build_model 
from optims import make_optimizer, make_lr_scheduler
from utils.metrics import * 
from utils.utils import * 


def train():
	
	# 1、make dataloader
	train_loader, val_loader,  num_class = imagenet_make_data_loader(cfg)
	#print("num_query:{},num_class:{}".format(num_query,num_class))

	# 2、make model
	model = build_model(cfg, num_class)

	# 3、 make optimizer
	optimizer = make_optimizer(cfg, model)

	# 4、 make lr_scheduler
	scheduler = make_lr_scheduler(cfg, optimizer)

	# 5、 make loss_func
	# directly use F.cross_entropy 

	# get paramters
	log_period = cfg.OUTPUT.LOG_PERIOD 
	ckpt_period =cfg.OUTPUT.CHECKPOINT_PERIOD
	eval_period = cfg.OUTPUT.EVAL_PERIOD
	output_dir = cfg.OUTPUT.ROOT_DIR
	device = cfg.MODEL.DEVICE
	epochs = cfg.SOLVER.MAX_EPOCHS
	use_gpu = device == "cuda"
	
	# how many batch for each log
	batch_size = cfg.SOLVER.IMGS_PER_BATCH
	dataset = train_loader.dataset 
	
	batch_num = len(dataset) // batch_size
	print("batch number: ",batch_num)
	
	log_iters = batch_num // log_period
	# print(log_iters)
	# exit(1)
	
	pretrained = cfg.MODEL.PRETRAIN_PATH != ''
	parallel = cfg.MODEL.PARALLEL 	

	
	ckpt_save_path = cfg.OUTPUT.ROOT_DIR + cfg.OUTPUT.CKPT_DIR
	if not os.path.exists(ckpt_save_path):
		os.makedirs(ckpt_save_path)

	
	logger = logging.getLogger('MobileNetReID.train')
	
	# count parameter
	size = count_parameters(model)
	logger.info("the param number of the model is {:.2f} M".format(size))
	
	

	logger.info("Start training")
	
	#count = 183, x, y = batch -> 11712 for train
	if pretrained:
		start_epoch = model.start_epoch

	if parallel:
		model = nn.DataParallel(model)

	if use_gpu:
		# model = nn.DataParallel(model)
		model.to(device)

	is_best = False
	best_acc = 0.
	# batch : img, pid, camid, img_path
	avg_loss, avg_acc = RunningAverageMeter(), RunningAverageMeter()
	avg_time, global_avg_time = AverageMeter(), AverageMeter()
	global_avg_time.reset()
	for epoch in range(epochs):
		scheduler.step()

		if pretrained and epoch < start_epoch - 1:
			continue
	
		model.train()
		# sum_loss, sum_acc = 0., 0.
		avg_loss.reset()
		avg_acc.reset()
		avg_time.reset()
		for i, batch in enumerate(train_loader):
			
			t0 = time.time()
			imgs,labels = batch

			if use_gpu:
				imgs = imgs.to(device)
				labels = labels.to(device)

			scores = model(imgs)
			loss = F.cross_entropy(scores, labels)
			
			loss.backward()
			
			optimizer.step()

			optimizer.zero_grad()

			acc = (scores.max(1)[1] == labels).float().mean()

			t1 = time.time()
			avg_time.update((t1 - t0) / batch_size)
			avg_loss.update(loss)
			avg_acc.update(acc)

			#log the info 
			if (i+1) % log_iters == 0:

				logger.info("epoch {}: {}/{} with loss is {:.5f} and acc is {:.3f}".format(
					         epoch+1, i+1, batch_num, avg_loss.avg, avg_acc.avg))

		lr = optimizer.state_dict()['param_groups'][0]['lr']
		logger.info("end epochs {}/{} with lr: {:.5f} and avg_time is {:.3f} ms".format(epoch+1, epochs, lr, avg_time.avg * 1000))
		global_avg_time.update(avg_time.avg)
		# change the lr 

		# eval the model 
		if (epoch+1) % eval_period == 0 or (epoch + 1) == epochs :
			
			model.eval()
			
			val_acc = RunningAverageMeter()
			with torch.no_grad():

				for vi, batch in enumerate(val_loader):
					
					imgs, labels = batch

					if use_gpu:
						imgs = imgs.to(device)
						labels = labels.to(device)

					scores = model(imgs)
					acc = (scores.max(1)[1] == labels).float().mean()
					val_acc.update(acc)

				
				logger.info("validation results at epoch:{}".format(epoch + 1))
				logger.info("acc:{:.2%}".format(val_acc.avg))

				if val_acc.avg > best_acc:
					logger.info("get a new best acc")
					is_best = True 

				


		# we hope that eval_period == ckpt_period or eval_period == k* ckpt_period where k is int			
		# whether to save the model
		if (epoch+1) % ckpt_period == 0 or is_best:

			if parallel:
				torch.save(model.module.state_dict(), ckpt_save_path + "checkpoint_{}.pth".format(epoch + 1 ))
			else:
				torch.save(model.state_dict(), ckpt_save_path +  "checkpoint_{}.pth".format(epoch + 1 ))

			logger.info("checkpoint {} saved !".format(epoch + 1))

			if is_best:
				if parallel:
					torch.save(model.module.state_dict(), ckpt_save_path + "best_ckpt.pth")
				else:
					torch.save(model.state_dict(), ckpt_save_path + "best_ckpt.pth")
				logger.info("best checkpoint was saved")
				is_best = False
		
	
	
	logger.info("training is end, time for per imgs is {} ms".format(global_avg_time.avg *1000))


		

def parse_config():
	# create the parser
	parser = argparse.ArgumentParser(description = "MobileNetReID baseline")

	parser.add_argument("--config_file", default = '', help = "path to specified config file", type = str)
	
	#remainder parameters in a list
	parser.add_argument("opts", default = None, help = 'modify some value for the config file in command line', nargs = argparse.REMAINDER)

	args = parser.parse_args()

	if args.config_file != "":
		# use config file to update the default config value
		cfg.merge_from_file(args.config_file)
	# note that opts is a list
	cfg.merge_from_list(args.opts)
	# cfg.freeze()

	output_dir = cfg.OUTPUT.ROOT_DIR
	if output_dir != "":
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)
	else:
		print("ERROR:please specify an output path")
		exit(1)

	#config the logger
	logger = setup_logger("MobileNetReID",output_dir,0,cfg.OUTPUT.LOG_NAME)

	use_gpu = cfg.MODEL.DEVICE == 'cuda'

	if use_gpu:
		logger.info("Train with GPU: {}".format(cfg.MODEL.DEVICE_ID))
	else:
		logger.info("Train with CPU")
	#print the all arguments
	logger.info(args)
	#read the config file
	if args.config_file != "":
		logger.info("load configuration file {}".format(args.config_file))
		"""
		with open(args.config_file,'r') as cf:
			strs = '\n' + cf.read()
			logger.info(strs)
		"""
	#config after update by config file	
	logger.info("runing with config:\n{}".format(cfg))

	if use_gpu:
		os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID
	#this setup will facilitate the training
	cudnn.benchmark = True


def main():
	print("solicucu")
	parse_config()
	train()



if __name__ == '__main__':

	main()
