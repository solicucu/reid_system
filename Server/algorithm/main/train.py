# -*- coding: utf-8 -*-
# @Author: solicucu
# @E-mail: 2748847613@qq.com
# @Date:   2020-01-08 13:05:48
# @Last Modified time: 2020-04-30 09:15:00

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
sys.path.append(root)

from config import cfg  
from utils import setup_logger,get_image,R1_mAP 
from torch.backends import cudnn
from data import make_data_loader, build_transforms
from model import build_model 
from optims import make_optimizer, make_lr_scheduler, make_loss
from utils.metrics import * 
from utils.utils import * 

# return a list with softmax_fn and triplet_fn 
def get_softmax_triplet_loss_fn(cfg, num_class):

	cfg.SOLVER.LOSS_NAME = 'softmax'
	softmax_fn = make_loss(cfg, num_class)
	cfg.SOLVER.LOSS_NAME = 'triplet'
	triplet_fn = make_loss(cfg, num_class)

	return [softmax_fn, triplet_fn]

# we can use the global cfg, no need to pass as paramter
def compute_loss_acc(use_neck, res, labels, loss_fn):

	if use_neck:
		score1, feat1 = res[0]
		score2, feat2 = res[1]
		loss_g = loss_fn(score1, feat1, labels)
		loss_p = loss_fn(score2, feat2, labels)
		loss = loss_g + loss_p

		acc_g = (score1.max(1)[1] == labels).float().mean()
		acc_p = (score2.max(1)[1] == labels).float().mean()

		acc = (acc_g + acc_p) / 2.

		return loss, acc 
	# for pcb 
	elif type(loss_fn) == list:
		# pcb 
		scores, feat = res[0], res[1]
		part_num = len(scores)
		softmax_fn, triplet_fn = loss_fn
		softmax_loss = [softmax_fn(score, None, labels) for score in scores]
		softmax_loss_sum = sum(softmax_loss)

		# triplet loss
		triplet_loss = triplet_fn(None, feat, labels)

		loss = softmax_loss_sum + triplet_loss

		acc = [(score.max(1)[1] == labels).float().mean() for score in scores]
		avg_acc = sum(acc) / part_num 

		return loss, avg_acc 

	else:
		score, feat = res
		loss = loss_fn(score, feat, labels)
		acc = (score.max(1)[1] == labels).float().mean()

		return loss, acc 


def train():
	"""
	# get an image for test the model 
	train_transform = build_transforms(cfg, is_train = True)
	imgs = get_image("1.jpg")
	img_tensor = train_transform(imgs[0])
	# c,h,w = img_tensor.shape
	# img_tensor = img_tensor.view(-1,c,h,w)
	# add an axis
	img_tensor = img_tensor.unsqueeze(0)
	"""
	# 1、make dataloader
	train_loader, val_loader, num_query, num_class = make_data_loader(cfg)
	#print("num_query:{},num_class:{}".format(num_query,num_class))

	# 2、make model
	model = build_model(cfg, num_class)

	# model.eval()
	# x = model(img_tensor)
	# print(x.shape)
	# 3、 make optimizer
	optimizer = make_optimizer(cfg, model)

	# 4、 make lr_scheduler
	scheduler = make_lr_scheduler(cfg, optimizer)

	# 5、 make loss_func
	if cfg.MODEL.PCB_NECK:
		# make loss specificially for pcb 
		loss_func = get_softmax_triplet_loss_fn(cfg, num_class)
	else:
		loss_func = make_loss(cfg, num_class)

	# get paramters
	log_period = cfg.OUTPUT.LOG_PERIOD 
	ckpt_period =cfg.OUTPUT.CHECKPOINT_PERIOD
	eval_period = cfg.OUTPUT.EVAL_PERIOD
	output_dir = cfg.OUTPUT.ROOT_DIR
	device = cfg.MODEL.DEVICE
	epochs = cfg.SOLVER.MAX_EPOCHS
	use_gpu = device == "cuda"
	use_neck = cfg.MODEL.NECK or cfg.MODEL.LEARN_REGION 
	# how many batch for each log
	batch_size = cfg.SOLVER.IMGS_PER_BATCH
	batch_num = len(train_loader) 
	
	log_iters = batch_num // log_period
	pretrained = cfg.MODEL.PRETRAIN_PATH != ''
	parallel = cfg.MODEL.PARALLEL 	
	grad_clip = cfg.DARTS.GRAD_CLIP 

	feat_norm = cfg.TEST.FEAT_NORM 
	ckpt_save_path = cfg.OUTPUT.ROOT_DIR + cfg.OUTPUT.CKPT_DIR
	if not os.path.exists(ckpt_save_path):
		os.makedirs(ckpt_save_path)


	# create *_result.xlsx
	# save the result for analyze
	name = (cfg.OUTPUT.LOG_NAME).split(".")[0] + ".xlsx"
	result_path = cfg.OUTPUT.ROOT_DIR + name

	wb = xl.Workbook()
	sheet = wb.worksheets[0]
	titles = ['size/M','speed/ms','final_planes', 'acc', 'mAP', 'r1', 'r5', 'r10', 'loss',
			  'acc', 'mAP', 'r1', 'r5', 'r10', 'loss','acc', 'mAP', 'r1', 'r5', 'r10', 'loss']
	sheet.append(titles)
	check_epochs = [40, 80, 120, 160, 200, 240, 280, 320, 360, epochs]
	values = []

	logger = logging.getLogger('MobileNetReID.train')
	
	# count parameter
	size = count_parameters(model)
	logger.info("the param number of the model is {:.2f} M".format(size))
	
	values.append(format(size, '.2f'))
	values.append(model.final_planes)

	logger.info("Start training")
	
	#count = 183, x, y = batch -> 11712 for train
	if pretrained:
		start_epoch = model.start_epoch

	if parallel:
		model = nn.DataParallel(model)

	if use_gpu:
		# model = nn.DataParallel(model)
		model.to(device)
	
	# save the best model
	best_mAP, best_r1 = 0., 0.
	is_best = False
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

			res = model(imgs)
			# score, feat = model(imgs)
			# loss = loss_func(score, feat, labels)
			loss, acc = compute_loss_acc(use_neck, res, labels, loss_func)
			
			loss.backward()
			if grad_clip != 0:
				nn.utils.clip_grad_norm(model.parameters(), grad_clip)

			optimizer.step()

			optimizer.zero_grad()

			# acc = (score.max(1)[1] == labels).float().mean()

			# sum_loss += loss
			# sum_acc += acc 
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
			metrics = R1_mAP(num_query, use_gpu = use_gpu, feat_norm = feat_norm)

			with torch.no_grad():

				for vi, batch in enumerate(val_loader):
					
					imgs, labels, camids = batch

					if use_gpu:
						imgs = imgs.to(device)

					feats = model(imgs)
					metrics.update((feats,labels, camids))

				#compute cmc and mAP
				cmc, mAP = metrics.compute()
				logger.info("validation results at epoch:{}".format(epoch + 1))
				logger.info("mAP:{:.2%}".format(mAP))
				for r in [1,5,10]:
					logger.info("CMC curve, Rank-{:<3}:{:.2%}".format(r,cmc[r-1]))	

				# determine whether cur model is the best 
				if mAP > best_mAP:
					is_best = True
					best_mAP = mAP
					logger.info("Get a new best mAP")
				if cmc[0] > best_r1:
					is_best = True
					best_r1 = cmc[0]
					logger.info("Get a new best r1")

				# add the result to sheet
				if (epoch + 1) in check_epochs:
					val = [avg_acc.avg, mAP, cmc[0], cmc[4], cmc[9]]
					change = [format(v * 100, '.2f') for v in val]
					change.append(format(avg_loss.avg, '.3f'))
					values.extend(change)


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
	
	values.insert(1, format(global_avg_time.avg * 1000, '.2f'))
	sheet.append(values)
	wb.save(result_path)

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
