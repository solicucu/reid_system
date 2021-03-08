# -*- coding: utf-8 -*-
# @Author: solicucu
# @E-mail: 2748847613@qq.com
# @Date:   2020-01-29 17:11:22
# @Last Modified time: 2020-03-13 22:47:27

# root = "D:/project/Paper/papercode/myReID/MobileNetReID" #fpr pc
root = "/home/hanjun/solicucu/ReID/MobileNetReID"  # for server
import os 
import sys 
import argparse
import logging  
import time 
import torch 
import torch.nn as nn 
import numpy as np 
sys.path.append(root)

from utils import setup_logger, R1_mAP 
from utils.utils import count_parameters
from torch.backends import cudnn 
from data import darts_make_data_loader 
from simple_model_search_res import SSNetwork
from full_model_search import FSNetwork  
from optimizer import darts_make_optimizer, darts_make_lr_scheduler
from optims import darts_make_loss
from architect import Architect 
from utils.metrics import * 
from main.train import compute_loss_acc

def parse_config():

	# create the parser
	parser = argparse.ArgumentParser("darts")
	# device 
	parser.add_argument("--device", type = str, default = "cpu", help = "use cpu or gpu to train")
	parser.add_argument("--device_ids", type = str, default = None, help = "specify the gpu device id if use gpu")

	# data 
	parser.add_argument("--dataset", type = str, default = 'market1501', help = "which dataset to be use for training")
	parser.add_argument("--dataset_dir", type = str, default = "D:/project/data/", help = "path to dataset")
	parser.add_argument("--in_planes", type = int, default = 30, help = "the input channel for first layer")
	parser.add_argument("--init_size", type = list, default = [256, 128], help = "the init spatial size of the image")
	
	# network
	parser.add_argument('--model_name', type = str, default = 'ssnet', help = "specify the name of the model")
	parser.add_argument("--layers", type = list, default = [3,4,6,3], help = 'specify the num of cell in each block')
	parser.add_argument("--use_attention", action = 'store_true', default = True, help = 'whether use attention module ')
	parser.add_argument("--pretrained", type = str, default = None, help = "load the lastest state_dict to start training")
	# parser.add_argument("--pretrained", type = str, default = '/home/share/solicucu/data/ReID/MobileNetReID/ssnet/checkpoints/search_atten_pcbneck_softmax_triplet/'
						# , help = "load the lastest state_dict to start training")
	# dataloader
	parser.add_argument("--num_workers", type = int, default = 4,  help = "number of threads to load data")
	parser.add_argument("--sampler", type = str, default = 'triplet', help = 'how to sample the training data')
	parser.add_argument("--num_instance", type = int, default =4, help = "number samples of each identity in one batch")

	# solver
	## epochs
	parser.add_argument("--max_epochs", type = int, default = 120, help = "the max epoch to train the network")
	parser.add_argument("--batch_size", type = int, default = 64, help = "number of image in each batch")

	## learning rate :for train the network
	parser.add_argument("--base_lr", type = float, default = 0.025, help = "the initial learning rate")
	parser.add_argument("--lr_decay_period", type = int, default = 10, help = "the period for learning rate decay")
	parser.add_argument("--lr_decay_factor", type = float, default = 0.1, help = "learning rate decay factor")
	parser.add_argument("--lr_scheduler_name", type = str, default = "CosineAnnealingLR", help = "the name of lr scheduler")
	parser.add_argument("--lr_min", type = float, default = 0.00001, help = "minimum learning rate for CosineAnnealingLR")
	## learning rate :for achitecture alpha
	parser.add_argument("--arch_lr", type = float, default = 3e-4, help = "learning rate for train the architecture param alpha" )
	
	# optimizer
	parser.add_argument("--optimizer_name", type = str, default = "SGD", help = "the name of the optimizer")
	parser.add_argument("--momentum", type = float, default = 0.9, help = "momentum for optimizer")
	parser.add_argument("--weight_decay", type = float, default = 3e-4, help = "weight decay for optimizer" )
	parser.add_argument("--arch_weight_decay", type = float, default = 1e-3, help = "weight decay factor for architecture")	

	# loss
	parser.add_argument("--loss_name", type = str, default = 'triplet', help = "choices: softmax, triplet, softmax_triplet" )
	parser.add_argument("--tri_margin", type = float, default = 0.3, help = "margin for triplet loss if use triplet loss")
	# output 
	parser.add_argument("--output_dir", type = str, default = "D:/project/data/ReID/MobileNetReID/darts/darts1", help = "path to output")
	parser.add_argument("--ckpt_dir", type = str, default = "checkpoints/atten/", help = "path to save the checkpoint")
	parser.add_argument("--log_period", type = int, default = 10, help = "the period for log")
	parser.add_argument("--log_name", type = str, default = "log.txt", help = "specify a name for log")
	parser.add_argument("--ckpt_period", type = int, default = 10, help = "the period for saving the checkpoint")
	parser.add_argument("--eval_period", type = int, default = 10, help = "the period for validation")

	# others
	parser.add_argument("--unrolled", action = "store_true", default = True, help = "use one-step unrolled validation loss")
	parser.add_argument("--grad_clip", type = float, default = 5, help = "gradient clipping")
	parser.add_argument("--seed", type = int, default = 5, help = "random seed")
	parser.add_argument("--use_pcb", action = 'store_true', default = False, help = 'use pcb_neck')
	parser.add_argument("--use_neck", action = 'store_true', default = True, help = 'use my neck')

	# test
	parser.add_argument("--best_ckpt", type = str, default = "", help = "specify the path to the best checkpoint for test")

	args = parser.parse_args()
	args_dict = vars(args)
	# config the logger
	logger = setup_logger("DARTS",args.output_dir, 0, args.log_name)
	
	logger.info("configs:")
	for key, value in args_dict.items():
		logger.info("{}: {}".format(key, value))
	
	use_gpu = args.device == 'cuda'
	if use_gpu:
		logger.info("Train with GPU: {}".format(args.device_ids))
	else:
		logger.info("Train with CPU")

	if use_gpu:
		os.environ["CUDA_VISIBLE_DEVICES"] = args.device_ids
	cudnn.benchmark = True
	
	#init rand seed 
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if use_gpu:
		torch.cuda.manual_seed(args.seed)
		
	return args 

def train(cfg):
	
	use_gpu = cfg.device == 'cuda'
	# 1、make dataloader
	train_loader, val_loader, test_loader, num_query, num_class =  darts_make_data_loader(cfg)
	# print(num_query)

	# 2、make model
	if cfg.model_name == 'ssnet':
		model = SSNetwork(num_class, cfg, use_gpu)
	
	elif cfg.model_name == 'fsnet':
		model = FSNetwork(num_class, cfg.in_planes, cfg.init_size, cfg.layers, use_gpu, cfg.pretrained) 


	# 3、make optimizer
	optimizer = darts_make_optimizer(cfg, model)
	# print(optimizer)

	# 4、make lr scheduler
	lr_scheduler = darts_make_lr_scheduler(cfg, optimizer)
	# print(lr_scheduler)

	# 5、make loss 
	loss_func = darts_make_loss(cfg)
	model._set_loss(loss_func, compute_loss_acc)
	
	# 6、make architect
	architect = Architect(model, cfg)
	
	# get parameters
	log_period = cfg.log_period
	ckpt_period = cfg.ckpt_period
	eval_period = cfg.eval_period
	output_dir =  cfg.output_dir
	device = cfg.device 
	epochs = cfg.max_epochs
	ckpt_save_path = output_dir + cfg.ckpt_dir 

	use_gpu = device == 'cuda'
	batch_size = cfg.batch_size
	batch_num = len(train_loader)
	log_iters = batch_num // log_period 
	pretrained = cfg.pretrained is not None
	parallel = False
	use_neck = cfg.use_neck 

	if not os.path.exists(ckpt_save_path):
		os.makedirs(ckpt_save_path)

	logger = logging.getLogger("DARTS.train")
	size = count_parameters(model)
	logger.info("the param number of the model is {:.2f} M".format(size))

	logger.info("Start training")
	
	
	if pretrained:
		start_epoch = model.start_epoch 
	if parallel:
		model = nn.DataParallel(model)
	if use_gpu:
		model = model.to(device)

	best_mAP, best_r1 = 0., 0.
	is_best = False
	avg_loss, avg_acc = RunningAverageMeter(), RunningAverageMeter()
	avg_time = AverageMeter()
	# num = 3 -> epoch = 2
	for epoch in range(epochs):
		lr_scheduler.step()
		lr = lr_scheduler.get_lr()[0]
		# architect lr.step
		architect.lr_scheduler.step()
		
		if pretrained and epoch < model.start_epoch :
			continue

		model.train()
		avg_loss.reset()
		avg_acc.reset()
		avg_time.reset()

		for i, batch in enumerate(train_loader):
			
			t0 = time.time()
			imgs, labels = batch
			val_imgs, val_labels = next(iter(val_loader))
			
			if use_gpu:
				imgs = imgs.to(device)
				labels = labels.to(device)
				val_imgs = val_imgs.to(device)
				val_labels = val_labels.to(device)

			# 1、update alpha
			architect.step(imgs, labels, val_imgs, val_labels, lr, optimizer, unrolled = cfg.unrolled)

			optimizer.zero_grad()
			res = model(imgs)
			# loss = loss_func(score, feats, labels)
			loss, acc = compute_loss_acc(use_neck, res, labels, loss_func)
			# print("loss:",loss.item())

			loss.backward()
			nn.utils.clip_grad_norm(model.parameters(), cfg.grad_clip)
			
			# 2、update weights
			optimizer.step()

			# acc = (score.max(1)[1] == labels).float().mean()
			# print("acc:", acc)
			t1 = time.time()
			avg_time.update((t1 - t0) / batch_size)
			avg_loss.update(loss)
			avg_acc.update(acc)
			

			# log info
			if (i+1) % log_iters == 0:
				logger.info("epoch {}: {}/{} with loss is {:.5f} and acc is {:.3f}".format(
					epoch+1, i+1, batch_num, avg_loss.avg, avg_acc.avg))

		logger.info("end epochs {}/{} with lr: {:.5f} and avg_time is: {:.3f} ms".format(epoch+1, epochs, lr, avg_time.avg * 1000))

		
		# test the model
		if (epoch + 1) % eval_period == 0:
			
			model.eval()
			metrics = R1_mAP(num_query, use_gpu = use_gpu)

			with torch.no_grad():

				for vi, batch in enumerate(test_loader):

					imgs, labels, camids = batch

					if use_gpu:
						imgs = imgs.to(device)

					feats = model(imgs)
					metrics.update((feats, labels, camids))

				# compute cmc and mAP
				cmc, mAP = metrics.compute()
				logger.info("validation results at epoch {}".format(epoch + 1))
				logger.info("mAP:{:2%}".format(mAP))
				for r in [1,5,10]:
					logger.info("CMC curve, Rank-{:<3}:{:.2%}".format(r, cmc[r-1]))

				# determine whether current model is the best
				if mAP > best_mAP:
					is_best = True
					best_mAP = mAP
					logger.info("Get a new best mAP")
				if cmc[0] > best_r1:
					is_best = True
					best_r1 = cmc[0]
					logger.info("Get a new best r1")

		# whether to save the model
		if (epoch + 1) % ckpt_period == 0 or is_best:

			if parallel:
				torch.save(model.module.state_dict(), ckpt_save_path + "checkpoint_{}.pth".format(epoch + 1))
				model.module._parse_genotype(file = ckpt_save_path + "genotype_{}.json".format(epoch + 1))
			else:
				torch.save(model.state_dict(), ckpt_save_path + "checkpoint_{}.pth".format(epoch + 1))
				model._parse_genotype(file = ckpt_save_path + "genotype_{}.json".format(epoch + 1))
			
			logger.info("checkpoint {} was saved".format(epoch + 1))

			if is_best:
				if parallel:
					torch.save(model.module.state_dict(), ckpt_save_path + "best_ckpt.pth")
					model.module._parse_genotype(file = ckpt_save_path + "best_genotype.json")
				else:
					torch.save(model.state_dict(), ckpt_save_path + "best_ckpt.pth")
					model._parse_genotype(file = ckpt_save_path + "best_genotype.json")

				logger.info("best_checkpoint was saved")
				is_best = False
		

	logger.info("training is end")

def main():
	print("solicucu")
	cfg = parse_config()
	train(cfg)


if __name__ == "__main__":

	main()