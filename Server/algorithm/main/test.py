# -*- coding: utf-8 -*-
# @Author: solicucu
# @E-mail: 2748847613@qq.com
# @Date:   2020-01-17 16:59:59
# @Last Modified time: 2020-01-17 18:00:33

root = "D:/project/Paper/papercode/myReID/MobileNetReID"

import os 
import sys
import argparse
import logging
import torch
sys.path.append(root)

from config import cfg 
from utils import setup_logger, R1_mAP
from data import make_data_loader
from model import build_model
from torch.backends import cudnn 



def parse_config():

	parser = argparse.ArgumentParser(description = 'MobileNetReID baseline')

	parser.add_argument("--config_file", default = "", help = "path to specified config file", type = str)

	parser.add_argument("opts", default = None, help = "modify some value in config file", nargs = argparse.REMAINDER)

	args = parser.parse_args()

	if args.config_file != "":
		cfg.merge_from_file(args.config_file)
	cfg.merge_from_list(args.opts)

	cfg.freeze()

	output_dir = cfg.OUTPUT.ROOT_DIR 
	if output_dir != "":
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)
	else:
		print("ERROR: please specify an output path")
		exit(1)

	logger = setup_logger("MobileNetReID", output_dir,0,cfg.OUTPUT.LOG_NAME)

	use_gpu = cfg.MODEL.DEVICE == 'cuda'

	if use_gpu:
		logger.info("Test with GPU: {}".format(cfg.MODEL.DEVICE_ID))
	else:
		logger.info("Test with CPU")

	logger.info(args)
	if args.config_file != "":
		logger.info("load configuratioin file {}".format(args.config_file))

	logger.info("test with config:\n{}".format(cfg))

	if use_gpu:
		os.environ["CUDA_VISIBLE_DEVICE"] =cfg.MODEL.DEVICE_ID
	cudnn.benchmark = True

def test():

	logger = logging.getLogger('MobileNetReID.test')

	# prepare dataloader
	train_loader, val_loader, num_query, num_class = make_data_loader(cfg)
	# prepare model
	model = build_model(cfg, num_class)

	# load param
	ckpt_path = cfg.OUTPUT.ROOT_DIR + cfg.OUTPUT.CKPT_DIR + cfg.TEST.BEST_CKPT 
	if os.path.isfile(ckpt_path):
		model.load_param(ckpt_path)
	else:
		logger.info("file: {} is not found".format(ckpt_path))
		exit(1)

	use_gpu = cfg.MODEL.DEVICE == 'cuda'
	device = cfg.MODEL.DEVICE_ID

	if use_gpu:
		model = nn.DataPararallel(model)
		model.to(device)

	model.eval()
	metrics = R1_mAP(num_query, use_gpu = use_gpu)

	with torch.no_grad():
		for batch in val_loader:
			data, pids, camids = batch 

			if use_gpu:
				imgs.to(device)
			feats = model(imgs)
			metrics.update(feats, labels, camids)

		cmc, mAP = metrics.compute()
		logger.info("test result as follows")
		logger.info("mAP:{:2%}".format(mAP))
		for r in [1,5,10]:
			logger.info("CMC cure, Rank-{:<3}:{:2%}".format(r, cmc[r-1]))

		print("test is endding")




if __name__ == "__main__":

	parse_config()
	test()
