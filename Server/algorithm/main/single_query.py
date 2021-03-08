# -*- coding: utf-8 -*-
# @Author: solicucu
# @E-mail: 2748847613@qq.com
# @Date:   2020-05-09 14:46:26
# @Last Modified time: 2020-05-10 17:12:22
import os 
import sys 
root = "D:/project/Paper/papercode/myReID/backend/Server/algorithm" # for pc
sys.path.append(root)
import glob 
from config import cfg 
# add ReID to adapt backend
from algorithm.data import make_batch_data 
from model import build_model 
from utils.top5 import Top5 

data_root = cfg.DATASET.ROOT_DIR 


numids = {"market1501":751,"dukemtmc":702}
configs = {
	"mobilenetv2x2":"mobilenetv2.yml",
	"shufflenetv2x2":"shufflenetv2.yml",
	"ssnetv4":"ssnet.yml"
}

def query(image_name, username = "hanjun", dataset = "market1501", model_name = "ssnetv4"):
	
	path = data_root + "{}/{}/query/".format(username, dataset)

	#merge the config file 
	config_file = root + "/configs/{}".format(configs[model_name])
	cfg.merge_from_file(config_file)

	# 更新chekpoint
	cfg.MODEL.PRETRAIN_PATH = cfg.MODEL.PRETRAIN_PATH + "{}/{}.pth".format(dataset, model_name)

	
	img_path = path + image_name
	print("query:", img_path)
	# 1、数据准备
	query = make_batch_data(cfg,[img_path]) # return a list of data

	# print(len(res))
	# 2、准备模型
	model = build_model(cfg, numids[dataset])
	# print(model)
	imgs, pids, camids, paths  = query
	# img = img.unsqueeze(0)
	img = imgs[0]

	# print(img.size())
	# 设置为eval 模式
	model.eval()
	result = model(img)

	query[0] = result
	# create the ranker
	ranker = Top5(username, data_root)
	# set the data 
	ranker.set_gallery(dataset, model_name)
	ranker.set_query(query)

	result = ranker.compute()
	return result 

	# print(result.size()) # 1x 2560




if __name__ == "__main__":
	query()

