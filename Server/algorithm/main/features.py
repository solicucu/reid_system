# -*- coding: utf-8 -*-
# @Author: solicucu
# @E-mail: 2748847613@qq.com
# @Date:   2020-05-09 22:28:17
# @Last Modified time: 2020-05-10 17:34:18

import sys 
root = "D:/project/Paper/papercode/myReID/backend/Server/algorithm" # for pc
sys.path.append(root)
import os
import torch 
from config import cfg 
from algorithm.data import make_batch_data
from model import build_model 

data_root = cfg.DATASET.ROOT_DIR 

models_name = ["mobilenetv2x2","shufflenetv2x2", "ssnetv4"]

numids = {"market1501":751,"dukemtmc":702}
configs = {
	"mobilenetv2x2":"mobilenetv2.yml",
	"shufflenetv2x2":"shufflenetv2.yml",
	"ssnetv4":"ssnet.yml"
}

def produce_features(username = "hanjun", dataset = "market1501"):
	
	path = data_root + "{}/{}/gallery/".format(username, dataset)

	img_paths = [path + name for name in os.listdir(path)] 
	lens = len(img_paths)
	
	batch_size = 2
	# 如果目录不存在，就创建目录
	features_root = data_root + "{}/features/".format(username)
	if not os.path.exists(features_root):
		os.makedirs(features_root)

	#依次对每个模型生成对应的features
	for name in models_name:
		# 准备模型
		#merge the config file 
		config_file = root + "/configs/{}".format(configs[name])
		cfg.merge_from_file(config_file)
		# 更新chekpoint
		if dataset not in numids:
			cfg.MODEL.PRETRAIN_PATH = cfg.MODEL.PRETRAIN_PATH + "{}/{}.pth".format("market1501", name)
			model = build_model(cfg, numids["market1501"])
		else:
			cfg.MODEL.PRETRAIN_PATH = cfg.MODEL.PRETRAIN_PATH + "{}/{}.pth".format(dataset, name)
			model = build_model(cfg, numids[dataset])
		model.eval()
		
		save_path = features_root + "{}_{}.feats".format(name,dataset)
		data = {}
		feats = []
		pids = []
		camids = []
		paths = []
		# 如果已经存在了就跳过
		if os.path.exists(save_path):
			print("feature {} is exist".format(save_path))
			continue
		print("use {} producing the gallery features of {}, please waiting".format(name,dataset))
		# 开始计算特征
		with torch.no_grad():
			for i in range(0,lens, batch_size):

				end = i + batch_size
				if end > lens:
					end = lens
				batch = img_paths[i:end]
				feat, pid, camid, path = make_batch_data(cfg, batch)
				feat = model(feat)
				feats.append(feat)
				pids.extend(pid)
				camids.extend(camid)
				paths.extend(path)
				
			feats = torch.cat(feats, dim = 0)
			data["feats"] = feats 
			data["pids"] = pids 
			data["camids"] = camids 
			data["paths"] = paths
			torch.save(data, save_path)
		


if __name__ == "__main__":
	produce_features()