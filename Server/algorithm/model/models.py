# -*- coding: utf-8 -*-
# @Author: solicucu
# @E-mail: 2748847613@qq.com
# @Date:   2020-01-10 22:30:21
# @Last Modified time: 2020-05-09 20:15:18
import glob 
import torch 
import logging 
import torch.nn as nn 
from torch.nn import init
from .backbone import MobileNet, MobileNetV2
from .backbone import get_shufflenet, ShuffleNetV2
from .backbone import get_seresnet 
from .backbone import ResNet 
from .backbone import SSNetwork 
from .backbone import FSNetwork 
# from .neck import Neck 
from .neck_old import Neck 
from .pcb_neck import PCBNeck 
from .backbone import HACNN 
from .backbone import RobustNet 

logger = logging.getLogger("MobileNetReID.train")
class BaseNet(nn.Module):

	def __init__(self, num_class, cfg = None):

		super(BaseNet, self).__init__()

		model_name = cfg.MODEL.NAME
		is_before_gap = cfg.MODEL.BEFORE_GAP 
		use_gpu = cfg.MODEL.DEVICE == "cuda"

		self.final_planes = 1024
		self.num_class = num_class
		self.pretrained = cfg.MODEL.IMAGENET 
		self.self_ckpt = cfg.MODEL.PRETRAIN_PATH
		self.imagenet_ckpt = cfg.MODEL.IMAGENET_CKPT 
		self.use_neck = cfg.MODEL.NECK 
		self.use_pcb = cfg.MODEL.PCB_NECK 
		self.use_bnneck = cfg.MODEL.USE_BNNECK
		self.learn_region = cfg.MODEL.LEARN_REGION 
		self.last_stride = cfg.TRICKS.LAST_STRIDE 
		self.train_on_imagenet = cfg.DATASET.NAME == 'imagenet'
		self.dropout = cfg.TRICKS.DROPOUT 

		if model_name == 'mobilenet':

			self.base = MobileNet(before_gap = is_before_gap)

		elif model_name == 'mobilenetv2':

			self.base = MobileNetV2( width_mult = cfg.MODEL.WIDTH_MULT, before_gap = is_before_gap, pretrained = self.pretrained, last_stride = self.last_stride)
			self.final_planes = self.base.last_channel
			
		elif model_name == 'shufflenetv2':

			self.base = ShuffleNetV2(width_mult = cfg.MODEL.WIDTH_MULT, before_gap = is_before_gap, pretrained = self.pretrained, last_stride = self.last_stride)
			self.final_planes = self.base.final_planes 

		elif 'shufflenet_' in model_name:
			
			gname = model_name.split('_')[1]
			self.base = get_shufflenet(gname, before_gap = is_before_gap)

		elif 'seresnet' in model_name:

			self.base = get_seresnet(model_name, before_gap = is_before_gap)

		elif 'resnet50' == model_name:

			self.base = ResNet(before_gap = is_before_gap, pretrained = self.pretrained)
			self.final_planes = 2048 

		elif 'ssnet' == model_name:

			self.base = SSNetwork(cfg, before_gap = is_before_gap)
			self.final_planes = self.base.input_channels
			self.pretrained = cfg.DARTS.PRETRAINED

		elif 'fsnet' == model_name:

			self.base = FSNetwork(cfg, before_gap = is_before_gap)
			self.final_planes = self.base.input_channels
			self.pretrained = cfg.DARTS.PRETRAINED

		elif 'hacnn' == model_name:

			self.base = HACNN(self.num_class, use_gpu = use_gpu)
			self.final_planes = self.base.feat_dim 

		elif 'robustnet' == model_name:

			self.base = RobustNet(self.num_class, cfg)
			self.final_planes = self.base.feat_dims

		else:
			raise RuntimeError("{} not implement".format(model_name))



		if self.use_pcb:
			logger.info("use PCBNeck for training")
			self.neck = PCBNeck(num_class, self.final_planes, mid_planes = cfg.MODEL.MID_PLANES)

		elif self.use_neck:
			logger.info("use my Neck for training")
			if self.dropout > 0:
				logger.info("use dropout with prob {}".format(self.dropout))
			# neck has entire classifier 
			self.neck = Neck(num_class, self.final_planes, mid_planes = cfg.MODEL.MID_PLANES, use_bnneck = self.use_bnneck, dropout = self.dropout)

		elif self.learn_region:
			# no need to add anything 
			pass 

		elif self.train_on_imagenet:
			logger.info("use global feature for training on imagenet")
			if self.use_bnneck:
				
				self.bnneck = nn.BatchNorm2d(self.final_planes)
	
			self.classifier = nn.Linear(self.final_planes, self.num_class)

		else:
			logger.info("use global feature for training")
			# whether separate the feature used by triplet loss and softmax
			# I choose the former to classify and latter for computing tripelt loss
			self.classifier = nn.Linear(self.final_planes, self.num_class)

			if self.use_bnneck:

				self.bnneck = nn.BatchNorm2d(self.final_planes)
		
			

		if self.pretrained:
			logger.info("use pretrained model from imagenet")

		elif self.self_ckpt != '':
			# load the latest checkpoint 
			self._load_state_dict()

		else:
			self.kaiming_init_()
			# pass 
		# if imagenet_ckpt is not null, load the pretrained model
		if self.imagenet_ckpt != '':
			self._load_imagenet_state_dict()
			
		logger.info("final_planes is {}".format(self.final_planes))

	def forward(self, x):

		feat = self.base(x)
		# size:[8,4]
		if self.use_neck or self.use_pcb:

			res = self.neck(feat)
			return res 

		elif self.learn_region:
			# actually the feat is result with global and local scores and feat 
			return feat 

		elif self.train_on_imagenet:

			if self.use_bnneck:
				feat = self.bnneck(feat)
			feat = feat.view(feat.shape[0], -1)

			cls_score = self.classifier(feat)

			return cls_score
			
		else:
			if self.use_bnneck:
				last_feat = self.bnneck(feat)
			else:
				last_feat = feat
			# flatten the feature
			
			feat = feat.view(feat.shape[0],-1)
			last_feat = last_feat.view(last_feat.shape[0], -1)

			if self.training:
				cls_score = self.classifier(feat)
				return cls_score, last_feat 
			else:
				return last_feat 

	def kaiming_init_(self):

		logger.info("use kaiming init")
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				init.kaiming_normal_(m.weight)
				if m.bias is not None:
					init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				if m.weight is not None:
					init.constant_(m.weight, 1)
				if m.bias is not None:
					init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				init.normal_(m.weight, std = 1e-3)
				if m.bias is not None:
					init.constant_(m.bias, 0)

	def _load_state_dict(self):

		# x修改一下模型加载的方式
		if self.self_ckpt.endswith(".pth"):
			print("load a specified checkpoint {}".format(self.self_ckpt))
			state_dict = torch.load(self.self_ckpt, map_location=lambda storage, loc: storage)
			self.load_state_dict(state_dict)
			return 

		logger.info("load the latest checkpoint")
		# checkpoint
		ckpt_list = glob.glob(self.self_ckpt + "checkpoint_*")
		ckpt_list = sorted(ckpt_list)
		# print(ckpt_list)
		# exit(1)
		ckpt_name = ckpt_list[-1]
		# print(ckpt_name)
		num = int(ckpt_name.split("/")[-1].split("_")[1].split(".")[0])
		self.start_epoch = num

		#self.load_state_dict(torch.load(ckpt_name)) 

		# or
		self_state_dict = self.state_dict()
		state_dict = torch.load(ckpt_name)
		for key in self_state_dict:
			self_state_dict[key].data.copy_(state_dict[key].data)
		logger.info("load checkpoint from {}".format(ckpt_name))

	def _load_imagenet_state_dict(self):
		logger.info("load the self-trained imagenet ckpt")
		
		state_dict = torch.load(self.imagenet_ckpt)

		rm = ['bnneck', 'classifier']
		self_state_dict = self.state_dict()
		for key in state_dict:
			remove = False
			# rm the parameter with bnneck and classifier
			for r in rm:
				if r in key:
					remove = True
					break

			if not remove:
				self_state_dict[key].data.copy_(state_dict[key].data)
		logger.info("load checkpoint from {}".format(self.imagenet_ckpt)) 
		""" 在cpu加载gpu的模型
		map_location = lambda storage, loc: storage
		model = torch.load(args.weight,map_location=map_location)
		"""

		



