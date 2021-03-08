# -*- coding: utf-8 -*-
# @Author: solicucu
# @E-mail: 2748847613@qq.com
# @Date:   2020-01-08 10:50:11
# @Last Modified time: 2020-05-09 22:33:09

from yacs.config import CfgNode as CN 

_C = CN()

#------------------------------
# MODEL
# config for model paramter
#------------------------------

_C.MODEL = CN()
# using gpu or cpu for the traing
_C.MODEL.DEVICE = 'cpu'
# specify the gpu to be used if use gpu
_C.MODEL.DEVICE_ID = '0'
# name of the backbone
_C.MODEL.NAME = 'mobilenet'
# load the specified checkpoint for the model, self pretrained
_C.MODEL.PRETRAIN_PATH = ''
# load the imagenet checkpoint which we trained
_C.MODEL.IMAGENET_CKPT = ''
# whether separate the feature use for triplet loss and softmax loss 
_C.MODEL.USE_BNNECK = False
# whether return the featrue map before GAP
_C.MODEL.BEFORE_GAP = False
# whether use DataParallel
_C.MODEL.PARALLEL = False
# whether use the pretraine model on imagenet
_C.MODEL.IMAGENET = False 
# whether use Neck
# note that if was True, then model.before_gap also is True
_C.MODEL.NECK = False
# if set extend_planes to zero, no extend, or extend the final_planes to specify planes
_C.MODEL.EXTEND_PLANES = 0
# for mobilenetv2 x1 x2 x1.5
_C.MODEL.WIDTH_MULT = 1.
# for neck 
_C.MODEL.MID_PLANES = 512 
# use pcb neck
_C.MODEL.PCB_NECK = False
# for HACNN 
_C.MODEL.LEARN_REGION = False

# for RobustNet
# block repeated number
_C.MODEL.LAYERS = [7, 10, 7]
# out_channels
_C.MODEL.OUT_PLANES = [128, 256, 384]
# final feat dims for both global and local
_C.MODEL.FEAT_DIMS = 512

#--------------------------------
# TRICKS
# set some tricks here
#--------------------------------

# tircks 
_C.TRICKS = CN()
# use the label smooth to prevent overfiting
_C.TRICKS.LABEL_SMOOTH = False
# if the stride for the last layer, if set to 1, we will get a bigger feature map 
_C.TRICKS.LAST_STRIDE = 2
# specify the dropout probability
_C.TRICKS.DROPOUT = 0.



#--------------------------------
# DATASET
# information for dataset
#--------------------------------

_C.DATASET = CN()
# the name of the dataset used
_C.DATASET.NAME = 'market1501'
# the path to the dataset 
# _C.DATASET.ROOT_DIR = "/home/share/solicucu/data/"  #for sever
_C.DATASET.ROOT_DIR = "D:/project/data/server/data/"   # for pc 


#--------------------------------
# DATA
# preprocess the data
#--------------------------------

_C.DATA = CN()
# size of the image during the training
_C.DATA.IMAGE_SIZE = [256,128]
# the probobility for random image horizontal flip
_C.DATA.HF_PROB = 0.5
# the probobility for random image erasing
_C.DATA.RE_PROB = 0.5
# rgb means used for image normalization
_C.DATA.MEAN = [0.485, 0.456, 0.406]
# rgb stds used for image normalization
_C.DATA.STD = [0.229, 0.224, 0.225]
# value of padding size
_C.DATA.PADDING = 10


#--------------------------------
# DATALOADER
#--------------------------------

_C.DATALOADER = CN()
# number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# types of Sampler for data loading
_C.DATALOADER.SAMPLER = 'triplet'
# number of instance for single person
_C.DATALOADER.NUM_INSTANCE = 4




#--------------------------------
# SOLVER
#--------------------------------

_C.SOLVER = CN()
# total number of epoch for training
_C.SOLVER.MAX_EPOCHS = 30
# end_epoch for cos
_C.SOLVER.END_EPOCH = 120 
# number of images per batch
_C.SOLVER.IMGS_PER_BATCH = 64

# learning rate
# the initial learning
_C.SOLVER.BASE_LR = 0.01
# the period for lerning decay
_C.SOLVER.LR_DECAY_PERIOD = 10
# learning rate decay fator
_C.SOLVER.LR_DECAY_FACTOR = 0.1
# lr scheduler [StepLR, ConsineAnnealingLR, WarmupMultiStepLR]
_C.SOLVER.LR_SCHEDULER_NAME = "StepLR" 
# min_lr for ConsineAnnealingLR
_C.SOLVER.LR_MIN = 0.001 

# for warmupMultiStepLR 
# at which epoch change the lr
_C.SOLVER.MILESTONES = [40, 70]
# lr list for multistep
_C.SOLVER.LR_LIST = [3.5e-4, 3.5e-5, 3.5e-6]
# coefficient for linear warmup 
_C.SOLVER.GAMA = 0.1
# use to calculate the start lr, init_lr = base_lr * warmup_factor
_C.SOLVER.WARMUP_FACTOR = 0.1
# how many epoch to warmup, 0 denote do not use warmup 
_C.SOLVER.WARMUP_ITERS = 0
# method for warmup 
_C.SOLVER.WARMUP_METHOD = 'linear'

# optimizer
# the name of the optimizer
_C.SOLVER.OPTIMIZER_NAME = "SGD"
# momentum for SGD
_C.SOLVER.MOMENTUM = 0.9
# weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0005

# loss
# loss type:softmax, triplet , softmax_triplet
_C.SOLVER.LOSS_NAME = "softmax_triplet"
# the margin for triplet loss
_C.SOLVER.TRI_MARGIN = 0.3
# triplet loss weight 
_C.SOLVER.TRIPLET_LOSS_WEIGHT = 0.5 

#--------------------------------
# TEST
#--------------------------------
_C.TEST = CN()
# batch size for test
_C.TEST.IMGS_PER_BATCH = 128
# whether feature is normalized before test
_C.TEST.FEAT_NORM = 'yes'
# the name of best checkpoint for test
_C.TEST.BEST_CKPT = ''

#--------------------------------
# OUTPUT
#--------------------------------

_C.OUTPUT = CN()

# the root directory for all output
_C.OUTPUT.ROOT_DIR = 'D:/project/data/ReID/MobileNetReID/ssnet/'
_C.OUTPUT.CKPT_DIR = 'checkpoint/kaiming_cos/'
# period for the log
_C.OUTPUT.LOG_PERIOD = 1
# specify a name for log text
_C.OUTPUT.LOG_NAME = 'log.txt'
# the period to save the checkpoint 
_C.OUTPUT.CHECKPOINT_PERIOD = 1
# the period to eval the model on eval dataset
_C.OUTPUT.EVAL_PERIOD = 1

#------------------------------
# DARTS
#------------------------------
_C.DARTS = CN()
# input channel for the first layers
_C.DARTS.IN_PLANES = 60
# initial size of the image
_C.DARTS.INIT_SIZE = [256,128]
# the layers configuration 
_C.DARTS.LAYERS = [3, 4, 6, 3]
# model genotype path
_C.DARTS.GENOTYPE = 'best_genotype.json'
# pretrained path // if exits best_ckpt_fit.pth, directly load the state_dict
# or change the best_ckpt.pth to best_ckpt_fit.pth
_C.DARTS.CKPT_NAME = "best_ckpt_fit.pth"
# whether to use attention model
_C.DARTS.USE_ATTENTION = False
# whether load the param trained from search process
_C.DARTS.PRETRAINED = False
# for FSNetwork 
_C.DARTS.STEP = 3
# clip the gradient
_C.DARTS.GRAD_CLIP = 0.
# width multiplier  
_C.DARTS.MULTIPLIER = 2

