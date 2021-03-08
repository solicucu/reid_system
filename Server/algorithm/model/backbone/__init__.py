# -*- coding: utf-8 -*-
# @Author: solicucu
# @E-mail: 2748847613@qq.com
# @Date:   2020-01-10 22:29:16
# @Last Modified time: 2020-04-17 10:03:31

from .mobilenet import MobileNet 
from .shufflenetv1 import get_shufflenet 
from .seresnet import get_seresnet
from .resnet import ResNet 
from .shufflenetv2 import shufflenet_v2 as ShuffleNetV2
from .mobilenetv2 import MobileNetV2 
# from .dual_mobilenetv2 import MobileNetV2
# from .ssnet import SSNetwork  
# change the resnet with residual structure
from .ssnet_res import SSNetwork  
# from .ssnet_tiny import SSNetwork
from .fsnet import FSNetwork 
from .hacnn import HACNN 
from .robustreid import RobustNet 