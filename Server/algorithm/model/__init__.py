# -*- coding: utf-8 -*-
# @Author: solicucu
# @E-mail: 2748847613@qq.com
# @Date:   2020-01-10 22:48:55
# @Last Modified time: 2020-02-18 12:53:47

from .models import BaseNet 

def build_model(cfg, num_class):

	model = BaseNet(num_class, cfg)

	return model
