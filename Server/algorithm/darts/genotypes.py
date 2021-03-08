# -*- coding: utf-8 -*-
# @Author: solicucu
# @E-mail: 2748847613@qq.com
# @Date:   2020-02-12 15:20:01
# @Last Modified time: 2020-02-24 22:13:16

op_names = [
	'sep_conv_3x3',
	'sep_conv_5x5',
	'dil_conv_3x3',
	'dil_conv_5x5',
	'avg_pool_3x3',
	'max_pool_3x3',
	'skip_connect',
	# 'zeros'
]

Genotype = {
	'layer1':['sep_conv_3x3', 'dil_conv_5x5','skip_connect'],
	'layer2':['dil_conv_3x3', 'sep_conv_5x5','skip_connect'],
	'layer3':['sep_conv_3x3', 'sep_conv_5x5', 'avg_pool_3x3'],
	'layer4':['dil_conv_3x3', 'sep_conv_3x3', 'max_pool_3x3']
}

fop_names = [
	'sep_conv_3x3',
	'sep_conv_5x5',
	'dil_conv_3x3',
	'dil_conv_5x5',
	'avg_pool_3x3',
	'max_pool_3x3',
	'skip_connect',
	# 'zeros',
	'channel_atten',
	'spacial_atten'
]