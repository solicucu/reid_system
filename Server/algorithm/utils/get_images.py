# -*- coding: utf-8 -*-
# @Author: solicucu
# @E-mail: 2748847613@qq.com
# @Date:   2020-01-09 16:25:01
# @Last Modified time: 2020-01-09 16:44:08

import os 
import cv2
from PIL import Image 

"""
read the images form ../images
if names is None ,it will return all images, otherwise return a list contains an image
"""
def get_image(names = None):
	imgs = []
	root = "../images/"
	if names == None:
		names = os.listdir(root)
		for name in names:
			img_path = root + name 
			# img = cv2.imread(img_path)
			img = Image.open(img_path).convert("RGB")
			imgs.append(img)
	else:
		img_path = root + names
		# img = cv2.imread(img_path)
		img = Image.open(img_path).convert("RGB")
		imgs.append(img)

	return imgs  
