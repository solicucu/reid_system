# -*- coding: utf-8 -*-
# @Author: solicucu
# @E-mail: 2748847613@qq.com
# @Date:   2020-05-06 21:41:35
# @Last Modified time: 2020-05-13 17:41:22
import matplotlib.pyplot as plt 
import numpy as np 

def test():

	x = [1,2,3,4,5,6]
	y = [0.1,0.2,0.4, 0.6, 0.8, 1]
	plt.subplot(121)
	plt.plot(x, y)
	plt.subplot(122)
	plt.plot(y,x)
	plt.show()
	#plt.savefig("./sig_size.png")#保存图片

def draw(xs, ys, colors, markers, linestyles, labels, xticks, title, xlabel = "metrics", ylabel = "acc"):

	# 设置标题
	plt.title(title)
	# 设置高度限制
	plt.ylim(50,100)
	# 设置x轴标号
	# 转化xticks 为 字定义需要同时给出
	plt.xticks(xs[0],xticks)
	# 设置y的刻度
	plt.yticks(np.arange(50,101,5))
	# 设置坐标轴名称
	plt.ylabel(ylabel)
	plt.xlabel(xlabel)


	lens = len(xs)
	for i in range(lens):
		plt.plot(xs[i], ys[i], color = colors[i], marker = markers[i], linestyle = linestyles[i], label = labels[i])
	
	# 设置legend
	# 要写在后面，upper or lower 表示上下
	plt.legend(loc = 'lower right')
	# plt.show()

def effect_of_ppl():
	xticks = ["mAP", 'r1', 'r5', 'r10']
	x =  [1,2,3,4]

	# mobilenetv2x2 
	# mobilenetv2x2_neck 
	y = [
		 [55.13, 73.6, 88.03, 91.72],
		 [67.2, 83.43, 92.64, 95.04]
		]
	x = [x for i in range(2)]

	markers = ["*", "*"]
	colors = ["r", 'r']
	linestyles = ["-", ":"]
	labels = ["mobilenetv2x2","mobilenetv2x2_neck"]
	title = "Effect of PPL on mobilenetv2x2"
	# 多个画在一张图
	plt.subplot(131)
	draw(x,y,colors,markers,linestyles,labels,xticks,title)

	# 画 shufflenetv2x2
	y = [
		[50.81, 72.71, 88.33, 92.9], # shufflenetv2x2
		[59.79, 78.38, 89.85, 92.93]  # shufflenetv2x2_neck
	]
	labels = ["shufflenetv2x2", "shufflenetv2x2_neck"]
	title = "Effect of PPL on shufflenetv2x2"
	plt.subplot(132)
	draw(x,y,colors,markers,linestyles,labels,xticks,title)
	
	# 画ssnets
	y = [
		[43.98, 64.64, 83.11, 88.42], # ssnet
		[58.49, 79.04, 91.36, 94.39], # ssnet_neck
		[50.83, 72, 87.8, 92.07],     # ssnetv2
		[56.43, 77.67, 90.68, 93.44], # ssnetv2_neck
		[51.72, 72.39, 89.19, 92.93], # ssnetv3
		[57.91, 79.93, 91.3, 94.69],  # ssnetv3_neck
		[49.4, 70.43, 87.7, 91.51],   # ssnetv4
		[60.5, 81.09, 92.01, 94.66]	  # ssnetv4_neck
	]
	x = [[1,2,3,4] for i in range(8)]
	
	labels = ["ssnet","ssnet_neck","ssnetv2","ssnetv2_neck", "ssnetv3", "ssnetv3_neck", "ssnetv4", "ssnetv4_neck"]
	title = "Effect of PPL on ssnets"
	markers = markers * 4
	linestyles = linestyles * 4
	colors = ["b",'b','g','g','y','y','r','r']
	plt.subplot(133)
	draw(x,y,colors,markers,linestyles,labels,xticks,title)
	plt.show()
	
def effect_of_atten():

	# 画ssnets
	y = [
		[58.49, 79.04, 91.36, 94.39], # ssnet_neck
		[62.34, 82.48, 92.73, 95.37], # ssnet_neck_atten
		[57.91, 79.93, 91.3, 94.69],  # ssnetv3_neck
		[60.66, 80.23, 91.21, 93.79], # ssnetv3_neck_atten
		[60.5, 81.09, 92.01, 94.66],  # ssnetv4_neck
		[62.3, 81.95, 92.99, 95.67]   # ssnetv4_neck_atten
	]
	x = [[1,2,3,4] for i in range(6)]
	labels = ["ssnet_neck","ssnet_neck_atten", "ssnetv3_neck", "ssnetv3_neck_atten", "ssnetv4_neck", "ssnetv4_neck_atten"]
	title = "Effect of attention on ssnets"
	xticks = ["mAP", 'r1', 'r5', 'r10']
	markers = ["*", "*"]
	linestyles = ["-", ":"]
	markers = markers * 3
	linestyles = linestyles * 3
	colors = ["b",'b','g','g','r','r']
	# plt.subplot(133)
	draw(x,y,colors,markers,linestyles,labels,xticks,title)
	plt.show()

def draw_wamrup():
	base_lr = 0.025
	start_lr = base_lr * 0.1
	x = [i+1 for i in range(10)]
	y = [start_lr * i for i in x ]

	# cosine
	min_lr = 0.001
	max_sub_min = base_lr - min_lr 
	x2 = np.linspace(10,160,100)
	x.extend(x2)
	T = 150 
	y2 = [min_lr + max_sub_min * 0.5 * (1 + np.cos(((i-10) / T) * np.pi)) for i in x2]
	y.extend(y2)
	plt.xlim(1,160)
	plt.xlabel("epoch")
	plt.ylabel("lr")
	plt.title("warmup scheduler")
	plt.plot(x,y)
	plt.show()

def effect_of_tricks():
	# for mobilenetv2x2 
	x = [[1,2,3] for i in range(2)]

	y = [
		[66.62, 67.91, 68.86],
		[83.64, 84.38, 84.68]
	]

	labels = ["mAP", "rank-1"]
	xlabel = "tricks"
	ylabel = "acc"
	title = "Effect of tricks on mobilenetv2x2"
	xticks = ["none", "warmup", "label_smooth"]
	markers = ["*", "*"]
	linestyles = ["-", "-"]
	colors = ['r', 'b']
	draw(x,y,colors,markers,linestyles,labels,xticks,title,xlabel,ylabel)
	plt.show()

def compare_with_tricks():

	# 4个模型
	x = [[1,2,3,4] for i in range(4)]
	y = [
		[65.8, 83.7, 93.9, 96.7], # baseline
		[68.86, 84.68, 93.56, 95.67], # mobilenetv2x2 
		[60.46, 79.93, 90.8, 94.15],  # shufflenetv2x2
		[67.12, 85.24, 94.48, 96.44], # ssnetv4
	]
	labels = ["baseline_26M", "mobilenetv2x2_13.59M", "shufflenetv2x2_10.22M", "ssnetv4_9.68M"]
	title = "compare the performance with tricks"
	xticks = ["mAP", "r1", "r5", "r10"]
	markers = ["*"] * 4
	linestyles = ["-"] * 4
	colors = ['b', 'g', 'y', 'r']
	draw(x,y,colors,markers,linestyles,labels,xticks,title)
	plt.show()

def imagenet_pretraiend_market1501():
	# 4个模型
	x = [[1,2,3,4] for i in range(4)]
	y = [
		[68.86, 84.68, 93.56, 95.67], # mobilenetv2x2
		[71.35, 85.21, 93.74, 95.75], # mobilenetv2x2_pretrained 
		[67.12, 85.24, 94.48, 96.44],  # ssnetv4
		[75.87, 89.93, 95.78, 97.42], # ssnetv4_pretrained
	]

	labels = ["mobilenetv2x2","mobilenetv2x2(P)", "ssnetv4", "ssnetv4(P)"]
	title = "effect of pretrained on imagenet"
	xticks = ["mAP", "r1", "r5", "r10"]
	markers = ["*"] * 4 
	linestyles = ["-", ":"] * 2
	colors = ['b', 'b', 'r', 'r']
	plt.subplot(121)
	draw(x,y,colors,markers,linestyles,labels,xticks,title)

	y = [
		[70.01, 85.99, 93.62, 95.67], # mobilenetv2x2_ndw
		[66.06, 80.46, 90.88, 94.3], # mobilenetv2x2_ndw_pretrained 
		[66.98, 85.69, 94.03, 96.05],  # ssnetv4_ndw
		[74.76, 89.04, 95.72, 97.33], # ssnetv4_ndw_pretrained
	]
	labels = ["mobilenetv2x2_ndw","mobilenetv2x2_ndw(P)", "ssnetv4_ndw", "ssnetv4_ndw(P)"]
	title = "effect of pretrained on imagenet with neck_dw"
	xticks = ["mAP", "r1", "r5", "r10"]
	markers = ["*"] * 4 
	linestyles = ["-", ":"] * 2
	colors = ['b', 'b', 'r', 'r']
	plt.subplot(122)
	draw(x,y,colors,markers,linestyles,labels,xticks,title)
	plt.show()

def scatter(x, y, area, color, xlabel, ylabel, title, labels):
	# 设置标题
	plt.title(title)
	# 设置高度限制
	plt.ylim(0,100)
	# 设置x轴标号
	plt.xticks(np.arange(0, 6, 1))
	# 设置y的刻度
	plt.yticks(np.arange(0,101,10))
	# 设置坐标轴名称
	plt.ylabel(ylabel)
	plt.xlabel(xlabel)

	area = [v * 50 for v in area]

	for i in range(len(x)):
		x1 = [x[i]]
		y1 = [y[i]]
		c1 = [color[i]]
		s1 = [area[i]]
		l1 = labels[i]
		plt.scatter(x1, y1, c = c1, s = s1 )
		plt.scatter(x1, y1, c ='white', marker = "*")
		plt.text(x1[0], y1[0]-7, l1)
		

def show_param_acc_speed():
	# param_mAP_speed
	speed = [4, 1.39, 1.1, 2.1]
	mAP = [85.6, 71.35, 60.46, 74.76]
	rank1 = [94.1, 85.21, 79.93, 89.04]
	param = [26, 13.59, 10.22, 6.36]
	colors = ['b','y','g','r'] 
	xlabel = "speed/ms"
	ylabel = "mAP"
	title = "compare performance w.r.t speed,mAP,param"
	labels = ["baseline", "mobilenetv2x2","shufflenetv2x2", "ssnetv4"]

	plt.subplot(121)
	scatter(speed, mAP, param, colors, xlabel, ylabel, title, labels)
	plt.subplot(122)
	ylabel = "rank-1"
	title = "compare performance w.r.t speed,rank-1,param"
	scatter(speed, rank1, param, colors, xlabel, ylabel, title, labels)
	plt.show()

if __name__ == "__main__":
	# test()
	# effect_of_ppl()
	# effect_of_atten()
	# draw_wamrup()
	# effect_of_tricks()
	# compare_with_tricks()
	# imagenet_pretraiend_market1501()
	show_param_acc_speed()