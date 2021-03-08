from django.core.paginator import Paginator
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render

# Create your views here.
from django.views import View

from reid.models import User
from .data import *
from Server.settings import data_root
import jsonpickle
import numpy as np
import time

from algorithm.main.single_query import query
from algorithm.main.features import produce_features

# 实现一个普通类来封装数据
class Account(object):
	def __init__(self, name, password):
		self.name = name
		self.password = password

	# 隐藏某些字段，不进行序列化,这里是隐藏密码
	def __getstate__(self):
		data = self.__dict__.copy()
		del data['password']
		return data

class LoginView(View):

	def get(self, request, *args, **kwargs):

		return render(request, 'login.html')

	def post(self, request, *args, **kwargs):

		# 1、获取参数
		name = request.POST.get("name", '')
		password = request.POST.get("password", '')
		if name and password:
			num = User.objects.filter(name = name, password = password).count()
			if num > 0:
				# return HttpResponse("success loging, welcome {}".format(name))
				# 设置session login 为 key name 为value, 真实key是一个跟login有关的随机字符串 session_id 存储在 cookie 里面
				# 1、 存储字符串数据
				request.session['login'] = name
				# 2、存储object类型数据
				"""
				account = Account(name, password)
				# 需要转化为json字符串
				request.session['login'] = jsonpickle.dumps(account)
				"""
				return render(request, "home.html", {"name": name})
			else:
				num = User.objects.filter(name = name).count()
				if num > 0:
					return render(request, "login.html", {"error": "密码错误"})
				return HttpResponse("no such account, regester a new one ?")

		return  HttpResponse("name and password can not be null")

class RegisterView(View):

	def get(self, request, *args, **kwargs):

		return render(request, 'register.html')

	def post(self, request, *args, **kwargs):

		# 1、获取参数
		name = request.POST.get("name", '')
		password = request.POST.get("password", '')
		# 2、check the params 已经在前端验证了，所以这里不需要验证合法性，只需要验证唯一性
		if name and password:
			num = User.objects.filter(name = name).count()
			if num > 0:
				return render(request, 'register.html', {"error": "{}已经被占用了".format(name)})
			else:
				user = User(name = name, password = password)
				user.save()
				return render(request, "login.html")

		return render(request, "register.html")

class HomeView(View):

	def get(self, request, *args, **kwargs):

		# return render(request, 'test.html')
		name = request.session['login']
		return render(request, 'home.html', {"name":name})

class Dataupload(View):

	def get(self, request, *args, **kwargs):
		name = request.session['login']
		return  render(request, 'upload.html', {"name":name})

	def post(self, request, *args, **kwargs):
		username = request.session['login']
		file = request.FILES.get("filename", '')
		# gets the filename
		filename = file.name
		if filename.split(".")[1] != "zip":
			return render(request, 'upload.html', {"info": "限上传zip压缩文件", "name": username})
		res = save_dataset(username, file)
		if res:
			dataset = file.name.split(".")[0]
			produce_features(username, dataset)
			return render(request, 'upload.html',{"info": "上传成功","name": username})
		else:
			return render(request, 'upload.html', {"info": "上传失败","name": username})

def get_page(files, num):
	pager = Paginator(files, 1)
	# 防止页码超出
	if num <= 0:
		num = 1
	if num > pager.num_pages:
		num = pager.num_pages
	page = pager.page(num)  # return a list

	return page, num

class DataPreview(View):

	def get(self,request, *args, **kwargs):
		username = request.session['login']
		# 获取数据集名字
		# datasets = ["market10",'dukemtmc10']
		path = data_root + "{}/".format(username)
		if not os.path.exists(path):
			return HttpResponse("you haven't upload anydata,please upload before preview")
		datasets = get_dirs(path)
		if "features" in datasets:
			datasets.remove("features")
		dirs = ['query', 'gallery']
		# 判段数据集是否为空
		if len(datasets) == 0:
			datasets = ["None"]
			dirs = ["None"]
			return render(request, "preview.html", {"name":username, "datasets":datasets, "dirs":dirs})

		relpath = "{}/{}/{}/".format(username, datasets[0], dirs[0])
		path = data_root + relpath
		files = get_files(path)
		image = relpath + files[0]
		return render(request, 'preview.html', {"name": username, "datasets":datasets, 'dirs': dirs, "image":image})

	def post(self, request, *args, **kwargs):
		username = request.session['login']
		dataset = request.POST.get("dataset")
		dirname = request.POST.get("dirname")

		if dataset == "None":
			return JsonResponse({"error":True})
		num = request.POST.get("num")
		num = int(num)

		relpath = "{}/{}/{}/".format(username,dataset,dirname)
		path = data_root + relpath
		files = get_files(path)
		page, num = get_page(files, num)
		# print(page[0])
		# 取第0个是因为我们这里每页只有一个图片
		image = relpath + page[0]
		# 传 num 回去是为了同步global_number
		return JsonResponse({"image":image, "num":num, "error":False})
		# return  HttpResponse("{} and {}".format(dataset, dirname))
		# return render(request, 'preview.html', {"name": username, "datasets":datasets, 'dirs': dirs,"images":images})


class FastReID(View):

	def get(self,request, *args, **kwargs):
		username = request.session['login']

		path = data_root + "{}/".format(username)
		if not os.path.exists(path):
			return HttpResponse("you haven't upload anydata,please upload before use ReID")

		num = [0,1,2,3,4]
		datasets = os.listdir(data_root + "{}/".format(username))
		keep = []
		for item in datasets:
			if item == "features" or item.endswith(".zip"):
				continue
			else:
				keep.append(item)
		return render(request, 'fastreid.html', {"name":username, "num":num, "datasets": keep})

	def post(self,request, *args, **kwargs):
		types = request.POST.get("type")
		username = request.session['login']
		dataset = request.POST.get("dataset")

		if types == "select":
			relpath = "{}/{}/query/".format(username, dataset)
			path = data_root + relpath
			files = get_files(path)
			image = relpath + files[0]
			# 根据模型类型，判断对应数据集特征是否已经生成，如果没有调用算法去提取所有gallery的特征
			# generate_feature(username, model, dataset)
			return JsonResponse({"image":image})

		elif types == "prev_next":
			relpath = "{}/{}/query/".format(username, dataset)
			num = request.POST.get("num")
			num = int(num)
			path = data_root + relpath
			files = get_files(path)
			page, num = get_page(files, num)
			image = relpath + page[0]
			return JsonResponse({"image": image, "num": num})
		else:
			# query
			modeltype = request.POST.get("modeltype")
			ranktype = request.POST.get("ranktype")
			imagepath = request.POST.get("imagepath")
			name = imagepath.split("/")[-1]
			relpath = "{}/{}/query/{}".format(username, dataset,name)
			abspath = data_root + relpath
			begin = time.time()
			result = query(name, username, dataset, modeltype)
			end = time.time()
			cost = end - begin
			cost = format(cost, ".2f")
			print(cost)
			print(result)
			paths = result["paths"]
			matches = result["matches"]
			names = [path.split("/")[-1] for path in paths]
			labels = ["√" if m == 1 else "×" for m in matches]
			# select rank number
			rank = int(ranktype.split("-")[-1])
			labels = list(labels)[:rank]
			images = ["{}/{}/gallery/{}".format(username, dataset, name) for name in names[:rank]]
			return JsonResponse({"images":images, "labels":labels, "cost": cost})


def get_session_data(request):
	# 1、获取字符串数据
	name = request.session['login']
	# 2、获取对象数据
	# account = request.session['login']
	# account = jsonpickle.loads(account)
	# name = account.name
	return HttpResponse(name)

def get_view(request):
	name = request.GET.get("username")
	print(name)
	return JsonResponse({'flag':True})


class UserCenter(View):

	def get(self, request, *args, **kwargs):
		name = request.session['login']
		return render(request, 'usercenter.html', {"name":name})


class ChangePasswd(View):

	def get(self, request, *args, **kwargs):
		name = request.session['login']
		return render(request, 'changepwd.html', {"name":name})

	def post(self,requset, *args, **kwargs):
		name = requset.session['login']
		old = requset.POST.get("old")
		new = requset.POST.get("new")
		res = User.objects.filter(name = name, password=old).count()
		if res == 0:
			return JsonResponse({"hint":"原密码错误"})
		else:

			User.objects.filter(name = name).update(password = new)
			return JsonResponse({"hint": "修改成功"})

import shutil
import glob
class ManageData(View):
	def get(self,request, *args, **kwargs):
		name = request.session['login']
		path = data_root + "{}/".format(name)
		if not os.path.exists(path):
			return HttpResponse("you haven't upload anydata,please upload before manage data")
		path = data_root + "{}/".format(name)
		datasets = get_dirs(path)
		if "features" in datasets:
			datasets.remove("features")
		return render(request, "managedata.html", {"name":name, "datasets": datasets})

	def post(self, request, *args, **kwargs):
		name = request.session['login']
		deldatasets = request.POST.getlist("deldatasets", [])
		path = data_root + "{}/".format(name)
		feats_path = path + "features/"
		if len(deldatasets):
			# 删除所有选择的数据集
			for n in deldatasets:
				p = path + n
				# 递归删除文件夹
				shutil.rmtree(p)
				file = p + ".zip"
				if os.path.exists(file):
					os.remove(file)
				# list the feats files
				feat_files = glob.glob(feats_path + "*_{}.feats".format(n))
				for file in feat_files:
					os.remove(file)


		datasets = get_dirs(path)
		if "features" in datasets:
			datasets.remove("features")
		return render(request, "managedata.html", {"name": name, "datasets": datasets})

class Performance(View):

	def get(self, request, *args, **kwargs):
		name = request.session['login']
		return render(request, "performance.html", {"name": name})

	def post(self, request, *args, **kwargs):
		pass