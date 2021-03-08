import os
import zipfile
from Server.settings import data_root
# 保存数据集
def save_dataset(name, file):
    save_dir = data_root + name
    filename = file.name
    # 判断是否已经存在用户专有目录，没有就创建
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = save_dir + "/" + filename
    if os.path.exists(file_path):
        return True
    try:
        with open(file_path, 'wb') as f:
            # file.read() 是一次性把文件读进内存，如果文件很大就不好，所以需要采用分片
            # f.write(file.read())
            for chunk in file.chunks():
                f.write(chunk)
        # 解压文件
        fz = zipfile.ZipFile(file_path, 'r')
        for file in fz.namelist():
            fz.extract(file, save_dir)
        return True
    except:
        return False

# import glob
# 给定一个路径，返回该路径下的所有文件夹
def get_dirs(path):
    # filename = glob.glob(path)
    filename = os.listdir(path)
    # 注意判断是否为文件夹或这文件要用绝对路径
    directory = [name for name in filename if os.path.isdir(path + name)]
    return directory

def get_files(path):
    filename = os.listdir(path)
    files = [name for name in filename if os.path.isfile(path + name)]
    return files
