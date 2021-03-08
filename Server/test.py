import os
data_root = "D:/project/data/server/"
def test():

    path = data_root + "hanjun/"
    filename = os.listdir(path)
    directory = [name for name in filename if os.path.isdir(path + name)]
    print(directory)
import numpy as np
def test_choice():
    nums = [i for i in range(10)]
    idx = np.random.choice(nums,size = 3)
    print(idx)

if __name__ == "__main__":
    # test()
    test_choice()