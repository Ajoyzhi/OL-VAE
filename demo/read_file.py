import pandas as pd
# coding=UTF-8
import numpy as np
from sklearn import preprocessing

from dataset.KDD99.processed import util

"""

a = [1,2,3]
b = [4,5,6]
c = [7,8,9]

a.append(b)
c.extend(b)

print(a)
print(c)

my_matrix = np.loadtxt(open("data.csv"), delimiter=",", skiprows=0)
print(my_matrix)

a = my_matrix.shape[0]
print("the number of rows:", a)

def sample(array, number):
    # 将矩阵转化为list，随机采样，返回array 
    rand_arr = np.arange(array.shape[0])
    np.random.shuffle(rand_arr)
    data_sample = array[rand_arr[0:number]]

    return data_sample

data = sample(my_matrix, 2)
# 获取除最后一行的所有列
print("data:", data.shape)
data = np.array(data)
print("array_data:", data)

data1 = [row[:-1] for row in data]
data1 = np.array(data1)

# 获取第2-3列（index，且[)）
data2 = [row[1:3] for row in data]
print(data2)
"""
source_final  = "F:/pycharm/OL_VAE/dataset/KDD99/raw/kddcup.data_10_percent_corrected_number.cvs"

# 从文件中加载所有数据
data_label = np.loadtxt(source_final, delimiter=",", skiprows=0)
#print("data_label:", data_label[0])
#print("data_label:", data_label[3])
#print("data_label size:", data_label.shape)
data_sample = data_label[0:10]

#print("data_simple:", data_sample)
x_data = util.select_features(data_sample, 15)
# print("x_data:", x_data)


min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(x_data)
print("x_min_max:", X_train_minmax)

X_z_score = preprocessing.scale(X_train_minmax)
print("z_score:", X_z_score)