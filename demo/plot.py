# _*_ coding:utf-8 _*_
import torch
import matplotlib.pyplot as plt
from other.path import Picture, pro_root
from matplotlib import font_manager
import numpy as np
import operator

from mytest.VAE_AE_test import my_bar
import csv
"""
PI = 3.141592657
# 数据
x = torch.unsqueeze(torch.linspace(-10, 10, 500), dim=1)
# y为标准正态分布，加入部分均匀分布噪声
y = 1/torch.sqrt(torch.tensor(2*PI)) * torch.exp(-x.pow(2)/2) + torch.rand(x.size())

# test ones
a = torch.ones(10, 1)
print(a.data.shape)

# test range()
for i in range(1, 10):
    print(i)
"""
"""
# test plot
def my_plot(org_data, online_data, name: str):
    line1, = plt.plot(range(len(org_data)), org_data, 'b-', label='origin')
    line2, = plt.plot(range(len(online_data)), online_data, 'r--', label='online')
    plt.xlabel('batch')
    plt.ylabel(name)
    plt.title(name + ' of original VAE vs. online VAE')
    font = {'family':'SimHei',
             'weight':'normal',
             'size':15}
    plt.legend(handles=[line1, line2], prop=font) # 不指定位置，则选择不遮挡图像位置
"""
"""
# test bar
bar_width = 0.1
x_data = [0.3, 0.6]
x_label = ['x1', 'x2']
y = [1, 2]
plt.bar(x_data[0], height=y[0], width=bar_width, color='w', hatch='*', edgecolor='k', label='x1')
plt.bar(x_data[1], height=y[1], width=bar_width, color='w', hatch='+', edgecolor='k', label='x2')
plt.xlim((0.0, 1.0))
plt.xticks(x_data, x_label)
plt.legend()
plt.show()
"""
"""
# test figure
# 创建画布
fig1 = plt.figure('Figure1',figsize = (6,4)).add_subplot(111)
fig1.plot([1,2,3,4],[5,6,7,8])
# 创建画布
fig2 = plt.figure('Figure2',figsize = (6,4)).add_subplot(111)
fig2.plot([4,5,2,1],[3,6,7,8])

# 创建标签
fig1.set_title('Figure1')
fig2.set_title('Figure2')
fig1.set_xlabel('This is x axis')
fig1.set_ylabel('This is y axis')

plt.show()
"""

# plot
# read data from the file
file_path = "F:/vae_ae_kdd_comp.csv"
# accurancy	precision	recall	FPR	MCC	detection time	train time
data = np.loadtxt(file_path, dtype=float, delimiter=",", skiprows=0)
print("data:", data)

acc = data[:, 0]
pre = data[:, 1]
recall = data[:, 2]
FPR = data[:, 3]
MCC= data[:, 4]
detect_time = data[:, 5]
train_time = data[:, 6]
print("acc:", acc,
      "pre:", pre,
      "recall:", recall,
      "FPR:", FPR,
      "MCC:", MCC,
      "detection time:", detect_time,
      "train_time:", train_time)
my_bar(acc, "accurancy")
my_bar(pre, "precision")
my_bar(recall, "recall")
my_bar(FPR, "FPR")
my_bar(MCC, 'MCC')
my_bar(detect_time, "detection time")
my_bar(train_time, "train time")

"""
# plot combine1:pkt_rate, byte_rate, rx_prate, rx_brate, tx_prate, tx_brate
file_path = pro_root + "normalization1.csv"
data = np.loadtxt(file_path, dtype=float, delimiter=',', skiprows=0, usecols=(6,7,10,11,14,15))

pkt_rate1 = data[:, 0]
byte_rate1 = data[:, 1]
rx_prate1 = data[:, 2]
rx_brate1 = data[:, 3]
tx_prate1 = data[:, 4]
tx_brate1 = data[:, 5]

plt.plot(range(len(pkt_rate1)), pkt_rate1)
plt.title("pkt_rate1")
plt.show()
plt.plot(range(len(byte_rate1)), byte_rate1)
plt.title("byte_rate1")
plt.show()
plt.plot(range(len(rx_prate1)), rx_prate1)
plt.title("rx_prate1")
plt.show()
plt.plot(range(len(rx_brate1)), rx_brate1)
plt.title("rx_brate1")
plt.show()
plt.plot(range(len(tx_prate1)), tx_prate1)
plt.title("tx_prate1")
plt.show()
plt.plot(range(len(tx_brate1)), tx_brate1)
plt.title("tx_brate1")
plt.show()
"""
def myplot(y, name):
    plt.plot(range(len(y)), y)
    plt.title(name)
    plt.show()
"""
file_path = pro_root + "normalization1.csv"
id = np.loadtxt(file_path,dtype="str", delimiter=',', skiprows=0, usecols=(1,2,3))
data = np.loadtxt(file_path, dtype=float, delimiter=',', skiprows=0, usecols=(6,7,10,11,14,15))
# 选择对应的特征
pkt_rate1 = []
byte_rate1 = []
rx_prate1 = []
rx_brate1 = []
tx_prate1 = []
tx_brate1 = []
pkt_rate2 = []
byte_rate2 = []
rx_prate2 = []
rx_brate2 = []
tx_prate2 = []
tx_brate2 = []
pkt_rate3 = []
byte_rate3 = []
rx_prate3 = []
rx_brate3 = []
tx_prate3 = []
tx_brate3 = []
pkt_rate4 = []
byte_rate4 = []
rx_prate4 = []
rx_brate4 = []
tx_prate4 = []
tx_brate4 = []
pkt_rate5 = []
byte_rate5 = []
rx_prate5 = []
rx_brate5 = []
tx_prate5 = []
tx_brate5 = []
for i in range(len((id))):
    if (id[i] == ['1', '2', '00:00:00:00:00:01']).all():
        pkt_rate1.append(data[i][0])
        byte_rate1.append(data[i][1])
        rx_prate1.append(data[i][2])
        rx_brate1.append(data[i][3])
        tx_prate1.append(data[i][4])
        tx_brate1.append(data[i][5])
    if (id[i] == ['1', '1', '00:00:00:00:00:01']).all():
        pkt_rate2.append(data[i][0])
        byte_rate2.append(data[i][1])
        rx_prate2.append(data[i][2])
        rx_brate2.append(data[i][3])
        tx_prate2.append(data[i][4])
        tx_brate2.append(data[i][5])
    if (id[i] == ['1', '3', '00:00:00:00:00:0f']).all():
        pkt_rate3.append(data[i][0])
        byte_rate3.append(data[i][1])
        rx_prate3.append(data[i][2])
        rx_brate3.append(data[i][3])
        tx_prate3.append(data[i][4])
        tx_brate3.append(data[i][5])
    if (id[i] == ['1', '3', '00:00:00:00:00:05']).all():
        pkt_rate4.append(data[i][0])
        byte_rate4.append(data[i][1])
        rx_prate4.append(data[i][2])
        rx_brate4.append(data[i][3])
        tx_prate4.append(data[i][4])
        tx_brate4.append(data[i][5])
    if (id[i] == ['1', '0','0']).all():
        pkt_rate5.append(data[i][0])
        byte_rate5.append(data[i][1])
        rx_prate5.append(data[i][2])
        rx_brate5.append(data[i][3])
        tx_prate5.append(data[i][4])
        tx_brate5.append(data[i][5])
# ['1', '2', '00:00:00:00:00:01']
myplot(pkt_rate1, "pkt_rate1")
myplot(byte_rate1, "byte_rate1")
myplot(rx_prate1, "rx_prate1")
myplot(rx_brate1, "rx_brate1")
myplot(tx_prate1, "tx_prate1")
myplot(tx_brate1, "tx_brate1")
#['1', '1', '00:00:00:00:00:01']
myplot(pkt_rate2, "pkt_rate2")
myplot(byte_rate2, "byte_rate2")
myplot(rx_prate2, "rx_prate2")
myplot(rx_brate2, "rx_brate2")
myplot(tx_prate2, "tx_prate2")
myplot(tx_brate2, "tx_brate2")
# ['1', '3', '00:00:00:00:00:0f']
myplot(pkt_rate3, "pkt_rate3")
myplot(byte_rate3, "byte_rate3")
myplot(rx_prate3, "rx_prate3")
myplot(rx_brate3, "rx_brate3")
myplot(tx_prate3, "tx_prate3")
myplot(tx_brate3, "tx_brate3")
# ['1', '3', '00:00:00:00:00:05']
myplot(pkt_rate4, "pkt_rate4")
myplot(byte_rate4, "byte_rate4")
myplot(rx_prate4, "rx_prate4")
myplot(rx_brate4, "rx_brate4")
myplot(tx_prate4, "tx_prate4")
myplot(tx_brate4, "tx_brate4")
# ['1', '0','0']
myplot(pkt_rate5, "pkt_rate5")
myplot(byte_rate5, "byte_rate5")
myplot(rx_prate5, "rx_prate5")
myplot(rx_brate5, "rx_brate5")
myplot(tx_prate5, "tx_prate5")
myplot(tx_brate5, "tx_brate5")
"""
"""
file_path = pro_root + "normalization2.csv"
id = np.loadtxt(file_path,dtype="str", delimiter=',', skiprows=0, usecols=(1,2,3))
data = np.loadtxt(file_path, dtype=float, delimiter=',', skiprows=0, usecols=(6,7,10,11,14,15))
pkt_rate6 = []
byte_rate6 = []
rx_prate6 = []
rx_brate6 = []
tx_prate6 = []
tx_brate6 = []
for i in range(len((id))):
    if (id[i] == ['2', '3', '00:00:00:00:00:01']).all():
        pkt_rate6.append(data[i][0])
        byte_rate6.append(data[i][1])
        rx_prate6.append(data[i][2])
        rx_brate6.append(data[i][3])
        tx_prate6.append(data[i][4])
        tx_brate6.append(data[i][5])
# ['2', '3', '00:00:00:00:00:01']
myplot(pkt_rate6, "pkt_rate6")
myplot(byte_rate6, "byte_rate6")
myplot(rx_prate6, "rx_prate6")
myplot(rx_brate6, "rx_brate6")
myplot(tx_prate6, "tx_prate6")
myplot(tx_brate6, "tx_brate6")
"""
"""
file_path = pro_root + "normalization2.csv"
id = np.loadtxt(file_path,dtype="str", delimiter=',', skiprows=0, usecols=(1,2,3))
data = np.loadtxt(file_path, dtype=float, delimiter=',', skiprows=0, usecols=(6,7,10,11,14,15))
pkt_rate7 = []
byte_rate7 = []
rx_prate7 = []
rx_brate7 = []
tx_prate7 = []
tx_brate7 = []
for i in range(len((id))):
    if (id[i] == ['2', '3', '00:00:00:00:00:01']).all():
            pkt_rate7.append(data[i][0])
            byte_rate7.append(data[i][1])
            rx_prate7.append(data[i][2])
            rx_brate7.append(data[i][3])
            tx_prate7.append(data[i][4])
            tx_brate7.append(data[i][5])
# ['3', '3', '00:00:00:00:00:01']
myplot(pkt_rate7, "pkt_rate7")
myplot(byte_rate7, "byte_rate7")
myplot(rx_prate7, "rx_prate7")
myplot(rx_brate7, "rx_brate7")
myplot(tx_prate7, "tx_prate7")
myplot(tx_brate7, "tx_brate7")
"""
"""
file_path = pro_root + "normalization5.csv"
id = np.loadtxt(file_path,dtype="str", delimiter=',', skiprows=0, usecols=(1,2,3))
data = np.loadtxt(file_path, dtype=float, delimiter=',', skiprows=0, usecols=(6,7,10,11,14,15))
pkt_rate8 = []
byte_rate8 = []
rx_prate8 = []
rx_brate8 = []
tx_prate8 = []
tx_brate8 = []
for i in range(len((id))):
    if (id[i] == ['5', '4', '00:00:00:00:00:01']).all():
        pkt_rate8.append(data[i][0])
        byte_rate8.append(data[i][1])
        rx_prate8.append(data[i][2])
        rx_brate8.append(data[i][3])
        tx_prate8.append(data[i][4])
        tx_brate8.append(data[i][5])
# ['5', '4', '00:00:00:00:00:01']
myplot(pkt_rate8, "pkt_rate8")
myplot(byte_rate8, "byte_rate8")
myplot(rx_prate8, "rx_prate8")
myplot(rx_brate8, "rx_brate8")
myplot(tx_prate8, "tx_prate8")
myplot(tx_brate8, "tx_brate8")
"""

for i in range(1):
    plt.axhline(y=2)
    y = [1,2,3,4]
    label = [0,0,1,1]
    plt.scatter(x=range(len(y)), y=y, c=label)
    plt.show()
    plt.close()

