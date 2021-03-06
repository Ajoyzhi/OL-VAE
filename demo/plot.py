import torch
import matplotlib.pyplot as plt
from other.path import Picture
from matplotlib import font_manager
import numpy as np

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
# test plot
x = [1,2,3,4,5,6,7,8,9,10]
y1 = [10,9,8,7,6,5,4,3,2,1]
y2 = [1,2,3,4,5,6,7,8,9,10]
# my_font=font_manager.FontProperties(fname=r"c:\windows\fonts\simsun.ttc",size=30)
# plt.xlabel(u'X轴',fontproperties=my_font)
# plt.ylabel(u'Y轴',fontproperties=my_font)
A, =plt.plot(x, y1, 'b-', linewidth=0.5, label='org')
B, =plt.plot(x, y2, 'r--', linewidth=0.5, label='ol')
font1={'family':'SimHei',
       'weight':'normal',
       'size':15,}
legend=plt.legend(handles=[A, B], prop=font1, loc='upper right')
plt.show()
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
