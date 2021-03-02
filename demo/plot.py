import torch
import matplotlib.pyplot as plt
from other.path import Picture

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

# test plot
x = [1,2,3,4,5,6,7,8,9,10]
y1 = [10,9,8,7,6,5,4,3,2,1]
y2 = [1,2,3,4,5,6,7,8,9,10]
"""
line1 = plt.plot(x, y1, 'b-')
line2 = plt.plot(x, y2, 'r--')
plt.legend(handles = [line1, line2], labels=['org', 'ol'], loc='upper right')
# plt.savefig(Picture +'y.jpg')
plt.show()"""

import matplotlib.pyplot as plt
from matplotlib import font_manager
# my_font=font_manager.FontProperties(fname=r"c:\windows\fonts\simsun.ttc",size=30)
# plt.xlabel(u'X轴',fontproperties=my_font)
# plt.ylabel(u'Y轴',fontproperties=my_font)
A, =plt.plot(x, y1, 'b-', label='org')
B, =plt.plot(x, y2, 'r--', label='ol')
font1={'family':'SimHei',
       'weight':'normal',
       'size':15,}
legend=plt.legend(handles=[A, B], prop=font1, loc='upper right')
plt.show()
