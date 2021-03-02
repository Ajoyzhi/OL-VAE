import torch

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