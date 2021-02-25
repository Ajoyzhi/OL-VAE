import torch
from scipy.stats import multivariate_normal
import numpy as np
"""
# 测试标准正态分布概率计算函数
mu = torch.tensor([-1.1393e-03, -6.6651e-04,  8.1890e-04, -6.0028e-04,  1.2790e-03,
        -1.2880e-05, -1.9163e-03,  4.2241e-05,  2.3298e-03,  1.7627e-04,
         4.4342e-04,  6.4159e-04,  6.3276e-04,  2.9581e-04, -6.8058e-04])
std = torch. tensor([1.0004, 0.9990, 1.0001, 1.0012, 1.0004, 1.0006, 1.0011, 0.9999, 1.0005,
        1.0000, 0.9993, 1.0017, 0.9991, 1.0008, 0.9986])
con = std *std

x = torch.tensor([10,10,10,10,10,10,10,10,10,10,10,10,10,10,10])
p = multivariate_normal.pdf(x, mu, con)
print(p)
"""

# 测试np.array
a = np.arange(6).reshape(3, 2)
for item in a:
    print(item)