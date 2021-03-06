import torch
from torch import nn
from torch.nn import functional as F

class VAE_Online(nn.Module):
    def __init__(self):
        super(VAE_Online, self).__init__()

        self.fc1 = nn.Linear(1, 5)
        # 单支神经网络
        self.fc2 = nn.Linear(5, 1)
        self.fc3 = nn.Linear(1, 5)
        self.fc4 = nn.Linear(5, 1)

    def encode(self, x):
        h1 = F.sigmoid(self.fc1(x)) # 5 * 1
        h2 = self.fc2(h1) # 1-dim

        # 计算参数mu和sigma^2
        # 公式计算没有问题，但是维度问题可以调整
        w1 = self.fc1.weight # 5 * 1
        w2 = self.fc2.weight # 1 * 5
        dh1 = h1.transpose(0,1) * (1 - h1) * w1 # 5 * 1 = (1*5) * (5*1) * (5*1)
        dencoder = w2 * dh1 # 1 * 1 = (1*5) * (5*1)
        ddencoder = w2 * w1 * dh1.transpose(0,1) * (1 - 2*h1) # 1 * 1=(1*5)*(5*1)*(1*5)*(5*1)
        same = h2 / (dencoder.pow(2)-ddencoder * h2)
        mu = x + dencoder * same
        var = h2 * same # sigma^2

        return mu, var

    def reparameterize(self, mu, var):
        std = torch.sqrt(var)
        # Ajoy 生成与std大小一致的服从标准正态分布的数据 epsilon
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        # encoder和decoder完全对称的结构
        h3 = self.fc3(z)
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, var = self.encode(x.view(-1, 1))
        z = self.reparameterize(mu, var)
        return self.decode(z), mu, var