import torch
from torch import nn
from torch.nn import functional as F
"""
    作为VAE_1D_Online的对比算法
"""
class VAE_1D(nn.Module):
    def __init__(self):
        super(VAE_1D, self).__init__()

        self.fc1 = nn.Linear(1, 5)
        self.fc21 = nn.Linear(5, 1)
        self.fc22 = nn.Linear(5, 1)
        self.fc3 = nn.Linear(1, 5)
        self.fc4 = nn.Linear(5, 1)

    def encode(self, x):
        h1 = torch.sigmoid(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        # Ajoy 生成与std大小一致的服从标准正态分布的数据 epsilon
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        # encoder和decoder完全对称的结构
        h3 = self.fc3(z)
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 1))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar