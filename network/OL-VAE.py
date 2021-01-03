import torch
from torch import nn
import torch.nn.functional as F

# 可以加入对输入特征数量的选择
class OL_VAE(nn.Module):
    def __init__(self):
        super(OL_VAE, self).__init__()
        self.fc1 = nn.Linear(15, 50)
        # 只使用一层计算均值和方差
        self.fc2 = nn.Linear(50, 15)
        self.fc3 = nn.Linear(15, 50)
        self.fc4 = nn.Linear(50, 15)

    def encode(self, x):
        # 使用sigmoid激活或者使用tanh激活
        h1 = torch.sigmoid(self.fc1(x))
        h2 = self.fc2(h1)
        return h2

    # 从encoder部分中计算出mu和std
    def get_mu_std(self):
        # 加入对于的表达式
        mu = 0.0
        std = 0.0
        return mu, std

    def reparameterize(self, mu, std):
        # Ajoy 生成与std大小一致的服从标准正态分布的数据 epsilon
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        h2 = self.encode(x.view(-1, 15))
        mu, std = self.get_mu_std()
        z = self.reparameterize(mu, std)
        return self.decode(z), mu, std