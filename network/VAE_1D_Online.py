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
        h1 = torch.sigmoid(self.fc1(x)) # batch*5
        # print("h1_size:", h1.shape)
        h2 = self.fc2(h1) # batch*1
        # print("h2_size:", h2.shape)

        # 计算参数mu和sigma^2
        w1 = self.fc1.weight # 5 * 1
        # print("w1_size:", w1.data.shape)
        w2 = self.fc2.weight # 1 * 5
        # print("w2_size:", w2.data.shape)

        # 计算h‘(x)，计算为(batch*1)
        """ 
        de_tmp1 = h1.mm(w1) # batch*1 = (batch*5)*(5*1)
        de_tmp2 = (1-h1).mm(torch.transpose(w2, 0, 1)) # batch*1 = (batch*5)*(5*1)
        dencoder = de_tmp1 * de_tmp2 # batch*1
        """
        # 另一种实现
        de_tmp1 = h1.mm(torch.transpose(w2, 0, 1)) # batch*1 = (batch*5)*(5*1)
        de_tmp2 = (1-h1).mm(w1) # batch*1 = (batch*5)*(5*1)
        dencoder = de_tmp1 * de_tmp2

        # 计算h''(x)，计算为(batch*1)
        dde_tmp1 = (1-2*h1).mm(w1) # batch*1 = (batch*5)*(5*1)
        ddencoder = dencoder * dde_tmp1 # batch*1
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
        # print("X_SIZE:", x.data.shape)# batch * 1
        mu, var = self.encode(x.view(-1, 1))
        z = self.reparameterize(mu, var)
        return self.decode(z), mu, var