import torch
from torch import nn
from torch.nn import functional as F

class VAE_KDD99(nn.Module):
    def __init__(self):
        super(VAE_KDD99, self).__init__()

        self.fc1 = nn.Linear(15, 50)
        self.fc21 = nn.Linear(50, 15)
        self.fc22 = nn.Linear(50, 15)
        self.fc3 = nn.Linear(15, 50)
        self.fc4 = nn.Linear(50, 15)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        # Ajoy 求 e^(0.5*logvar) 相当于求var^0.5(方差)
        std = torch.exp(0.5*logvar)
        # Ajoy 生成与std大小一致的服从标准正态分布的数据 epsilon
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 15))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar