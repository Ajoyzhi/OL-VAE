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
        h1 = torch.sigmoid(self.fc1(x)) # 5 * 1
        h2 = self.fc2(h1) # 1-dim
        return h1, h2

    def decode(self, h2):
        # encoder和decoder完全对称的结构
        h3 = self.fc3(h2)
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        h1, h2 = self.encode(x.view(-1, 1))
        return h1, h2, self.decode(h2)