import torch
from torch import nn 

class AE_KDD99(nn.Module):
    def __init__(self):
        super(AE_KDD99, self).__init__()
        
        self.fe1 = nn.Linear(15, 50)
        self.fe2 = nn.Linear(50, 15)
        self.fd1 = nn.Linear(15, 50)
        self.fd2 = nn.Linear(50, 15)
        
    def forward(self, x):
        h11 = torch.sigmoid(self.fe1(x))
        h12 = self.fe2(h11)
        h21 = torch.sigmoid(self.fd1(h12))
        h22 = self.fd2(h21)
        
        return h12, h22