import torch
from torch.autograd import  Variable
import torch.nn as nn

class VAE_Online(nn.Module):
    def __init__(self):
        super(VAE_Online, self).__init__()

        self.fc1 = nn.Linear(1, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        h1 = torch.sigmoid(self.fc1(x))
        h2 = self.fc2(h1)
        return h2

net1 = VAE_Online()
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2 * torch.randn(x.size())

x, y = Variable(x),Variable(y)
optimizer = torch.optim.SGD(net1.parameters(), lr=0.5)
loss_func = torch.nn.MSELoss()

for t in range(100):
    prediction = net1(x)
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(
    "x_size:", x.data.size(),
    "\nw1_size:", net1.fc1.weight.size(), # 5*1
    "\nb1_size:", net1.fc1.bias.size(), # 5*1
    "\nw2_size:", net1.fc2.weight.size(), # 1*5
    "\nb2_size:", net1.fc2.bias.size() # 1*5
)
# test mul
w1 = net1.fc1.weight # 5*1
w2 = net1.fc2.weight # 1*5
pro1 = w1.mm(w2) # 5*5=(5*1)*(1*5)
print("pro1:", pro1)
pro2 = w1 * torch.transpose(w2, 0, 1) * w2# 5*1=(5*1)*(5*1)
print("pro2:", pro2)
pro3 = torch.transpose(w1, 0, 1) * w2 # 1*5=(1*5)*(1*5)
print("pro3:", pro3)