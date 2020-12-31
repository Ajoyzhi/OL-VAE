import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F

PI = 3.141592657

# 数据
x = torch.unsqueeze(torch.linspace(-10, 10, 500), dim=1)
# y为标准正态分布，加入部分均匀分布噪声
y = 1/torch.sqrt(torch.tensor(2*PI)) * torch.exp(-x.pow(2)/2) + torch.rand(x.size())

x, y = Variable(x), Variable(y)

# 显示
plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()

# 定义网络结构 sigmoid激活函数
class Net_sigmoid(torch.nn.Module):
    def __init__(self, n_input, n_hiddern, n_output):
        super(Net_sigmoid, self).__init__()
        self.hidden = torch.nn.Linear(n_input, n_hiddern)
        self.output = torch.nn.Linear(n_hiddern, n_output)

    def forward(self, x):
        x = self.hidden(x)
        x = torch.sigmoid(x)
        x = self.output(x)

        return x
# tanh激活函数
class Net_tanh(torch.nn.Module):
    def __init__(self, n_input, n_hiddern, n_output):
        super(Net_tanh, self).__init__()
        self.hidden = torch.nn.Linear(n_input, n_hiddern)
        self.output = torch.nn.Linear(n_hiddern, n_output)

    def forward(self, x):
        x = self.hidden(x)
        x = torch.tanh(x)
        x = self.output(x)

        return x

# relu激活
class Net_relu(torch.nn.Module):
    def __init__(self, n_input, n_hiddern, n_output):
        super(Net_relu, self).__init__()
        self.hidden = torch.nn.Linear(n_input, n_hiddern)
        self.output = torch.nn.Linear(n_hiddern, n_output)

    def forward(self, x):
        x = self.hidden(x)
        x = F.relu(x)
        x = self.output(x)

        return x

# 网络
net_sigmoid = Net_sigmoid(1, 10, 1)
net_tanh = Net_tanh(1, 10 ,1)
net_relu = Net_relu(1, 10, 1)
# 优化器
optimizer_sigmoid = torch.optim.SGD(net_sigmoid.parameters(), lr=0.05)
optimizer_tanh = torch.optim.SGD(net_tanh.parameters(), lr=0.05)
optimizer_relu = torch.optim.SGD(net_relu.parameters(), lr=0.05)
# 损失函数
loss_func = torch.nn.MSELoss()

# tanh训练损失最小
# sigmoid训练
plt.ion()
plt.show()
for i in range(100):
    prediction_sigmoid = net_sigmoid(x)
    loss_sigmoid = loss_func(prediction_sigmoid, y)
    optimizer_sigmoid.zero_grad()
    loss_sigmoid.backward()
    optimizer_sigmoid.step()

    if i % 10 == 0:
        plt.subplot(1, 3, 1)
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction_sigmoid.data.numpy(), '-r', lw=5)
        plt.text(0, 0, 'loss=%.4f' % loss_sigmoid.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.pause(0.1)

# tanh训练
for j in range(100):
    prediction_tanh = net_tanh(x)
    loss_tanh = loss_func(prediction_tanh, y)
    optimizer_tanh.zero_grad()
    loss_tanh.backward()
    optimizer_tanh.step()

    if j % 10 == 0:
        plt.subplot(1, 3, 2)
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction_tanh.data.numpy(), '-b', lw=5)
        plt.text(0, 0,  'loss=%.4f' % loss_tanh.data.numpy(), fontdict={'size': 20, 'color': 'blue'})
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.pause(0.1)

# relu训练
for t in range(100):
    prediction_relu = net_relu(x)
    loss_relu = loss_func(prediction_relu, y)
    optimizer_relu.zero_grad()
    loss_relu.backward()
    optimizer_relu.step()

    if t % 10 == 0:
        plt.subplot(1, 3, 3)
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction_relu.data.numpy(), '-g', lw=5)
        plt.text(0, 0, 'loss=%.4f' % loss_relu.data.numpy(), fontdict={'size': 20, 'color': 'green'})
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.pause(0.1)
plt.ioff()
plt.show()

# 测试
# 生成数据
x = torch.unsqueeze(torch.linspace(18, 20, 20), dim=1)
print('\nx:', x)
# y为标准正态分布，加入部分均匀分布噪声  真实标签
y = 1/torch.sqrt(torch.tensor(2*PI)) * torch.exp(-x.pow(2)/2) + torch.rand(x.size())
print('\ny:', y)

x = Variable(x)

prediction_sigmoid = net_sigmoid(x)
print('\nsigmoid:', prediction_sigmoid.data)
print('\ndiff_sigmoid:', prediction_sigmoid.data - y)

prediction_tanh = net_tanh(x)
print('\ntanh:', prediction_tanh.data)
print('\ndiff_tanh:', prediction_tanh.data - y)

prediction_relu = net_relu(x)
print('\nrelu:', prediction_relu)
print('\ndiff_relu:', prediction_relu.data - y)