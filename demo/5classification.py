import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

n_data = torch.ones(100, 2)
x0 = torch.normal(2*n_data, 1)  # class0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)           # class0 y data (tensor), shape=(100, 1)
x1 = torch.normal(-2*n_data, 1) # class1 x data (tensor), shape=(100, 2)
y1 = torch.ones(100)           # class1 y data (tensor), shape=(100, 1)
# cat((A,B),0)将TensorA和B按照行拼接；
# cat((A,B),1)将tensor A和B 按照列拼接
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # FloatTensor = 32-bit floating
y = torch.cat((y0, y1),).type(torch.LongTensor)     # LongTensor = 64-bit integer

x, y= Variable(x),Variable(y)
# 画散点图
plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
plt.show()

# method 1
class Net(torch.nn.Module):
    # 只定义网络结构，有多少输入节点、输出节点、中间层节点
    def __init__(self, n_features, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    # 具体的计算过程
    def forward(self, x):
        # 激活函数会截断值
        # 函数
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x
# [0,1]为标签1；[1,0]为标签0——》只有某一位上为1，说明分类器的预测为该类
net1 = Net(2, 10, 2)
print(net1)

# method 2
net = torch.nn.Sequential(
    torch.nn.Linear(2, 10),
    # 类对象
    torch.nn.ReLU(),
    torch.nn.Linear(10, 2),
)
print(net)

plt.ion()
plt.show()
# 网络参数 net.parameters()
optimizer = torch.optim.SGD(net.parameters(), lr = 0.02)
# 交叉熵损失函数输出的为概率[0.1,0.2.0.7]表示大概率分类为第3类
loss_func =torch.nn.CrossEntropyLoss()

for t in range(100):
    out = net(x)
    # 真实值在后
    loss = loss_func(out, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 2 == 0:
        plt.cla()
        # softmax将输出数据转化为概率;选择概率最大的标签？？
        prediction = torch.max(F.softmax(out), 1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        # 预测数据图
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y)/200
        plt.text(1.5, -4, 'Accuracy = %.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()

