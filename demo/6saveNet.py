import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

# 1D t0 2D
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2 * torch.randn(x.size())

x, y= Variable(x),Variable(y)

def save():
    # save net1
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    optimizer = torch.optim.SGD(net1.parameters(), lr = 0.5)
    loss_func = torch.nn.MSELoss()

    for t in range(100):
        prediction = net1(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 保存
    #torch.save(net1, 'net_pkl')                    # entire net 'net_pkl'就是网络名或者称为参数名，不是什么类型
    # torch.save(net1.state_dict(), 'net_param.pkl') # entire net parameters
    param_dict = net1.state_dict()
    # print(param_dict.keys())
    # print(param_dict.values())
    param_dict1 = net1.state_dict()
    new_dict = {}
    for key, value in param_dict.items():
        othe_value = param_dict1[key]
        temp = value + othe_value # 可计算
        new_dict[key] = temp

    print(new_dict)

    # 查看网络参数
    # param = list(net1.named_parameters())
    # print(param[0][1])
    # (name,param)
    # print(param)

    # 不支持这种方式下对网络参数进行修改
    # param1 = net1.named_parameters() # tuple
    # param2 = net1.named_parameters()

    # 可视化
    plt.figure(1, figsize=(10, 3))
    plt.subplot(131)
    plt.title('net1')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw = 5)
    #plt.show()

def restore_net():
    # 将net1的所有内容都保留下来
    net2 = torch.load('net_pkl')
    prediction = net2(x)
    # 可视化
    plt.figure(2, figsize=(10, 3))
    plt.subplot(132)
    plt.title('net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    #plt.show()

def restore_params():
    # 创建于net1相同的网络结构，加载net1的参数
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )

    net3.load_state_dict(torch.load("net_param.pkl"))

    prediction = net3(x)
    # 可视化
    plt.figure(3, figsize=(10, 3))
    plt.subplot(133)
    plt.title('net3')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    #plt.show()

save()
# restore_net()
# restore_params()
# plt.show()