import torch
import torch.utils.data as Data
from network.VAE_1D import VAE_1D
from network.VAE_1D_Online import VAE_Online
from optim.VAE_1D_trainer import VAE_1D_trainer
from optim.VAE_1D_Online_trainer import OLVAE_1D_trainer
from other.path import Picture
import matplotlib.pyplot as plt
import time
import numpy as np

# 生成训练数据：高斯分布 + (0,1)均匀分布噪声 100个
x_train_temp = torch.ones(1000, 1)
x_train_normal = torch.normal(x_train_temp, 2)# mu=1,std=2的离散高斯分布
x_train_noise = torch.rand_like(x_train_normal)# (0,1)均匀分布
x_train = x_train_normal + x_train_noise
y_train = torch.zeros_like(x_train)
# 将数据封装为dataset和dataloader
train_dataset = Data.TensorDataset(x_train, y_train)
train_loader = Data.DataLoader(
    dataset=train_dataset,
    batch_size=100,
    shuffle=True,
    num_workers=0
)
"""
# 生成测试数据：不同高斯分布 + (0,1)均匀分布噪声 300个
x_test_temp = torch.ones(150, 1)
x_test_normal = torch.normal(x_test_temp, 2) # 正常数据
y_normal = torch.zeros_like(x_test_normal)
x_test_adnormal = torch.normal(2 * x_test_temp, 3) # 异常数据
y_abnormal = torch.ones_like(x_test_adnormal)

x_test = torch.cat((x_test_normal, x_test_adnormal), 0).type(torch.FloatTensor) # 正常数据与异常数据拼接
y_test = torch.cat((y_normal, y_abnormal), 0).type(torch.FloatTensor) # 标记拼接
test_dataset = Data.TensorDataset(x_test, y_test)
test_loader = Data.DataLoader(
    dataset=test_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0
)
"""
"""
# 没有什么好显示的
# 画图显示训练数据
plt.scatter(x_train.data.numpy(), x_train.data.numpy(), s=100, lw=0, cmap='RdYlGn')
plt.show()
# 画图显示测试数据
plt.scatter(x_test.data.numpy(), x_test.data.numpy(), c=y_test.data.numpy(), s=100, lw=0, cmap='RdYlGn')
plt.show()
"""

# 生成原始VAE网络
VAE_1D_net = VAE_1D()
# 训练网络，其实trainer中并未使用testloader
vae_1D_trainer = VAE_1D_trainer(VAE_1D_net, train_loader, epochs=20)
vae_1D_trainer.train()
vae_1D_trainer.test()

# 生成改进VAE网络
VAE_Online_net = VAE_Online()
vae_Online_trainer = OLVAE_1D_trainer(VAE_1D_net, train_loader, epochs=20)
vae_Online_trainer.train()
vae_Online_trainer.test()

# 获取数据，并画图
org_train_loss = vae_1D_trainer.train_loss
org_train_mu = vae_1D_trainer.train_mu
org_train_var = vae_1D_trainer.train_var
org_train_time = vae_1D_trainer.train_time
org_test_time = vae_1D_trainer.test_time

ol_train_loss = vae_Online_trainer.train_loss
ol_train_mu = vae_Online_trainer.train_mu
ol_train_var = vae_Online_trainer.train_var
ol_train_time = vae_Online_trainer.train_time
ol_test_time = vae_Online_trainer.test_time

# 画折线图函数
def my_plot(org_data, online_data, name: str):
    # 获取本地时间
    real_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    line1, = plt.plot(range(len(org_data)), org_data, 'b-', label='origin')
    line2, = plt.plot(range(len(online_data)), online_data, 'r--', label='online')
    plt.xlabel('batch')
    plt.ylabel(name)
    plt.title(name + ' of original VAE vs. online VAE')
    font = {'family':'SimHei',
             'weight':'normal',
             'size':15}
    plt.legend(handles=[line1, line2], prop=font) # 不指定位置，则选择不遮挡图像位置
    # plt.savefig(Picture + real_time + name + '.jpg')
    plt.show()

# 画柱状图 柱间间距如何调整？？
def my_bar(y:tuple, name:str):
    bar_width = 0.1
    x = ['original VAE', 'online VAE']
    plt.bar(x[0], height=y[0], width=bar_width, hatch='x', label="origin", edgecolor='k')
    plt.bar(x[1], height=y[1], width=bar_width, hatch='+', label="online", edgecolor='k')
    plt.ylabel(name)
    plt.title(name + ' of original VAE vs. online VAE')
    plt.legend()
    real_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    # plt.savefig(Picture + real_time + name)
    plt.show()

# 画 train_loss的图
my_plot(org_train_loss, ol_train_loss, 'train_loss')
# 画train_mu的图
my_plot(org_train_mu, ol_train_mu, 'train_mu')
# 画train_var的图
my_plot(org_train_var, ol_train_var, 'train_var')
# 画train_time
train_time = (org_train_time, ol_train_time)
my_bar(train_time, "train time")
# 画test_time(一个值使用柱状图表示)
test_time = (org_test_time, ol_test_time)
my_bar(test_time, "test time")
