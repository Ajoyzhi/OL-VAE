import torch
from torch.autograd import Variable
import torch.utils.data as Data
from network.VAE_1D import VAE_1D
from network.VAE_1D_Online import VAE_Online
from optim.VAE_1D_trainer import VAE_1D_trainer
from optim.VAE_1D_Online_trainer import OLVAE_1D_trainer
import matplotlib.pyplot as plt

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
    batch_size=10,
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

# 生成网络
VAE_1D_net = VAE_1D()
VAE_Online_net = VAE_Online()

# 训练网络，其实trainer中并未使用testloader
vae_1D_trainer = VAE_1D_trainer(VAE_1D_net, train_loader)
vae_Online_trainer = OLVAE_1D_trainer(VAE_1D_net, train_loader)
vae_1D_trainer.train()
vae_1D_trainer.test()
vae_Online_trainer.train()
vae_Online_trainer.test()

# 获取数据，并画图
epoch = vae_1D_trainer.epochs
org_train_time = vae_1D_trainer.train_time
org_train_loss = vae_1D_trainer.train_loss
org_train_mu = vae_1D_trainer.train_mu
org_train_var = vae_1D_trainer.train_var
org_test_time = vae_1D_trainer.test_time

ol_train_time = vae_Online_trainer.train_time
ol_train_loss = vae_Online_trainer.train_loss
ol_train_mu = vae_Online_trainer.train_mu
ol_train_var = vae_Online_trainer.train_var
ol_test_time = vae_Online_trainer.test_time

# 画图函数
def my_plot(loc: int, name: str, org_data, online_data, epoch):
    my_plt = plt.subplot(loc)
    line1 = my_plt.plot(range(1, epoch+1), org_data, 'b-')
    line2 = my_plt.plot(range(1, epoch+1), online_data, 'r--')
    plt.xlabel('epoch')
    plt.ylabel(name)
    plt.title(name + 'of original VAE vs. online VAE')
    my_plt.legend(line1, ('org VAE'))
    my_plt.legend(line2, ('ol VAE'))

plt.figure(figsize=(12, 6))
# 画train_time的图
my_plot(151, 'train_time', org_train_time, ol_train_time, epoch)
# 画 train_loss的图
my_plot(152, 'train_loss', org_train_loss, ol_train_loss, epoch)
# 画train_mu的图
my_plot(153, 'train_mu', org_train_mu, ol_train_mu, epoch)
# 画train_var的图
my_plot(154, 'train_var', org_train_var, ol_train_var, epoch)
# 画test_time(一个值使用柱状图表示)
test_time = [org_test_time, ol_test_time]
x = ['original VAE', 'online VAE']
plt.subplot(155)
plt.bar(x, test_time)
plt.xlabel('type')
plt.ylabel('test time')
plt.title('test time of original VAE vs. online VAE')
plt.legend()
plt.show()
