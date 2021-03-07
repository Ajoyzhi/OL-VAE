import torch
import torch.utils.data as Data
from network.VAE_1D import VAE_1D
from network.VAE_1D_Online import VAE_Online
from optim.VAE_1D_trainer import VAE_1D_trainer
from optim.VAE_1D_Online_trainer import OLVAE_1D_trainer
from other.path import Picture
import matplotlib.pyplot as plt
import time

# 画折线图函数
def my_plot(org_data, online_data, name: str):
    line1, = plt.plot(range(len(org_data)), org_data, 'k*', label='origin')
    line2, = plt.plot(range(len(online_data)), online_data, 'k--', label='online')
    plt.xlabel('batch')
    plt.ylabel(name)
    plt.title(name + ' of original VAE vs. online VAE')
    font = {'family':'SimHei',
             'weight':'normal',
             'size':15}
    plt.legend(handles=[line1, line2], prop=font) # 不指定位置，则选择不遮挡图像位置

def my_bar(y:tuple, name:str):
    bar_width = 0.1
    x = [0.3, 0.6]
    x_label = ['original VAE', 'online VAE']
    plt.bar(x[0], height=y[0], width=bar_width, hatch='x', color='w', label="origin", edgecolor='k')
    plt.bar(x[1], height=y[1], width=bar_width, hatch='+', color='w', label="online", edgecolor='k')
    plt.xticks(x, x_label)
    plt.xlim((0.0, 1.0))
    plt.ylabel(name)
    plt.title(name + ' of original VAE vs. online VAE')
    plt.legend()

# 生成训练数据：高斯分布 1000个
x_train_temp = torch.ones(1000, 1)
x_train_normal = torch.normal(x_train_temp, 2)# mu=1,std=2的离散高斯分布
# x_train_noise = torch.rand_like(x_train_normal)# (0,1)均匀分布
# x_train = x_train_normal + x_train_noise
x_train = x_train_normal
y_train = torch.zeros_like(x_train)
# 将数据封装为dataset和dataloader
train_dataset = Data.TensorDataset(x_train, y_train)
train_loader = Data.DataLoader(
    dataset=train_dataset,
    batch_size=100,
    shuffle=True,
    num_workers=0,
    drop_last=False
)
# 加入测试数据，带噪声的数据
x_test_temp = torch.ones(500, 1)
x_test_normal = torch.normal(x_test_temp, 2)
x_test_noise = torch.rand_like(x_test_normal)
x_test = x_test_normal + x_test_noise
y_test = torch.zeros_like(x_test)
# 将测试数据封装为dataset和dataloader
test_dataset = Data.TensorDataset(x_test, y_test)
test_loader = Data.DataLoader(
    dataset=test_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0
)

# 生成原始VAE网络
VAE_1D_net = VAE_1D()
# 训练网络，其实trainer中并未使用testloader
vae_1D_trainer = VAE_1D_trainer(VAE_1D_net, train_loader, test_loader, epochs=30)
vae_1D_trainer.train()
vae_1D_trainer.get_normal_data()

# 生成改进VAE网络
VAE_Online_net = VAE_Online()
vae_Online_trainer = OLVAE_1D_trainer(VAE_Online_net, train_loader, test_loader, epochs=30)
vae_Online_trainer.train()
vae_Online_trainer.get_normal_data()

# 训练过程参数
org_train_loss = vae_1D_trainer.train_loss
org_train_mu = vae_1D_trainer.train_mu
org_train_var = vae_1D_trainer.train_var
org_train_logvar = vae_1D_trainer.train_logvar

ol_train_loss = vae_Online_trainer.train_loss
ol_train_mu = vae_Online_trainer.train_mu
ol_train_var = vae_Online_trainer.train_var
ol_train_logvar = vae_Online_trainer.train_logvar

# 将所有图放于一张图表中
plt.figure(figsize=(9.0, 9.0), dpi=100)
# 画 train_loss的图
ax11 = plt.subplot(221)
my_plot(org_train_loss, ol_train_loss, 'train loss')
# 画train_mu的图
ax12 = plt.subplot(222)
my_plot(org_train_mu, ol_train_mu, 'train mu')
# 画train_var的图
ax13 = plt.subplot(223)
my_plot(org_train_var, ol_train_var, 'train var')
# 画train_logvar的图
ax14 = plt.subplot(224)
my_plot(org_train_logvar, ol_train_logvar,'train logvar')

# 保存图
real_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
plt.savefig(Picture + real_time + "train")
plt.show()
plt.close()

# 测试数据参数
org_train_time = vae_1D_trainer.train_time
org_test_time = vae_1D_trainer.test_time
org_test_loss = vae_1D_trainer.test_loss
org_test_mu = vae_1D_trainer.test_mu
org_test_var = vae_1D_trainer.test_var
org_test_logvar = vae_1D_trainer.test_logvar

ol_train_time = vae_Online_trainer.train_time
ol_test_time = vae_Online_trainer.test_time
ol_test_loss = vae_Online_trainer.test_loss
ol_test_mu = vae_Online_trainer.test_mu
ol_test_var = vae_Online_trainer.test_var
ol_test_logvar = vae_Online_trainer.test_logvar

plt.figure(figsize=(25,16), dpi=150)
# 画test_loss的图
ax21 = plt.subplot(231)
my_plot(org_test_loss, ol_test_loss, "test loss")
# 画test mu的图
ax22 = plt.subplot(232)
my_plot(org_test_mu, ol_test_mu, "test mu")
# 画test var的图
ax23 = plt.subplot(233)
my_plot(org_test_var, ol_test_var, "test var")
# 画test logvar的图
ax24 = plt.subplot(234)
my_plot(org_test_logvar, ol_test_logvar, "test logvar")
# 画train_time
ax25 = plt.subplot(235)
train_time = (org_train_time, ol_train_time)
my_bar(train_time, "train time")
# 画test_time(一个值使用柱状图表示)
ax26 = plt.subplot(236)
test_time = (org_test_time, ol_test_time)
my_bar(test_time, "test time")

# 保存图
real_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
plt.savefig(Picture + real_time + "test")
plt.show()
plt.close()