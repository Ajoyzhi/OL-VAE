import torch
import torch.utils.data as Data
from network.VAE_1D import VAE_1D
from network.VAE_1D_Online import VAE_Online
from optim.VAE_1D_trainer import VAE_1D_trainer
from optim.VAE_1D_Online_trainer import OLVAE_1D_trainer
from other.path import Picture,Log_Path
import matplotlib.pyplot as plt
import time
from other.log import init_log
import numpy as np

class test_VAE_1D():
    def __init__(self, mu, std, epoch, train_num, test_num, train_batch_size, logvar: bool):
        self.mu = mu # 生成数据的分布
        self.std = std
        self.epoch = epoch # 训练次数
        self.train_num = train_num # 训练数据个数
        self.test_num = test_num # 测试数据个数
        self.train_batch_size = train_batch_size # 训练数据的batch_size
        self.logvar = logvar

        # 生成的中间变量
        self.train_loader = None
        self.test_loader = None
        self.vae_1D_trainer = None
        self.vae_Online_trainer = None
        # 参数
        self.org_train_time = None
        self.org_test_time = None
        self.org_test_loss = None
        self.org_test_mu = None
        self.org_test_var = None
        self.org_test_logvar = None
        self.ol_train_time = None
        self.ol_test_time = None
        self.ol_test_loss = None
        self.ol_test_mu = None
        self.ol_test_var = None
        self.ol_test_logvar = None

        # 保存参数
        self.logger = init_log(Log_Path,"test_VAE_1D")


    # 生成训练数据和测试数据，并生成loader
    def get_dataloader(self):
        # 生成训练数据：高斯分布
        x_train_temp = torch.ones(self.train_num, 1)
        x_train_normal = torch.normal(self.mu * x_train_temp, self.std)
        x_train = x_train_normal
        y_train = torch.zeros_like(x_train)
        # 将数据封装为dataset和dataloader
        train_dataset = Data.TensorDataset(x_train, y_train)
        self.train_loader = Data.DataLoader(
            dataset=train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=0
        )
        # 加入测试数据，带噪声的数据
        x_test_temp = torch.ones(self.test_num, 1)
        x_test_normal = torch.normal(self.mu * x_test_temp, self.std)
        x_test_noise = torch.rand_like(x_test_normal)
        x_test = x_test_normal + x_test_noise
        y_test = torch.zeros_like(x_test)
        # 将测试数据封装为dataset和dataloader
        test_dataset = Data.TensorDataset(x_test, y_test)
        self.test_loader = Data.DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0
        )

    # 生成网络，训练网络并得到对比参数
    def get_param(self):
        # 生成原始VAE网络
        VAE_1D_net = VAE_1D()
        # 训练网络，其实trainer中并未使用testloader
        self.vae_1D_trainer = VAE_1D_trainer(VAE_1D_net, self.train_loader, self.test_loader, epochs=self.epoch)
        self.vae_1D_trainer.train()
        self.vae_1D_trainer.get_normal_data()

        # 生成改进VAE网络
        VAE_Online_net = VAE_Online()
        self.vae_Online_trainer = OLVAE_1D_trainer(VAE_Online_net, self.train_loader, self.test_loader, epochs=self.epoch)
        self.vae_Online_trainer.train()
        self.vae_Online_trainer.get_normal_data()

        # 测试数据参数
        self.org_train_time = self.vae_1D_trainer.train_time
        self.org_test_time = self.vae_1D_trainer.test_time
        self.org_test_loss = self.vae_1D_trainer.test_loss
        self.org_test_mu = self.vae_1D_trainer.test_mu
        self.org_test_var = self.vae_1D_trainer.test_var
        self.org_test_logvar = self.vae_1D_trainer.test_logvar

        self.ol_train_time = self.vae_Online_trainer.train_time
        self.ol_test_time = self.vae_Online_trainer.test_time
        self.ol_test_loss = self.vae_Online_trainer.test_loss
        self.ol_test_mu = self.vae_Online_trainer.test_mu
        self.ol_test_var = self.vae_Online_trainer.test_var
        self.ol_test_logvar = self.vae_Online_trainer.test_logvar

    # 画对比参数的图
    def plot_fig(self):
        plt.figure(figsize=(25, 16), dpi=100)
        # 画test_loss的图
        ax21 = plt.subplot(231)
        my_plot(self.org_test_loss, self.ol_test_loss, "test loss")
        # 画test mu的图
        ax22 = plt.subplot(232)
        my_plot(self.org_test_mu, self.ol_test_mu, "test mu")
        # 画test var的图
        ax23 = plt.subplot(233)
        my_plot(self.org_test_var, self.ol_test_var, "test var")
        # 画train_time
        ax25 = plt.subplot(235)
        train_time = (self.org_train_time, self.ol_train_time)
        my_bar(train_time, "train time")
        # 画test_time(一个值使用柱状图表示)
        ax26 = plt.subplot(236)
        test_time = (self.org_test_time, self.ol_test_time)
        my_bar(test_time, "test time")

        if self.logvar:
            # 画test logvar的图
            ax24 = plt.subplot(234)
            my_plot(self.org_test_logvar, self.ol_test_logvar, "test logvar")

        # 保存图
        real_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        plt.savefig(Picture + real_time + "test_VAE_1D")
        plt.show()
        plt.close()

    # 保存数据
    def save_data(self):
        self.logger.info("origin train time:", self.org_train_time)
        self.logger.info("online train time:", self.ol_train_time)
        self.logger.info("origin test time:", self.org_test_time)
        self.logger.info("online test time:", self.ol_test_time)
        self.logger.info("origin test mu:", self.org_test_mu)
        self.logger.info("online test mu:", self.ol_test_mu)
        self.logger.info("origin test var:", self.org_test_var)
        self.logger.info("online test var:", self.ol_test_var)
        self.logger.info("origin test logvar:", self.org_test_logvar)
        self.logger.info("online test logvar:", self.ol_test_logvar)


def my_plot(org_data, online_data, name:str):
    plt.scatter(range(len(org_data)), org_data, marker='x', color='r', s=40, label='origin')
    plt.scatter(range(len(online_data)), online_data, marker='o', color='w', s=40, label='online', edgecolors='b')
    plt.xlabel('batch')
    plt.ylabel(name)
    plt.title(name + ' of original VAE vs. online VAE')
    plt.legend()

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

"""
# 计算测试数据的平均值
mean_org_test_loss = np.array(vae_1D_trainer.test_loss).mean()
mean_org_test_mu = np.array(vae_1D_trainer.test_mu).mean()
mean_org_test_var = np.array(vae_1D_trainer.test_var).mean()

mean_ol_test_loss = np.array(vae_Online_trainer.test_loss).mean()
mean_ol_test_mu = np.array(vae_Online_trainer.test_mu).mean()
mean_ol_test_var = np.array(vae_Online_trainer.test_var).mean()

plt.figure(figsize=(8,8), dpi=100)# 画测试平均数据的图 画于同一个图中
origin = [mean_org_test_loss, mean_org_test_mu, mean_org_test_var]
online = [mean_ol_test_loss, mean_ol_test_mu, mean_ol_test_var]
x = np.arange(len(origin))
x_label = ['test loss', 'test mu', 'test var']
width = 0.2
plt.bar(x, origin, label='origin', color='w', width=width, hatch='x', edgecolor='k')
plt.bar(x+width, online, label='online', color='w', width=width, hatch='+', edgecolor='k')
plt.xticks(x, x_label)
plt.title("the mean of test loss, test mu and test var")
plt.legend()

real_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
plt.savefig(Picture + real_time + "mean_test")
plt.show()
plt.close()
"""