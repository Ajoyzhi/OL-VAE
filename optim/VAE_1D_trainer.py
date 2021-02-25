import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from other.log import init_log
from performance.performance import performance
import time
import numpy as np
import matplotlib.pyplot as plt

"""
    对改进的VAE算法和原始VAE算法进行训练
    输入：1-D数据
    输出（主要对比参数）：均值和方差是否一致；损失函数；训练时间；测试时间
"""


# 数据集和损失函数是相关联的，所以必须对不同的损失函数（数据集）建立不同的trainer
class VAE_1D_trainer():
    def __init__(self, net, trainloader: DataLoader, testloader: DataLoader, epochs: int = 10, lr: float = 0.001,
                 lr_milestones: tuple = (), weight_decay: float = 1e-6):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader

        self.epochs = epochs
        self.lr = lr
        self.milestones = lr_milestones
        # L2正则化的系数
        self.weight_decay = weight_decay

        # 训练时，平均参数
        self.train_time = []
        self.train_loss = []
        self.train_mu = []
        self.train_var = []
        # 测试时，每个数据的参数
        self.test_batch_time = []

    def train(self):
        print("starting training VAE with 1-D data...")
        # 设置优化算法
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # 设置学习率的下降区间和速度 gamma为学习率的下降速率
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=0.1)
        # 训练
        start_time = time.time()
        self.net.train()
        for epoch in range(self.epochs):
            # 训练过程
            epoch_loss = 0.0
            epoch_mu = 0.0
            epoch_var = 0.0
            count_batch = 0
            epoch_start_time = time.time()
            for item in self.trainloader:
                data, _, _ = item
                optimizer.zero_grad()
                # 执行的是forward函数 mu:1-D;var:1-D(方差)
                recon_batch, mu, var = self.net(data)
                # 损失函数必须和网络结构、数据集绑定在一起
                loss = loss_function(recon_batch, data, mu, var)
                loss.backward()
                optimizer.step()
                scheduler.step()
                # 一次迭代中所有数据的平均误差loss，平均均值mu，平均方差var
                epoch_loss += loss.mean()
                epoch_mu += mu.mean()
                epoch_var += var.mean()
                count_batch += 1

            # 一个epoch中的所有batch的平均值
            epoch_loss /= count_batch
            epoch_mu /= count_batch
            epoch_var /= count_batch
            print("Epoch{}/{}\t the average loss in each batch:{}\t the average mu in each batch:{}\t"
                  "the average var in each batch:{}\t"
                  .format(epoch, self.epochs, epoch_loss, epoch_mu, epoch_var))

            # 统计参数，以便画图
            self.train_loss.append(epoch_loss)
            self.train_mu.append(epoch_mu)
            self.train_var.append(epoch_var)

            # 显示学习率的变化
            if epoch in self.milestones:
                print("LR scheduler: new learning rate is %g" % float(scheduler.get_lr()[0]))

            # 统计每次epoch的训练时间
            epoch_train_time = time.time() - epoch_start_time
            self.train_time.append(epoch_train_time)
        print("finishing training VAE with 1-D data.")

    """
        获取正常数据的测试时间
    """
    def test(self):
        print("starting getting test time...")
        self.net.eval()
        start_time = time.time()
        with torch.no_grad:
            count_batch = 0
            for item in self.trainloader:
                batch_start_time = time.time()
                data, _, _ = item
                recon, mu, var = self.net(data)
                loss = loss_function(recon, data, mu, var)
                count_batch += 1
                each_batch_time = time.time() - batch_start_time
                self.test_batch_time.append(each_batch_time)
            using_time = start_time - time.time()
        print("the average test time of each batch:{}".format(using_time / count_batch))


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, var):
    # 累加重构误差
    loss_rec = torch.nn.MSELoss()
    BCE = loss_rec(recon_x, x)
    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 15), reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + 2 * torch.log(var) - mu.pow(2) - var)
    return BCE + KLD