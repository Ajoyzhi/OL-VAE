import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from other.log import init_log
from other.path import Log_Path
from performance.performance import performance
import time
import numpy as np
import matplotlib.pyplot as plt

"""
    对改进的VAE算法和原始VAE算法进行训练
    输入：1-D数据
    输出（主要对比参数）：均值和方差是否一致；损失函数；训练时间；测试时间
    其它：并未使用testloader
"""
# 数据集和损失函数是相关联的，所以必须对不同的损失函数（数据集）建立不同的trainer
class OLVAE_1D_trainer():
    def __init__(self, net, trainloader: DataLoader, testloader: DataLoader=None, epochs: int = 10, lr: float = 0.001,
                 lr_milestones: tuple = (), weight_decay: float = 1e-6):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader

        self.epochs = epochs
        self.lr = lr
        self.milestones = lr_milestones
        # L2正则化的系数
        self.weight_decay = weight_decay

        self.logger = init_log(Log_Path, "VAE_ID_ol")
        # 训练参数（多组参数）
        self.train_time = 0.0
        self.train_loss = []
        self.train_mu = []
        self.train_var = []
        self.train_logvar = []

        # 测试时所有数据的总测试时间
        self.test_time = 0.0
        self.test_loss = []
        self.test_mu = []
        self.test_var = []
        self.test_logvar = []

    def train(self):
        self.logger.info("starting training online VAE with 1-D data...")
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
            epoch_logvar = 0.0
            count_batch = 0
            epoch_start_time = time.time()
            for step, (data, label) in enumerate(self.trainloader):
                optimizer.zero_grad()
                # 执行的是forward函数 mu:1-D;var:1-D(方差)
                recon_batch, mu, var = self.net(data)
                # 损失函数必须和网络结构、数据集绑定在一起
                loss = loss_function(recon_batch, data, mu, var)
                loss.backward()
                optimizer.step()
                scheduler.step()

                # 记录每个batch中的平均loss、mu、var 共有epoch*count_batch这么多数据
                logvar = torch.log(var)
                self.train_loss.append(loss.mean())
                self.train_mu.append(mu.mean())
                self.train_var.append(var.mean())
                self.train_logvar.append(logvar.mean())
                # 一个epoch中所有batch的平均误差loss，平均均值mu，平均方差var之和
                epoch_loss += loss.mean()
                epoch_mu += mu.mean()
                epoch_var += var.mean()
                epoch_logvar += logvar.mean()
                count_batch += 1

            # 一个epoch中的所有batch的平均值
            epoch_loss /= count_batch
            epoch_mu /= count_batch
            epoch_var /= count_batch
            epoch_logvar /= count_batch
            # 统计每次epoch的训练时间
            epoch_train_time = time.time() - epoch_start_time
            self.logger.info("Epoch{}/{}\t training time：{}\t the average loss in each batch:{}\t the average mu:{}\t"
                             "the average var:{}\t the average logvar:{}"
                             .format(epoch+1, self.epochs, epoch_train_time, epoch_loss, epoch_mu, epoch_var, epoch_logvar))
            # 显示学习率的变化
            if epoch in self.milestones:
                 print("LR scheduler: new learning rate is %g" % float(scheduler.get_lr()[0]))

        self.train_time = time.time() - start_time
        self.logger.info("finishing training online VAE with 1-D data.")

    """
        获取正常数据的测试时间
        对比两个算法，只要得到最终的均值和方差可以达到相同水平即可，所以获取正常数据的均值、方差、测试时间
    """
    def get_normal_data(self):
        self.logger.info("starting getting test time for online VAE...")
        self.net.eval()
        start_time = time.time()
        with torch.no_grad():
            count_batch = 0
            for step, (data, label) in enumerate(self.testloader):
                # batch=1，即参数为每个数据的参数
                recon, mu, var = self.net(data)
                loss = loss_function(recon, data, mu, var)
                count_batch += 1

                # 记录数据
                logvar = torch.log(var)
                self.test_loss.append(loss)
                self.test_mu.append(mu)
                self.test_var.append(var)
                self.test_logvar.append(logvar)

            self.test_time = time.time() - start_time
        self.logger.info("the average test time of each batch:{} for online VAE".format(self.test_time/count_batch))

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
    KLD = -0.5 * torch.sum(1 + torch.log(var) - mu.pow(2) - var)
    return BCE + KLD