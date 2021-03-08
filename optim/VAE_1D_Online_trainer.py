import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from other.log import init_log
from other.path import Train_Log_Path
from performance.performance import performance
import time
import numpy as np
import math
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

        self.logger = init_log(Train_Log_Path, "VAE_1D_ol")
        # 训练参数（多组参数）
        self.train_time = 0.0

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
            count_batch = 0
            epoch_start_time = time.time()
            for step, (data, label) in enumerate(self.trainloader):
                optimizer.zero_grad()
                # 执行的是forward函数 h1:5*1,h2:1*1
                h1, h2, recon_batch = self.net(data)
                # 损失函数必须和网络结构、数据集绑定在一起
                loss = loss_function(recon_batch, data, h2)
                loss.backward()
                optimizer.step()
                scheduler.step()

                # 一个epoch中所有batch的平均误差loss
                epoch_loss += loss.mean()
                count_batch += 1

            # 一个epoch中的所有batch的平均值
            epoch_loss /= count_batch
            # 统计每次epoch的训练时间
            epoch_train_time = time.time() - epoch_start_time
            self.logger.info("Epoch{}/{}\t training time:{}\t the average loss in each batch:{}\t "
                             .format(epoch+1, self.epochs, epoch_train_time, epoch_loss))
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
                h1, h2, recon = self.net(data) # h1:1*5,h2:1*1
                loss = loss_function(recon, data, h2)
                count_batch += 1

                # 公式计算均值和方差
                w1 = self.net.fc1.weight # 5*1
                w2 = self.net.fc2.weight # 1*5
                dh1_tmp = h1.mm(torch.transpose((1 - h1), 0, 1)) # (1*1)=(1*5)*(5*1)
                dh1 = w1.mm(dh1_tmp) # (5*1)=(5*1)*(1*1)
                dencoder = w2.mm(dh1) # (1*1)=(1*5)*(5*1)

                dde_tmp1 = w2.mm(w1) # (1*1)=(1*5)*(5*1)
                dde_tmp2 = dde_tmp1.mm(dh1.transpose(0, 1)) # (1*5)=(1*1)*(1*5)
                ddencoder = dde_tmp2.mm(torch.transpose((1 - 2*h1), 0, 1))# 1*1=(1*5)*(5*1)

                same = h2 / (dencoder.pow(2)-ddencoder * h2)
                mu = data + dencoder * same
                var = h2 * same  # sigma^2

                # 记录数据
                logvar = torch.log(var)
                self.test_loss.append(loss)
                self.test_mu.append(mu)
                self.test_var.append(var)
                self.test_logvar.append(logvar)

            self.test_time = time.time() - start_time
        self.logger.info("the average test time of each batch:{} for online VAE".format(self.test_time/count_batch))

def loss_function(recon_x, x, h2):
    # 累加重构误差
    loss = torch.nn.MSELoss()
    recon_x = loss(recon_x, x)
    # 计算中间变量与标准正态分布的误差
    normal = normal_pdf(x)
    recon_h2 = loss(h2, normal)
    return recon_x + recon_h2

def normal_pdf(x):
    result = 1/torch.sqrt(torch.tensor(2*math.pi)) * torch.exp(-x*x/2)
    return result