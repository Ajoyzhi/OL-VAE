import torch
import torch.optim as optim
from torch.nn import functional as F

from torch.utils.data import DataLoader
from other.log import init_log
import other.path as path
import time

class VAE_Kdd99_trainer():
    def __init__(self, net, trainloader: DataLoader, testloader: DataLoader, epochs: int = 150, lr: float = 0.001,
                 lr_milestones: tuple = (), weight_decay: float = 1e-6):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader

        self.epochs = epochs
        self.lr = lr
        self.milestones = lr_milestones
        # L2正则化的系数
        self.weight_decay = weight_decay

        self.train_time = 0.0
        self.train_loss = 0.0
        self.train_mu = 0.0
        self.train_logvar = 0.0
        self.test_time = 0.0
        self.test_loss = 0.0
        self.test_mu = 0.0
        self.test_logvar = 0.0

    def train(self):
        logger = init_log(path.Log_Path)
        # 设置优化算法
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # 设置学习率的下降区间和速度 gamma为学习率的下降速率
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=0.1)
        # 训练
        logger.info("Starting training VAE with Kdd99...")
        start_time = time.time()
        self.net.train()
        for epoch in range(self.epochs):
            # 训练过程
            epoch_loss = 0.0
            epoch_mu = 0.0
            epoch_logvar = 0.0
            count_batch = 0
            epoch_start_time = time.time()
            for item in self.trainloader:
                data, _ = item
                optimizer.zero_grad()
                # 执行的是forward函数 mu一个15维的向量；logvar为15维的向量
                recon_batch, mu, logvar = self.net(data)
                # 损失函数必须和网络结构、数据集绑定在一起
                loss = loss_function(recon_batch, data, mu, logvar)
                loss.backward()
                optimizer.step()
                scheduler.step()

                # 一次迭代中所有数据的误差loss，所有数据的均值mu，所有数据的方差logvar
                epoch_loss += loss.item()
                epoch_mu += mu.sum().item()
                epoch_logvar += logvar.sum().item()
                count_batch += 1

            # 显示学习率的变化
            if epoch in self.milestones:
                logger.info("LR scheduler: new learning rate is %g" % float(scheduler.get_lr()[0]))

            epoch_train_time = time.time() - epoch_start_time
            # 每个batch(多个单个样本)的平均误差；epoch_loss就是一次迭代所有数据的总误差
            logger.info("\n Epoch{}/{}\t Time:{:.3f}\t Loss of each batch:{:.8f}\t mu of each batch:{:.5f}\t logvar of each batch:{:.5f}".
                        format(epoch+1, self.epochs, epoch_train_time, epoch_loss / count_batch, epoch_mu / count_batch, epoch_logvar / count_batch))

            self.train_loss += epoch_loss
            self.train_mu += epoch_mu
            self.train_logvar += epoch_logvar

        self.train_time = time.time() - start_time
        self.train_loss /= (self.epochs * len(self.trainloader.dataset))
        self.train_mu /= (self.epochs * len(self.trainloader.dataset))
        self.train_logvar /= (self.epochs * len(self.trainloader.dataset))
        # all_loss就相当于所有迭代所有数据的总误差
        logger.info("Training time:{:.3f}\t Training loss:{:.8f}\t Normal mu:{:.5f}\t Normal logvar:{:.5f}".
                    format(self.train_time, self.train_loss, self.train_mu, self.train_logvar))
        logger.info("Finishing training VAE with Kdd99...")

        return self.net


    def test(self):
        logger = init_log(path.Log_Path)
        logger.info("Starting testing VAE with kdd99...")
        start_time = time.time()
        self.net.eval()
        with torch.no_grad():
            for item in self.testloader:
                data, _ = item
                # 只是一个batch的损失，mu，logvar
                recon_batch, mu, logvar = self.net(data)
                test_loss = loss_function(recon_batch, data, mu, logvar)
                # 累计所有batch的loss，mu, logvar
                self.test_loss += test_loss.item()
                self.test_mu += mu.sum().item()
                self.test_logvar += logvar.sum().item()

        self.test_time = time.time() - start_time
        self.test_loss /= len(self.testloader)
        self.test_mu /= len(self.testloader)
        self.train_logvar /= len(self.testloader)

        logger.info("Test time:{:.3f}\t Test loss:{:.8f}\t Test mu:{:.5f}\t Test logvar:{:.5f}".
                    format(self.test_time, self.test_loss, self.test_mu, self.test_logvar))
        logger.info("Finishing testing VAE with Kdd99...")


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    # 累加重构误差
    loss_rec = torch.nn.MSELoss(reduction='sum')
    BCE = loss_rec(recon_x, x)
    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 15), reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD