import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from other.log import init_log
import other.path as path
from performance.performance import performance
import time

# 数据集和损失函数是相关联的，所以必须对不同的损失函数（数据集）建立不同的trainer
class VAE_Kdd99_trainer():
    def __init__(self, net, trainloader: DataLoader, testloader: DataLoader, epochs: int = 150, lr: float = 0.001,
                 lr_milestones: tuple = (), weight_decay: float = 1e-6, thr: float = 0.01):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader

        self.epochs = epochs
        self.lr = lr
        self.milestones = lr_milestones
        # L2正则化的系数
        self.weight_decay = weight_decay
        # 训练时，平均参数
        self.train_time = 0.0
        self.train_loss = 0.0
        # 15维向量的均值
        self.train_mu = 0.0
        self.train_std = 0.0
        # 测试时，每个数据的参数
        self.test_time = 0.0
        self.test_loss = 0.0
        self.test_mu = 0.0
        self.test_std = 0.0

        self.thr = thr

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
            epoch_std = 0.0
            count_batch = 0
            epoch_start_time = time.time()
            for item in self.trainloader:
                data, _, _ = item
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
                epoch_mu += mu.mean().item()
                epoch_std += torch.sqrt(logvar.exp()).mean()
                count_batch += 1

            # 显示学习率的变化
            if epoch in self.milestones:
                logger.info("LR scheduler: new learning rate is %g" % float(scheduler.get_lr()[0]))

            epoch_train_time = time.time() - epoch_start_time
            # 每个batch(多个单个样本)的平均误差；epoch_loss就是一次迭代所有数据的总误差
            logger.info("\n Epoch{}/{}\t Time:{:.3f}\t Loss of each batch:{:.8f}\t mu of each batch:{:.5f}\t std of each batch:{:.5f}".
                        format(epoch+1, self.epochs, epoch_train_time, epoch_loss / count_batch, epoch_mu / count_batch, epoch_std / count_batch))

            self.train_loss += epoch_loss
            self.train_mu += epoch_mu
            self.train_std += epoch_std

        self.train_time = time.time() - start_time
        self.train_loss /= (self.epochs * len(self.trainloader.dataset))
        self.train_mu /= (self.epochs * len(self.trainloader.dataset))
        self.train_std /= (self.epochs * len(self.trainloader.dataset))

        # all_loss就相当于所有迭代所有数据的总误差
        logger.info("Training time:{:.3f}\t Training loss:{:.8f}\t Normal mu:{:.5f}\t Normal std:{:.5f}".
                    format(self.train_time, self.train_loss, self.train_mu, self.train_std))
        logger.info("Finishing training VAE with Kdd99...")

        return self.net


    def test(self):
        prediction = []
        index_list= []
        label_list = []

        logger = init_log(path.Log_Path)
        logger.info("Starting testing VAE with kdd99...")
        start_time = time.time()
        upbound = self.train_mu + self.thr * self.train_std
        lowbound = self.train_mu - self.thr * self.train_std
        self.net.eval()
        with torch.no_grad():
            for item in self.testloader:
                data, label, index = item
                # 只是一个batch的损失，mu，logvar
                # 如果batch为1，则以下变量对应一个数据的loss、mu、logvar
                recon_batch, mu, logvar = self.net(data)
                test_loss = loss_function(recon_batch, data, mu, logvar)
                self.test_mu = mu.mean()
                self.test_std = torch.sqrt(logvar.exp()).mean()
                self.test_loss = test_loss.mean()

                index_list.append(index)
                label_list.append(label)
                # 只考虑均值和方差（准确度不是很高，但是不影响）
                if (self.test_mu >= lowbound) and (self.test_mu <= upbound):
                    prediction.append(0)
                else:
                    prediction.append(1)
                # 打印label和预测结果
                # logger.info("index:{:.0f}\t label:{}\t prediction:{}\t mu:{:.5f}\t std:{:.5f}\t loss;{:.5f}".
                #            format(int(index), label, pred_label, self.test_mu, self.test_std, self.test_loss))
                # 将index，label，predict_label封装在一个list中
            index_label_prediction = list(zip(index_list, label_list, prediction))
            logger.info(index_label_prediction)

        # 输出性能
        per_obj = performance(index_label_prediction)
        per_obj.get_base_metrics()
        per_obj.AUC_ROC()

        self.test_time = time.time() - start_time
        logger.info("Test time:{:.3f}\t accurancy:{}\t precision:{}\t recall:{}\t f1score:{}\t AUC:{}\t".
                    format(self.test_time, per_obj.accurancy, per_obj.precision, per_obj.recall, per_obj.f1score, per_obj.AUC))
        logger.info("Finishing testing VAE with Kdd99...")

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    # 累加重构误差
    loss_rec = torch.nn.MSELoss()
    BCE = loss_rec(recon_x, x)
    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 15), reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD