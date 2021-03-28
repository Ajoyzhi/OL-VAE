import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from other.log import init_log
from other.path import Train_Log_Path, Test_Log_Path
import time
import csv
from scipy.stats import multivariate_normal
import numpy as np
from other.path import Model

"""
    save train data(log)
        1. average training time of each batch
        2. average training loss of each batch
        3. Trainer time
        4. average trainer time of each epoch
        5. normal mean, var, threshold, time, loss
    into /other/log/train/VAE_KDD99.log
    save test data(csv)
        index label prediction
    into /other/log/test/VAE_KDD99.csv
"""
# 数据集和损失函数是相关联的，所以必须对不同的损失函数（数据集）建立不同的trainer
class VAE_Kdd99_trainer():
    def __init__(self, net, trainloader: DataLoader, testloader: DataLoader=None, epochs: int = 10, lr: float = 0.001,
                  weight_decay: float = 1e-6, sample_num:int=10):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.epochs = epochs
        self.lr = lr
        # L2正则化的系数
        self.weight_decay = weight_decay
        # 采样次数
        self.M = sample_num

        # 保存数据
        self.train_logger = init_log(Train_Log_Path, "VAE_KDD99")
        self.train_loss = 0.0
        # 异常判断的门限值
        self.threshold = 5e-9
        self.train_mu = []
        self.train_var = []
        self.train_loss = []

        # 测试时，每个数据的参数
        self.test_time = 0.0
        self.train_time = 0.0
        self.get_param_time = 0.0
        self.index_label_prediction = []

    def train(self):
        self.train_logger.info("Start training VAE with Kdd99...")
        # 设置优化算法
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # 训练
        start_time = time.time()
        self.net.train()
        # 整体的训练误差
        train_loss = 0.0
        for epoch in range(self.epochs):
            # 每次迭代的训练误差
            epoch_loss = 0.0
            # 每个epoch的batch数量
            count_batch = 0
            epoch_start_time = time.time()
            for item in self.trainloader:
                data, _, _ = item
                optimizer.zero_grad()
                data = data.float()
                # print("data type: float32", data.dtype)
                # 执行的是forward函数 mu一个15维的向量；logvar为15维的向量
                recon_batch, mu, logvar = self.net(data)
                batch_loss, _ = loss_function(recon_batch, data, mu, logvar)
                batch_loss.backward()
                optimizer.step()
                # 一次迭代中所有数据的误差loss
                epoch_loss += batch_loss.item()
                count_batch += 1

            # 输出每个epoch的训练时间和每个batch的训练损失
            train_loss += epoch_loss
            epoch_train_time = time.time() - epoch_start_time
            # epoch_loss就是一次迭代所有数据的总误差
            self.train_logger.info("Epoch{}/{} triaining tme of each batch:{:.3f}\t average loss of each batch:{:.8f}".
                        format(epoch+1, self.epochs, epoch_train_time, epoch_loss/count_batch))
        # 计算所有数据的训练时间和每个epoch的训练损失
        self.train_time = time.time() - start_time
        train_loss /= self.epochs
        # train_loss就相当于所有迭代所有数据的总误差
        self.train_logger.info("training time:{:.3f}\t average training loss of each epoch:{:.8f}".format(self.train_time, train_loss))
        self.train_logger.info("Finish training VAE with Kdd99.")

    def get_normal_parm(self):
        mu_list = []
        var_list = []
        loss_list = []

        self.train_logger.info("Starting getting the mean and variance of normal data...")
        start_time = time.time()
        self.net.eval()
        count_batch = 0
        with torch.no_grad():
            for item in self.trainloader:
                data, _, _ = item
                data = data.float()
                recon, mu, logvar = self.net(data)
                loss, var = loss_function(recon, data, mu, logvar)
                count_batch += 1
                # 求每个batch（batch=10个数据）的平均值向量
                batch_mu_list = torch.mean(mu, dim=0)
                batch_var_list = torch.mean(var, dim=0)
                batch_loss_list = torch.mean(loss, dim=0)

                mu_list.append(batch_mu_list)
                var_list.append(batch_var_list)
                loss_list.append(batch_loss_list)

                # 获取正常数据分布的阈值
                mu_numpy = mu.numpy()
                var_numpy = var.numpy()
                std = torch.exp(0.5 * logvar)
                std_numpy = std.numpy()
                normaldata_prob = 0.0
                for i in range(len(mu_numpy)):
                    mu = torch.from_numpy(mu_numpy[i])
                    std = torch.from_numpy(std_numpy[i])
                    var = torch.from_numpy(var_numpy[i])
                    # 每个数据隐变量采样M个数据的概率mean
                    eachdata_prob = prob_avrg(self.M, mu, std, mu, var)
                    normaldata_prob += eachdata_prob
                normaldata_prob /= len(mu_numpy)

            self.threshold = normaldata_prob
            self.train_mu = list_avrg(mu_list)
            self.train_var = list_avrg(var_list)
            self.train_loss = list_avrg(loss_list)

        self.get_param_time = time.time() - start_time

        self.train_logger.info("the threshold is {}\n "
                         "the mean of normal distribution is {}\n"
                         "the variance of normal distribution is {}\n"
                         "the loss of training data is {:.8f}\n"
                         "the using time of getting param is {:.3f}\n"
                         .format(self.threshold, self.train_mu, self.train_var, self.train_loss, self.get_param_time))
        self.train_logger.info("Finish getting parameters.")

    """
        测试样本是否正常
        输入：M：采样的次数; self.threshold: 判断异常与否的阈值[0,1]
        输出：异常分数score [0,1]；标签flag {0,1}
        其它：test()函数中的batch一定是1.
    """
    def test(self):
        index_list = []
        prediction_list = []
        label_list = []

        start_time = time.time()
        self.net.eval()
        with torch.no_grad():
            for item in self.testloader:
                data, label, index = item
                data = data.float()
                # 如果batch为1，则以下变量对应一个数据的loss、mu、logvar
                _, mu, logvar = self.net(data)
                std = torch.exp(0.5 * logvar)
                avrg_prob = prob_avrg(self.M, mu, std, self.train_mu, self.train_var)
                # 统计结果
                index_list.append(index)
                label_list.append(label)
                if avrg_prob < self.threshold:
                    prediction_list.append(1)# 异常
                else:
                    prediction_list.append(0)# 正常
            self.index_label_prediction = list(zip(index_list, label_list, prediction_list))
            self.test_time = time.time() - start_time
        # save test result into csv
        filepath = Test_Log_Path + "VAE_KDD99.csv"
        file = open(file=filepath, mode='w', newline='')
        writer = csv.writer(file, dialect='excel')
        header = ['index', 'label', 'prediction']
        writer.writerow(header)
        for item in self.index_label_prediction:
            writer.writerow(item)
        file.close()
    """
        调用该方法时需要创建新的trainloader；net为之前的self.net结构
    """
    """
    def update_model(self):
        # 获取原网络结构的参数。字典类型，key value
        dict_org = self.net.state_dict()
        # 使用新的trainloder训练网络
        self.train_logger.info("strating updating VAE with new kdd99...")
        self.train()
        self.train_logger.info("finishing updating VAE model.")
        # 获取更新网络的参数
        dict_update = self.net.state_dict()
        # 利用原始参数和更新的参数计算网络权重
        dict_new = {}
        for key, org_value in dict_org:
            # 参数更新公式
            temp = self.alpha * org_value + (1 - self.alpha) * dict_update[key]
            # 将更新的参数放入新的字典
            dict_new[key] = temp
        torch.save(dict_new, Model + 'net_parm.pth')
        # 将新参数加载到self.net中
        self.net.load_stat_dict(torch.load(Model + 'net_parm.pth'))
    """
# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    # 累加重构误差
    loss_rec = torch.nn.MSELoss()
    BCE = loss_rec(recon_x, x)
    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 15), reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    var = logvar.exp()
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, var

# compute the mean of data in list
def list_avrg(list):
    sum = 0
    for item in list:
        sum += item
    return sum/len(list)

# compute the mean prob of M simples
"""
    input:  M: the number of sample(int); 
            simple_mu: the mu of simple distribution(15-dim vector)
            simple_std: the standard of simple distribution(15-dim vector)
            prob_mu: the mu of normal distribution(15-dim vector)
            prob_var:the standard of normal distribution(15-dim vector)
    return: the mean probability of M samples  
"""
def prob_avrg(M: int, simple_mu, simple_std, prob_mu, prob_var):
    prob = 0.0
    for i in range(M):
        # get M simples
        eps = torch.randn_like(simple_std)
        z = simple_mu + eps * simple_std
        # get prob
        each_prob = multivariate_normal.pdf(z, prob_mu, np.diag(prob_var))
        # compute the sum prob of M simples
        prob += each_prob
    prob = prob / M
    return prob