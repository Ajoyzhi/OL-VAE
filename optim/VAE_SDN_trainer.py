import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from scipy.stats import multivariate_normal
import numpy as np

# 数据集和损失函数是相关联的，所以必须对不同的损失函数（数据集）建立不同的trainer
class VAE_SDN_trainer():
    def __init__(self, net, dataloader: DataLoader, logger):
        self.net = net
        self.dataloader = dataloader
        self.logger = logger

        # 保存数据
        self.train_loss = 0.0
        # 异常判断的门限值
        self.threshold_mean = 0.0
        self.train_mu = []
        self.train_var = []
        self.train_loss = []

        # 测试时，每个数据的参数
        self.test_time = 0.0
        self.train_time = 0.0
        self.get_param_time = 0.0
        self.id_prob_loss_prediction = []

    """
        param:
            input: net, dataloader, lr, weight_decay, epochs, dataloader
            return: train_time,train_loss
    """
    def train(self, epochs: int = 10, lr: float = 0.001, weight_decay: float = 1e-6):
        print("Start training VAE with SDN data...")
        # 设置优化算法
        optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        self.net.train()
        start_time = time.time()
        # 整体的训练误差
        train_loss = 0.0
        for epoch in range(epochs):
            epoch_loss = 0.0
            count_batch = 0
            epoch_start_time = time.time()
            for item in self.dataloader:
                data, _, _ = item
                optimizer.zero_grad()
                recon, mu, logvar = self.net(data)
                # batch_size=1, batch_loss；float
                loss, _ = loss_function(recon, data, mu, logvar)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                count_batch += 1

            # 输出每个epoch的训练时间和每个batch的训练损失
            train_loss += epoch_loss
            epoch_train_time = time.time() - epoch_start_time
            self.logger.info("Epoch:{}/{}\ttrain time of each batch:{:.5f}\tloss of each batch:{:.3f}\t"
                  .format(epoch+1, epochs, epoch_train_time/count_batch, epoch_loss/count_batch))
        # 计算所有数据的训练时间和每个epoch的训练损失
        self.train_time = time.time() - start_time
        train_loss /= epochs
        self.logger.info("train time:{:.5f}\ttrain loss:{:.3f}".format(self.train_time, train_loss))

    """
        param:
            input: net, dataloader
            return: normal_mu, normal_var, threshold, using_time 
    """
    def get_normal_parm(self):
        mu_list = []
        var_list = []
        loss_list = []

        mu_var = []
        start_time = time.time()
        self.net.eval()
        count_batch = 0
        with torch.no_grad():
            for item in self.dataloader:
                data, _, _ = item
                recon, mu, logvar = self.net(data)
                loss, var = loss_function(recon, data, mu, logvar)
                count_batch += 1
                # batch=1,所以mu[[]]只有一个元素
                mu_list.append(mu[0])
                var_list.append(var[0])
                loss_list.append(loss)

                # record the mu and var of each data to avoid computing again
                mu_var_list = list(zip(mu[0], var[0]))
                mu_var.append(mu_var_list)

        # get normal mean and var with median(按照tensor中位数计算中位数位置)
        # print("mu_list:", mu_list)
        # print("var_list:", var_list)
        self.train_mu = list_median(mu_list)
        self.train_var = list_median(var_list)
        self.train_loss = list_avrg(loss_list)
        # print("train_mu:",self.train_mu)
        # print("train var:", self.train_var)

        # get threshold
        normaldata_prob = []
        for mu_var_item in mu_var:
            # 每个数据隐变量采样M个数据的概率mean
            mu_each_data, var_each_data = zip(*mu_var_item)
            eachdata_prob = prob(mu_each_data, self.train_mu, self.train_var)
            normaldata_prob.append(eachdata_prob)

        self.threshold_mean = list_avrg(normaldata_prob)
        self.get_param_time = time.time() - start_time
        self.logger.info("normal_mu:{}\tnormal_var:{}\tthreshold:{}\ttime of getting param:{:.5f}".
                         format(self.train_mu, self.train_var, self.threshold_mean, self.get_param_time))

    """
        param:
            input: net, dataloader, normal_mu, normal_var, threshold
            return: result
    """
    def test(self, normal_mu, normal_var, threshold):
        id_list = []
        prob_list = []
        prediction_list = []
        loss_list = []

        start_time = time.time()
        self.net.eval()
        with torch.no_grad():
            for item in self.dataloader:
                data, id, index = item
                # data = data.float()
                # 如果batch为1，则以下变量对应一个数据的loss、mu、logvar
                recon, mu, logvar = self.net(data)
                loss = loss_function(recon, data, mu, logvar)
                id_list.append(id)
                loss_list.append(loss[0])
                # std = torch.exp(0.5 * logvar)
                # data_prob = prob_avrg(self.M, mu, std, self.train_mu, self.train_var)
                data_prob = prob(mu, normal_mu, normal_var)
                prob_list.append(data_prob)
                if data_prob < threshold:
                    # 异常 1
                    prediction_list.append(1)
                    print("id:{}\t probability:{}\t loss:{}\t label:abnormal".format(id, data_prob, loss[0]))
                else:
                    # 正常 0
                    prediction_list.append(0)
                    print("id:{}\t probability:{}\t loss:{}\t label:normal".format(id, data_prob, loss[0]))
        self.id_prob_loss_prediction = list(zip(id_list, prob_list, loss_list, prediction_list))
        self.test_time = time.time() - start_time

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    # 每个batch和每个维度的平均，得到标量
    loss_rec = torch.nn.MSELoss()
    MSE = loss_rec(recon_x, x)
    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 15), reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    var = logvar.exp()
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD, var

# compute the mean of data in list composing with tensor
def list_avrg(list):
    sum = 0
    for item in list:
        sum += item
    return sum/len(list)

# compute the median of data in list
def get_list_median(list):
    sort_list = sorted(list)
    half = len(sort_list) // 2
    return (np.array(sort_list[half]) + np.array(sort_list[~half])) / 2

# compute the median of data in list composing with tensor
def list_median(list):
    sum_tensor = []
    for item in list:
        tmp = get_list_median(item.tolist())
        sum_tensor.append((tmp, item))

    sort_tensor1 = sorted(sum_tensor, key=lambda x:x[0])
    half = len(sort_tensor1) // 2
    return (np.array(sort_tensor1[half][1]) + np.array(sort_tensor1[~half][1])) / 2

# compute the min of data in list composing with tensor
def list_min(list):
    sum_tensor = []
    for item in list:
        tmp = torch.min(abs(item))
        # tmp = torch.sum(item)
        sum_tensor.append((tmp, item))

    sort_tensor1 = sorted(sum_tensor, key=lambda x: x[0])
    # print("sort_tensor1:", sort_tensor1)
    return sort_tensor1[0][1]

# compute the prob
""" 
    input:  simple_mu: the mu of simple distribution(9-dim vector)
            nor_mu: the mu of normal distribution(9-dim vector)
            nor_var:the standard of normal distribution(9-dim vector)
    return: the mean probability of M samples  
"""
def prob(mu, nor_mu, nor_var):
    prob = multivariate_normal.pdf(mu, nor_mu, np.diag(nor_var))
    return prob