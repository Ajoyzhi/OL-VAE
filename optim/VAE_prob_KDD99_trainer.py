import torch
import torch.optim as optim 
from scipy.stats import multivariate_normal
import time
import numpy as np
from torch.utils.data import DataLoader
from other.path import Train_Log_Path, Test_Log_Path
from other.log import init_log
from performance.performance import performance

class VAE_prob_KDD99_trainer():
    def __init__(self, net, trainloader:DataLoader, testloader:DataLoader, epoch, lr, weight_decay, simple_num, alpha):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.epoch = epoch
        self.lr = lr
        self.weight_decay = weight_decay
        
        self.L = simple_num
        self.alpha = alpha
        
        self.train_logger = init_log(Train_Log_Path, "VAE_prob")
        self.test_logger = init_log(Test_Log_Path, "VAE_prob")

    def train(self):
        self.train_logger.info("Start train VAE prob with KDD99...")
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.net.train()
        train_loss = 0.0
        start_time = time.time()
        for epoch in range(self.epoch):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            count_batch = 0
            for item in self.trainloader:
                data, _, _ = item
                recon_data, mu ,logvar = self.net(data)
                loss = loss_func(recon_data, data, mu, logvar)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss
                count_batch += 1

            epoch_using_time = time.time() -epoch_start_time
            self.train_logger.info("Epoch{}/{} average training time of each batch:{.3f}\t average training loss of each batch:{.8f}"
                                   .format(epoch, self.epoch, epoch_using_time/ count_batch, epoch_loss / count_batch, ))
            train_loss += epoch_loss
        using_time = time.time() -start_time
        self.train_logger.info("Training time:{.3f}\t average training loss of each epoch:{.8f}".format(using_time, train_loss/self.epoch))
        self.train_logger.info("Finish train VAE prob with KDD99.")
        
    def test(self):
        index_list = []
        prediction_list = []
        label_list = []
        index_label_prediction = []
        
        self.test_logger.info("Start test VAE prob with KDD99...")
        self.net.eval()
        start_time = time.time()
        with torch.no_grad():
            for item in self.testloader:
                data, label, index = item
                label_list.append(label)
                index_list.append(index)
                mu, logvar = self.net.encode(data)
                # get L simples and recon_data, and compute prob
                prob_data = 0.0
                for i in range(self.L):
                    z = self.net.reparameterize(mu, logvar)
                    recon_data = self.net.decode(z)
                    prob_var = recon_data - data
                    prob = multivariate_normal.pdf(data, recon_data, np.diag(prob_var))
                    prob_data += prob
                prob_data /= self.L
                
                if prob_data < self.alpha:
                    prediction_list.append(1)# anomaly
                else:
                    prediction_list.append(0) # normal

                # 打印label和预测结果
                self.test_logger.info("index:{}\t label:{}\t prediction:{}\t probability:{}\t mu:{}\t var:{}\t"
                                      .format(index, label, prediction_list[index], prob_data, mu, logvar.exp()))
            index_label_prediction = list(zip(index_list, label_list, prediction_list))
            using_time = time.time() - start_time
            self.test_logger.info("detection time:{.3f}".format(using_time))
        # 输出性能
        per_obj = performance(index_label_prediction)
        per_obj.get_base_metrics()
        per_obj.AUC_ROC()

        self.test_logger.info("accurancy:{:.5f}\t precision:{:.5f}\t recall:{:.5f}\t f1score:{:.5f}\t AUC:{:.5f}\t".
                    format(per_obj.accurancy, per_obj.precision, per_obj.recall, per_obj.f1score, per_obj.AUC))

        self.test_logger.info("Finishing testing VAE prob with Kdd99...")
   
def loss_func(recon_x, x, mu, logvar):
    loss_func = torch.nn.MSELoss()
    BCK = loss_func(recon_x, x)
    
    var = logvar.exp()
    KL = 0.5 * torch.sum(1 + logvar - mu.pow(2) - var)
    return BCK + KL