import torch
import torch.optim as optim 
from scipy.stats import multivariate_normal
import time
import csv
import numpy as np
from torch.utils.data import DataLoader
from other.path import Train_Log_Path, Test_Log_Path
from other.log import init_log

"""
    save train data(log)
        1. average training time of each batch
        2. average training loss of each batch
        3. Trainer time
        4. average trainer time of each epoch
        5. center of cluster and radius
    into /other/log/train/VAE_prob_KDD99.log
    save test data(csv)
        index label prediction
    into /other/log/test/VAE_prob_KDD99.csv
"""
class VAE_prob_KDD99_trainer():
    def __init__(self, net, trainloader:DataLoader, testloader:DataLoader, epoch:int=10, lr:float=0.001, weight_decay:float=1e-6, simple_num:int=10, alpha:float=1.5e+19):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.epoch = epoch
        self.lr = lr
        self.weight_decay = weight_decay
        
        self.L = simple_num
        self.alpha = alpha
        
        self.train_logger = init_log(Train_Log_Path, "VAE_prob")
        self.train_time = 0.0
        self.test_time = 0.0
        self.index_label_prediction = []

    def train(self):
        self.train_logger.info("Start training VAE prob with KDD99...")
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
                data = data.float()
                recon_data, mu ,logvar = self.net(data)
                loss = loss_func(recon_data, data, mu, logvar)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss
                count_batch += 1

            epoch_using_time = time.time() -epoch_start_time
            self.train_logger.info("Epoch{}/{} triaining tme of each batch:{:.3f}\t average loss of each batch:{:.8f}"
                                   .format(epoch+1, self.epoch, epoch_using_time/ count_batch, epoch_loss / count_batch))
            train_loss += epoch_loss
        self.train_time = time.time() - start_time
        self.train_logger.info("training time:{:.3f}\t average training loss of each epoch:{:.8f}".format(self.train_time, train_loss/self.epoch))
        self.train_logger.info("Finish training VAE prob with KDD99.")

    def test(self):
        index_list = []
        prediction_list = []
        label_list = []
        prob_list = []

        self.net.eval()
        start_time = time.time()
        with torch.no_grad():
            for item in self.testloader:
                data, label, index = item
                data = data.float()
                label_list.append(label)
                index_list.append(index)
                mu, logvar = self.net.encode(data)
                # get L simples and recon_data, and compute prob
                prob_data = 0.0
                for i in range(self.L):
                    z = self.net.reparameterize(mu, logvar)
                    # tensor[[1*15]]
                    recon_data = self.net.decode(z)
                    data_minus = (data - recon_data).numpy()
                    cov_data1 = 1/14 * data_minus.T.dot(data_minus)
                    # 防止协方差非半正定
                    cov_data = cov_data1 + 0.0001 * np.identity(15)
                    # tensor[15]
                    mu_data = recon_data.squeeze(0)
                    prob = multivariate_normal.pdf(data, mu_data, cov_data)
                    prob_data += prob
                prob_data /= self.L
                prob_list.append(prob_data)
                
                if prob_data < self.alpha:
                    prediction_list.append(1)# anomaly
                else:
                    prediction_list.append(0) # normal

            self.index_label_prediction = list(zip(index_list, label_list, prediction_list, prob_list))
        self.test_time = time.time() - start_time
        # save test result into csv
        filepath = Test_Log_Path + "VAE_prob_KDD99.csv"
        file = open(file=filepath, mode='w', newline='')
        writer = csv.writer(file, dialect='excel')
        header = ['index', 'label', 'prediction','prob']
        writer.writerow(header)
        for item in self.index_label_prediction:
            writer.writerow(item)
        file.close()
   
def loss_func(recon_x, x, mu, logvar):
    loss_func = torch.nn.MSELoss()
    MSE = loss_func(recon_x, x)
    
    var = logvar.exp()
    KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - var)
    return MSE + KL