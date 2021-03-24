import torch
import torch.optim as optim 
import time
import numpy as np
from sklearn.cluster import KMeans
from network.AE_KDD99 import AE_KDD99
from torch.utils.data import DataLoader
from other.log import init_log
from other.path import Train_Log_Path,Test_Log_Path

class AE_KDD99_trainer():
    def __init__(self, net:AE_KDD99, trainloader:DataLoader, testloader:DataLoader, epoch:int=10, lr:float=0.001, weight_decay:float=1e-6, cluster_num:int=4):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.epoch = epoch
        self.lr = lr
        self.weight_decay = weight_decay

        self.cluster = cluster_num
        self.kmeans = None
        self.center = []
        self.radius = []
        
        self.train_logger = init_log(Train_Log_Path, "AE_KDD99")
        self.test_logger = init_log(Test_Log_Path, "AE_KDD99")
        self.train_time = 0.0
        self.test_time = 0.0
        self.index_label_prediction = []
        
    def train(self):
        self.train_logger.info("Start train AE with KDD99...")
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_func = torch.nn.MSELoss()
        self.net.train()

        start_time = time.time()
        train_loss = 0.0
        for epoch in range(self.epoch):
            epoch_start_time = time.time()
            count_batch = 0
            epoch_loss = 0.0
            for item in self.trainloader:
                data, _, _ = item
                middle_data, data_recon = self.net(data)
                loss = loss_func(data_recon, data)
                optimizer.zero_grad()
                loss.backforward()
                optimizer.step()

                epoch_loss += loss
                count_batch += 1
            using_time = time.time() - epoch_start_time
            self.train_logger.info("Epoch{}/{} average training time of each batch:{.3f}\t average training loss of each batch:{.8f}"
                                   .format(epoch, self.epoch, using_time/count_batch, epoch_loss/count_batch))
            train_loss += epoch_loss
        self.train_time = time.time() - start_time
        self.train_logger.info("Trainer time:{.3f}\t average trainer time of each epoch:{.8f}"
                               .format(self.train_time, train_loss/self.epoch))
        self.train_logger.info("Finish train AE with KDD99.")

    def get_param(self):
        loss_func = torch.nn.MSELoss()
        loss_all_list = []
        with torch.no_grad:
            for item in self.trainloader:
                data, _, _ = item
                middle_data, data_recon = self.net(data)
                loss = loss_func(data_recon, data)
                # get all the loss(1*10)
                loss_all_list.append(loss)
            # cluster the loss
            loss_all_tensor = torch.Tensor(loss_all_list)
            # loss：1 row
            loss_all = loss_all_tensor.view((1, -1))
            self.kmeans = KMeans(n_clusters=self.cluster)
            self.kmeans.fit(loss_all)
            self.center = self.kmeans.cluster_centers_.tolist()
            self.radius = self.get_radius(loss_all)

    def test(self):
        index_list = []
        label_list = []
        prediction_list = []

        loss_func = torch.nn.MSELoss()
        self.test_logger.info("Start test AE with KDD99...")
        self.net.eval()
        start_time = time.time()
        with torch.no_grad:
            for item in self.testloader:
                data, label, index = item
                index_list.append(index)
                label_list.append(label)
                middle_data, data_recon = self.net(data)
                loss = loss_func(data_recon, data)
                # get loss label batch_size =1
                predict = self.kmeans.predict(loss)
                dis = abs(self.center[predict] - loss)
                if dis > self.radius[predict]:
                    prediction_list.append(1)# anomaly
                else:
                    prediction_list.append(0)# normal

            self.index_label_prediction = list(zip(index_list, label_list, prediction_list))
        self.test_time = time.time() - start_time
        self.test_logger.info("Finish test AE with KDD99.")

    def get_radius(self, X):
        radius = []
        for i in range(self.cluster):
            cls = X[self.kmeans.labels_ == i]
            dis = []
            for k in range(cls.shape[0]):
                dis.append(abs(self.center[i] - cls[k]))
            radius.append(max(dis))
        return radius