import torch
import torch.optim as optim 
import time
import csv
import numpy as np
from sklearn.cluster import KMeans
from network.AE_KDD99 import AE_KDD99
from torch.utils.data import DataLoader
from other.log import init_log
from other.path import Train_Log_Path,Test_Log_Path

"""
    save train data(log)
        1. average training time of each batch
        2. average training loss of each batch
        3. Trainer time
        4. average trainer time of each epoch
        5. center of cluster and radius
    into /other/log/train/AE_KDD99.log
    save test data(csv)
        index label prediction
    into /other/log/test/AE_KDD99.csv
"""
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
        self.train_time = 0.0
        self.test_time = 0.0
        self.get_param_time = 0.0
        self.index_label_prediction = []
        
    def train(self):
        self.train_logger.info("Start training AE with KDD99...")
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.net.train()

        start_time = time.time()
        train_loss = 0.0
        for epoch in range(self.epoch):
            epoch_start_time = time.time()
            count_batch = 0
            epoch_loss = 0.0
            for item in self.trainloader:
                data, _, _ = item
                data = data.float()
                middle_data, data_recon = self.net(data)
                # 对batch数据每一列求和（batch_szie*15）->(1*15)
                scores = torch.sum((data_recon - data) ** 2, dim=tuple(range(1, data_recon.dim())))
                loss = torch.mean(scores)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss
                count_batch += 1
            using_time = time.time() - epoch_start_time
            self.train_logger.info("Epoch {}/{} triaining tme of each batch:{:.3f}\t average loss of each batch:{:.8f}"
                                   .format(epoch+1, self.epoch, using_time/count_batch, epoch_loss/count_batch))
            train_loss += epoch_loss
        self.train_time = time.time() - start_time
        self.train_logger.info("training time:{:.3f}\t average training loss of each epoch:{:.8f}"
                               .format(self.train_time, train_loss/self.epoch))
        self.train_logger.info("Finish training AE with KDD99.")

    def get_param(self):
        self.train_logger.info("Start getting KDD99 normal parameters...")
        # a list of numpy.ndarray
        loss_all = []
        start_time = time.time()
        with torch.no_grad():
            for item in self.trainloader:
                data, _, _ = item
                data = data.float()
                middle_data, data_recon = self.net(data)
                # 10 * 15
                loss = (data_recon - data) ** 2
                # get each data loss 97 * 15
                for each_data_loss in loss:
                    loss_all.append(each_data_loss.numpy())

            loss_all = np.array(loss_all)
            self.kmeans = KMeans(n_clusters=self.cluster).fit(loss_all)
            self.center = self.kmeans.cluster_centers_.tolist()
            self.radius = self.get_radius(loss_all)
            self.train_logger.info("{} clusters' center:{} and radius:{}".format(self.cluster, self.center, self.radius))
        self.get_param_time = time.time() - start_time
        self.train_logger.info("Finish getting KDD99 normal parmeters.")

    def test(self):
        index_list = []
        label_list = []
        prediction_list = []

        self.net.eval()
        start_time = time.time()
        with torch.no_grad():
            for item in self.testloader:
                data, label, index = item
                data = data.float()
                index_list.append(index)
                label_list.append(label)
                middle_data, data_recon = self.net(data)
                loss = (data_recon - data) ** 2
                # get loss label batch_size =1, but predict is a array
                predict = self.kmeans.predict(loss)
                loss = loss.numpy()
                dis = self.manhattan_distance(self.center[predict[0]], loss)
                if dis > self.radius[predict[0]]:
                    prediction_list.append(1)# anomaly
                else:
                    prediction_list.append(0)# normal

            self.index_label_prediction = list(zip(index_list, label_list, prediction_list, index_list))
        self.test_time = time.time() - start_time
        # save test result into csv
        filepath = Test_Log_Path + "AE_KDD99.csv"
        file = open(file=filepath, mode='w', newline='')
        writer = csv.writer(file, dialect='excel')
        header = ['index', 'label', 'prediction', 'index']
        writer.writerow(header)
        for item in self.index_label_prediction:
            writer.writerow(item)
        file.close()

    def get_radius(self, X):
        radius = []
        for i in range(self.cluster):
            cls = X[self.kmeans.labels_ == i]
            dis = []
            for k in range(cls.shape[0]):
                dis.append(self.manhattan_distance(self.center[i], cls[k]))
            # select the 0.9 quantile of dis as radius
            radius.append(np.quantile(dis, 0.9))
        return radius

    def manhattan_distance(self, x, y):
        """ 曼哈顿距离 """
        return np.sum(abs(x - y))