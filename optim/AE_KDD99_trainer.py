import torch
import torch.optim as optim 
import time
from network.AE_KDD99 import AE_KDD99
from torch.utils.data import DataLoader
from other.log import init_log
from other.path import Train_Log_Path,Test_Log_Path

class AE_KDD99_trainer():
    def __init__(self, net:AE_KDD99, trainloader:DataLoader, testloader:DataLoader, epoch, lr, weight_decay):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.epoch = epoch
        self.lr = lr
        self.weight_decay = weight_decay
        
        self.train_logger = init_log(Train_Log_Path, "AE_KDD99")
        self.test_logger = init_log(Test_Log_Path, "AE_KDD99")
        
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
        train_time = time.time() - start_time
        self.train_logger.info("Trainer time:{.3f}\t average trainer time of each epoch:{.8f}"
                               .format(train_time, train_loss/self.epoch))
        self.train_logger.info("Finish train AE with KDD99.")    
    
    def test(self):
        loss_func = torch.nn.MSELoss()
        self.test_logger.info("Start test AE with KDD99...")
        self.net.eval()
        start_time = time.time()
        with torch.no_grad:
            for item in self.testloader:
                data, _, _ = item
                middle_data, data_recon = self.net(data)
                loss = loss_func(data_recon, data)

        self.test_logger.info("Finish test AE with KDD99.")
    