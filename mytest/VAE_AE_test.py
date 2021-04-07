import matplotlib.pyplot as plt
import csv
import math
from other.path import Picture, Performance
from network.AE_KDD99 import AE_KDD99
from network.VAE_KDD99 import VAE_KDD99
from optim.AE_KDD99_trainer import AE_KDD99_trainer
from optim.VAE_KDD99_trainer import VAE_Kdd99_trainer
from optim.VAE_prob_KDD99_trainer import VAE_prob_KDD99_trainer

"""
   save all the performance data into /other/performance/vae_ae_comp.csv 
   no print
"""
class VAE_AE_test():
    def __init__(self, trainloader, testloader, ae_epoch, vae_epoch, vae_prob_epoch, ae_cluster_num, vae_prob_sample_num, vae_prob_alpha,
                 lr:float=0.001, weight_decay:float=1e-6):
        self.trainloader = trainloader
        self.testloader = testloader
        self.lr = lr
        self.weight_decay = weight_decay

        self.ae_epoch = ae_epoch
        self.vae_epoch = vae_epoch
        self.vae_prob_epoch = vae_prob_epoch

        self.ae_cluster_num = ae_cluster_num

        self.prob_sample_num = vae_prob_sample_num
        self.prob_alpha = vae_prob_alpha

        self.accurancy = []
        self.precision = []
        self.recall = []
        self.FPR = []
        self.MCC = []
        self.train_time = []
        self.detection_time = []

    # generate network and train
    def get_param(self):
        # AE
        ae_net = AE_KDD99()
        ae_trainer = AE_KDD99_trainer(ae_net, self.trainloader, self.testloader, self.ae_epoch, self.lr, self.weight_decay, self.ae_cluster_num)
        ae_trainer.train()
        ae_trainer.get_param()
        ae_trainer.test()
        self.train_time.append(ae_trainer.train_time)
        self.detection_time.append(ae_trainer.get_param_time+ae_trainer.test_time)
        acc, pre, recall, FPR, MCC = metric(ae_trainer.index_label_prediction)
        self.accurancy.append(acc)
        self.precision.append(pre)
        self.recall.append(recall)
        self.FPR.append(FPR)
        self.MCC.append(MCC)

        # VAE
        vae_net = VAE_KDD99()
        vae_trainer = VAE_Kdd99_trainer(vae_net, self.trainloader, self.testloader, self.vae_epoch, self.lr, self.weight_decay)
        vae_trainer.train()
        vae_trainer.get_normal_parm()
        vae_trainer.test()
        self.train_time.append(vae_trainer.train_time)
        self.detection_time.append(vae_trainer.get_param_time+vae_trainer.test_time)
        acc, pre, recall, FPR, MCC = metric(vae_trainer.index_label_prediction)
        self.accurancy.append(acc)
        self.precision.append(pre)
        self.recall.append(recall)
        self.FPR.append(FPR)
        self.MCC.append(MCC)

        # VAE prob
        vae_prob_net = VAE_KDD99()
        vae_prob_trainer = VAE_prob_KDD99_trainer(vae_prob_net, self.trainloader, self.testloader, self.vae_prob_epoch, self.lr, self.weight_decay, self.prob_sample_num, self.prob_alpha)
        vae_prob_trainer.train()
        vae_prob_trainer.test()
        self.train_time.append(vae_prob_trainer.train_time)
        self.detection_time.append(vae_prob_trainer.test_time)
        acc, pre, recall, FPR, MCC = metric(vae_prob_trainer.index_label_prediction)
        self.accurancy.append(acc)
        self.precision.append(pre)
        self.recall.append(recall)
        self.FPR.append(FPR)
        self.MCC.append(MCC)

    def my_plot(self):
        my_bar(self.accurancy, "accurancy")
        my_bar(self.precision, "precision")
        my_bar(self.recall, "recall")
        my_bar(self.FPR, "FPR")
        my_bar(self.MCC, "MCC")
        my_bar(self.train_time, "training time")
        my_bar(self.detection_time, "detection time")

    def save_data(self):
        performance = list(zip(self.accurancy, self.precision, self.recall, self.FPR, self.MCC, self.detection_time, self.train_time))
        header = ['accurancy', 'precision', 'recall', 'FPR', 'MCC', 'detection time', 'train time']
        file_path = Performance + "vae_ae_kdd_comp.csv"
        file = open(file=file_path, mode='w', newline='')
        writer = csv.writer(file, dialect='excel')
        writer.writerow(header)
        for item in performance:
            writer.writerow(item)
        file.close()

def metric(index_label_prediction:list):
    index, label, prediction, _ = zip(*index_label_prediction)
    TP = 0.0001
    TN = 0.0001
    FP = 0.0001
    FN = 0.0001
    for i in range(len(index)):
        if label[i] != 0:
            if prediction[i] == 1:
                TP += 1
            else:
                FN += 1
        else:
            if prediction[i] == 1:
                FP += 1
            else:
                TN += 1

    N = TN + TP + FN + FP
    acc = (TP + TN) / N
    pre = TP / (TP + FP)
    recall = TP / (TP + FN)
    FPR = FP / (TN + FP)

    S = (TP + FN) / N
    P = (TP + FP) / N
    MCC = (TP/N-S*P)/math.sqrt(P*S*(1-S)*(1-P))
    return acc, pre, recall, FPR, MCC

def my_bar(y, name:str):
    bar_width = 0.1
    x = [0.2, 0.4, 0.6]
    x_label = ['AE', 'VAE', 'VAE prob']
    # {'/', '\', '|', '-', '+', 'x', 'o', 'O', '.', '*'}
    plt.bar(x[0], height=y[0], width=bar_width, hatch='x', color='w', label="AE", edgecolor='k')
    plt.bar(x[1], height=y[1], width=bar_width, hatch='/', color='w', label="VAE", edgecolor='k')
    plt.bar(x[2], height=y[2], width=bar_width, hatch='.', color='w', label="VAE prob", edgecolor='k')
    plt.xticks(x, x_label)
    plt.xlim((0.0, 1.0))
    plt.ylabel(name)
    plt.title(name + ' of AE, VAE and VAE prob')
    plt.legend(loc="upper right")
    # save the figure
    plt.savefig(Picture + name + ".jpg")
    plt.show()
    plt.close()