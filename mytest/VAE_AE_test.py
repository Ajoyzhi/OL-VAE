import matplotlib.pyplot as plt
from other.path import Picture
from network.AE_KDD99 import AE_KDD99
from network.VAE_KDD99 import VAE_KDD99
from optim.AE_KDD99_trainer import AE_KDD99_trainer
from optim.VAE_KDD99_trainer import VAE_Kdd99_trainer
from optim.VAE_prob_KDD99_trainer import VAE_prob_KDD99_trainer

class VAE_AE_test():
    def __init__(self, trainloader, testlaoder, epoch:int=10, lr:float=0.001, weight_decay:float=1e-6, prob_simple_num:int=10, prob_alpha:float=0.5):
        self.trainloader = trainloader
        self.testloader = testlaoder
        self.epoch = epoch
        self.lr = lr
        self.weight_decay = weight_decay

        self.prob_simple_num = prob_simple_num
        self.prob_alpha = prob_alpha

        self.accurancy = []
        self.precision = []
        self.FPR = []
        self.train_time = []
        self.detection_time = []

    # generate network and train
    def get_param(self):
        # AE
        ae_net = AE_KDD99()
        ae_trainer = AE_KDD99_trainer(ae_net, self.trainloader, self.testloader, self.epoch, self.lr, self.weight_decay)
        ae_trainer.train()
        ae_trainer.test()
        self.train_time.append(ae_trainer.train_time)
        self.detection_time.append(ae_trainer.test_time)
        acc, pre, FPR = metric(ae_trainer.index_label_prediction)
        self.accurancy.append(acc)
        self.precision.append(pre)
        self.FPR.append(FPR)

        # VAE
        vae_net = VAE_KDD99()
        vae_trainer = VAE_Kdd99_trainer(vae_net, self.trainloader, self.testloader, self.epoch, self.lr, self.weight_decay)
        vae_trainer.train()
        vae_trainer.get_normal_parm()
        vae_trainer.test()
        self.train_time.append(vae_trainer.train_time)
        self.detection_time.append(vae_trainer.get_param_time+vae_trainer.test_time)
        acc, pre, FPR = metric(vae_trainer.index_label_prediction)
        self.accurancy.append(acc)
        self.precision.append(pre)
        self.FPR.append(FPR)

        # VAE prob
        vae_prob_trainer = VAE_prob_KDD99_trainer(vae_net, self.trainloader, self.testloader, self.epoch, self.lr, self.weight_decay, self.prob_simple_num, self.prob_alpha)
        vae_prob_trainer.train()
        vae_prob_trainer.test()
        self.train_time.append(vae_prob_trainer.train_time)
        self.detection_time.append(vae_prob_trainer.test_time)
        acc, pre, FPR = metric(vae_prob_trainer.index_label_prediction)
        self.accurancy.append(acc)
        self.precision.append(pre)
        self.FPR.append(FPR)

    def my_plot(self):
        my_bar(self.accurancy, "accurancy")
        my_bar(self.precision, "precision")
        my_bar(self.FPR, "FPR")
        my_bar(self.train_time, "training time")
        my_bar(self.detection_time, "detection time")

def metric(index_label_prediction:list):
    index, label, prediction = zip(*index_label_prediction)
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(index)):
        if label[i] == 1:
            if prediction[i] == 1:
                TP += 1
            else:
                FN += 1
        else:
            if prediction[i] == 1:
                FP += 1
            else:
                TN += 1

    acc = (TP + TN) / (TP + FP + TN + FN)
    pre = TP / (TP + FP)
    FTP = FP / (TN + FP)

    return acc, pre, FTP

def my_bar(y, name:str):
    bar_width = 0.1
    x = [0.3, 0.6, 0.9]
    x_label = ['AE', 'VAE', 'VAE prob']
    plt.bar(x[0], height=y[0], width=bar_width, hatch='x', color='w', label="AE", edgecolor='k')
    plt.bar(x[1], height=y[1], width=bar_width, hatch='+', color='w', label="VAE", edgecolor='k')
    plt.bar(x[2], height=y[2], width=bar_width, hatch='*', color='w', label="VAE prob", edgecolor='k')
    plt.xticks(x, x_label)
    plt.xlim((0.0, 1.0))
    plt.ylabel(name)
    plt.title(name + ' of AE, VAE and VAE prob')
    plt.legend()
    # save the figure
    plt.savefig(Picture + name + ".jpg")
    plt.show()
    plt.close()