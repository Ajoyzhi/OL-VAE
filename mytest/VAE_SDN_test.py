from optim.VAE_SDN_trainer import VAE_SDN_trainer
from dataset.SDN.processd.SDN_dataloader import sdnDataloader
from dataset.SDN.processd.data_proccess import dataprocess
from dataset.SDN.processd.data_join import datajoin
from dataset.SDN.processd.utils import Utiles
from other.path import pro_root, raw_root
from network.VAE_SDN import VAE_SDN
import glob
from other.path import Train_Log_Path, Test_Log_Path, Picture
from other.log import init_log
import matplotlib.pyplot as plt
import csv


class VAE_SDN_test():
    def __init__(self, alpha:float=0):
        self.alpha = alpha

        self.switch_num = 5
        self.net = {}
        self.net = self.net.fromkeys(range(self.switch_num))
        self.normal_mu = {}
        self.normal_var = {}
        self.threshold = {}

        self.id_prob_loss_label = {}
        self.each_threshold = {}

    def process_data(self):
        # compute rate
        for j in range(self.switch_num):
            i = j + 1
            fe_src = raw_root + "s" + str(i) + "_fe.csv"
            fe_dst = raw_root + "s" + str(i) + "_fe_pro.csv"
            port_src = raw_root + "s" + str(i) + "_port.csv"
            port_dst = raw_root + "s" + str(i) + "_port_pro.csv"
            dataprocess(src_fe_path=fe_src,
                        dst_fe_path=fe_dst,
                        src_port_path=port_src,
                        dst_port_path=port_dst)
        # combine fe and port
        for j in range(self.switch_num):
            i = j + 1
            fe_com = fe_dst = raw_root + "s" + str(i) + "_fe_pro.csv"
            port_com = raw_root + "s" + str(i) + "_port_pro.csv"
            com_path = pro_root + "combine" + str(i) + ".csv"
            datajoin(src_fe_path=fe_com,
                     src_port_path=port_com,
                     dst_path=com_path).join()

        # normalize the data
        for j in range(self.switch_num):
            i = j + 1
            src_com = pro_root + "combine" + str(i) + ".csv"
            dst_nor = pro_root + "normalization" + str(i) + ".csv"
            slice = pro_root
            u = Utiles(combine_path=src_com,
                       dst_path=dst_nor,
                       slice_path=slice)
            u.pro_data()
            u.slice_data()

    # each switch has a VAE model
    def detect(self):
        for j in range(self.switch_num):
            i = j + 1
            src_path = pro_root + str(i) + "/*.csv"
            file_num = len(glob.glob(src_path))
            train_logger = init_log(Train_Log_Path, "switch"+str(i))
            tmp_thr = []
            for k in range(file_num):
                file_path = pro_root + str(i) + "/" + str(k) + ".csv"
                sdndataloader = sdnDataloader(src_path=file_path, batch_size=1, shuffle=False, num_worker=0).loader
                # 如果还未建立网络结构，则新建网络结构并初始化参数
                if self.net[j] is None:
                    print("init VAE network of switch{}...".format(i))
                    # output:switch,file_no
                    train_logger.info("switch{},file{} is training.".format(i, k))
                    vae_sdn = VAE_SDN()
                    sdn_trainer = VAE_SDN_trainer(net=vae_sdn, dataloader=sdndataloader, logger=train_logger)
                    sdn_trainer.train(epochs=5, lr=0.001, weight_decay=1e-6)
                    sdn_trainer.get_normal_parm()

                    self.net[j] = sdn_trainer.net
                    self.normal_mu[j] = sdn_trainer.train_mu
                    self.normal_var[j] = sdn_trainer.train_var
                    self.threshold[j] = sdn_trainer.threshold_mean
                    tmp_thr.append(self.threshold[j])
                # 如果已经建立网络结构，就正常训练网络，等待检测的信号
                # 出现异常信号，检测数据，并清空网络结构
                elif k == 221:
                    self.each_threshold[j] = tmp_thr
                    print("start detecting abnormal data of switch{}...".format(i))
                    print("switch{},file{} is detecting.".format(i, k))
                    sdn_trainer = VAE_SDN_trainer(net=self.net[j], dataloader=sdndataloader, logger=None)
                    sdn_trainer.test(normal_mu=self.normal_mu[j], normal_var=self.normal_var[j], threshold=self.threshold[j])
                    self.net[j] = VAE_SDN()
                    self.id_prob_loss_label[j] = sdn_trainer.id_prob_loss_prediction
                    break
                # 否则，正常训练网络,更新参数
                else:
                    print("start training VAE network of switch{}..".format(i))
                    train_logger.info("switch{},file{} is training.".format(i, k))
                    sdn_trainer = VAE_SDN_trainer(net=self.net[j], dataloader=sdndataloader, logger=train_logger)
                    sdn_trainer.train(epochs=5, lr=0.001, weight_decay=1e-6)
                    sdn_trainer.get_normal_parm()

                    self.net[j] = sdn_trainer.net
                    # 参数只与当前文件中的数据相关，可以使用加权平均（后面为历史数据）
                    self.normal_mu[j] = (1 - self.alpha) * sdn_trainer.train_mu + self.alpha * self.normal_mu[j]
                    self.normal_var[j] = (1 - self.alpha) * sdn_trainer.train_var + self.alpha * self.normal_var[j]
                    self.threshold[j] = (1 - self.alpha) * sdn_trainer.threshold_mean + self.alpha * self.threshold[j]
                    tmp_thr.append(self.threshold[j])

    # 画图显示分类数据与阈值
    def get_result(self):
        for i in range(self.switch_num):
            # 对所有的threshold画一个散点图
            plt.figure(figsize=(20, 16))
            plt.subplot(121)
            id, prob, loss, label = zip(*self.id_prob_loss_label[i])
            # 待检测点prob与threshold的关系
            plt.axhline(y=self.threshold[i])
            plt.scatter(x=range(len(prob)), y=prob, c=label)
            plt.title('switch' + str(i+1) + 'prob_threshold')
            # 每个训练文件的threshold
            plt.subplot(122)
            plt.axhline(y=self.threshold[i])
            plt.scatter(x=range(len(self.each_threshold[i])), y=self.each_threshold[i])
            plt.title('switch' + str(i+1) + 'each_threshold')
            # 保存文件
            plt.savefig(Picture + '/threshold/' + 'switch' + str(i+1) + '.jpg')
            plt.show()
            plt.close()

            # 输出csv文件
            file_head = ['id', 'prob', 'loss', 'label']
            file = open(Test_Log_Path + 'switch' + str(i+1) + '.csv', mode='w', newline='')
            file_write = csv.writer(file, dialect='excel')
            file_write.writerow(file_head)
            for item in self.id_prob_loss_label[i]:
                file_write.writerow(item)
            file.close()