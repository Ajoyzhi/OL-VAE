from dataset.KDD99.processed.KDD99_loader import KDD99_Loader
from optim.AE_KDD99_trainer import AE_KDD99_trainer
from network.AE_KDD99 import AE_KDD99
from mytest.VAE_AE_test import metric

# 测试kdd99的数值化处理
# trainer data：97278； test_data：正常60593 异常223289
kdd99_train_loader = KDD99_Loader(ratio=0.01, isTrain=True, preprocess=False, loadData=True,
                                  batch_size=128, shuffle=False, num_workers=0).loader
kdd99_test_loader = KDD99_Loader(ratio=0.001, isTrain=False, preprocess=False,loadData=True,
                                 batch_size=1, shuffle=False, num_workers=0).loader

# 创建kdd99_VAE_Guass网络结构
kdd99_ae = AE_KDD99()
# 创建trainer，并训练网络
kdd99_ae_trainer = AE_KDD99_trainer(net=kdd99_ae, trainloader=kdd99_train_loader, testloader=kdd99_test_loader, epoch=50, cluster_num=4)
kdd99_ae_trainer.train()
kdd99_ae_trainer.get_param()
# 测试算法性能
kdd99_ae_trainer.test()
acc, pre, recall, FPR, MCC = metric(kdd99_ae_trainer.index_label_prediction)
print("acc:", acc,
      "pre:", pre,
      "recall:", recall,
      "FPR:", FPR,
      "MCC:", MCC)