from dataset.KDD99.processed.KDD99_loader import KDD99_Loader
from optim.VAE_KDD99_trainer import VAE_Kdd99_trainer
from network.VAE_KDD99 import VAE_KDD99
from mytest.VAE_AE_test import metric

# 测试kdd99的数值化处理
# trainer data：97278； test_data：正常60593 异常223289
kdd99_train_loader = KDD99_Loader(ratio=0.01, isTrain=True, preprocess=False,
                                  batch_size=128, shuffle=False, num_workers=0).loader
kdd99_test_loader = KDD99_Loader(ratio=0.001, isTrain=False, preprocess=False,
                                 batch_size=1, shuffle=False, num_workers=0).loader
# 创建kdd99_VAE_Guass网络结构
kdd99_vae = VAE_KDD99()
# 创建trainer，并训练网络
kdd99_vae_trainer = VAE_Kdd99_trainer(net=kdd99_vae, trainloader=kdd99_train_loader, testloader=kdd99_test_loader, epochs=50, sample_num=10, quantile=0.95)
kdd99_vae_trainer.train()
kdd99_vae_trainer.get_normal_parm()
# 测试算法性能
kdd99_vae_trainer.test()
acc, pre, recall, FTP, MCC = metric(kdd99_vae_trainer.index_label_prediction)
print("acc:", acc,
      "pre:", pre,
      "recall:", recall,
      "FTP:", FTP,
      "MCC:", MCC)