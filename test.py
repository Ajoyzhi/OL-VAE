# 测试kdd99的数值化处理
import time
from dataset.KDD99.processed.KDD99_loader import KDD99_Loader
from dataset.KDD99.processed.util import Util
from optim.VAE_KDD99_trainer import VAE_Kdd99_trainer
from network.VAE_KDD99 import VAE_KDD99
from other import path
"""
train_util = Util(path.Train_Src_Path, path.Train_Number_Path, path.Train_Feature_Path,path.Train_Des_Path, 0.001, True)
start_time = time.time()
# 数据数值化
# train_util.get_data()
# 选择特征
# train_util.select_features()
train_util.normalizations()
time = time.time()-start_time
# 大概需要7s
print("the time of get and process normal data is: {:.2f}".format(time))
"""

kdd99_train_loader = KDD99_Loader(ratio=0.001, isTrain=True, preprocess=False,
                                  batch_size=10, shuffle=False, num_workers=0).loader
kdd99_test_loader = KDD99_Loader(ratio=0.001, isTrain=False, preprocess=False,
                                     batch_size=10, shuffle=False, num_workers=0).loader

# 创建kdd99_VAE_Guass网络结构
kdd99_vae = VAE_KDD99()
# 创建trainer，并训练网络
kdd99_vae_trainer = VAE_Kdd99_trainer(net=kdd99_vae, trainloader=kdd99_train_loader, testloader=kdd99_test_loader, epochs=20)
kdd99_vae_trainer.train()
kdd99_vae_trainer.get_normal_parm()
# 测试算法性能
kdd99_vae_trainer.test()
