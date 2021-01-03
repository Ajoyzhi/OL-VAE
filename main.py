from dataset.KDD99.processed.util import Util
from dataset.KDD99.processed.KDD99_loader import KDD99_Loader
from network.VAE_Gauss import VAE_KDD99
from optim.VAE_KDD99_trainer import VAE_Kdd99_trainer
from other import path
import time



if __name__ == '__main__':
    """
    测试VAE对KDD99数据的训练和测试过程
    """
    start_time = time.time()
    # 加载kdd99的训练数据和测试
    train_util = Util(path.Train_Src_Path, path.Train_Des_Path, ratio=0.001, isTrain=True)# 97个数据
    test_util = Util(path.Test_Src_Path, path.Test_Des_Path, ratio=0.001, isTrain=False)# 300个测试数据

    # 训练数据数值化,选择特征，按比例加载数据，归一化，并未重新写入文件中
    train_util.get_data()
    train_util.normalizations()
    train_data = train_util.data
    train_label = train_util.label

    # 测试数据数值化，选择特征，按比例加载数据，归一化，并未重新写入文件中
    test_util.get_data()
    test_util.normalizations()
    test_data = test_util.data
    test_label = test_util.label

    process_KDD90_time = time.time() - start_time
    print("The time to process al kdd99 data is {:.3f}".format(process_KDD90_time))

    # 创建训练和测试的kdd99_loader
    kdd99_trian_loader = KDD99_Loader(train_data, train_label, batch_size=10, shuffle=True, num_workers=0).kdd_loader()
    kdd99_test_loader = KDD99_Loader(test_data, test_label, batch_size=1, shuffle=False, num_workers=0).kdd_loader()

    # 创建kdd99_VAE
    kdd99_vae = VAE_KDD99()
    # 创建trainer，并训练网络
    kdd99_trainer = VAE_Kdd99_trainer(net=kdd99_vae, trainloader=kdd99_trian_loader, testloader=kdd99_test_loader, epochs=20)
    kdd99_trainer.train()
    kdd99_trainer.test()

