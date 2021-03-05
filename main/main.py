from dataset.KDD99.processed.KDD99_loader import KDD99_Loader
from network.VAE_KDD99 import VAE_KDD99
from optim.VAE_KDD99_trainer import VAE_Kdd99_trainer
import time

DURATION = 10

if __name__ == '__main__':
    """
    测试VAE对KDD99数据的训练和测试过程
    """
    # 创建训练和测试的kdd99_loader
    kdd99_trian_loader = KDD99_Loader(ratio=0.001, isTrain=True, preprocess=False, batch_size=10, shuffle=False, num_workers=0).loader  # 97个数据
    kdd99_test_loader = KDD99_Loader(ratio=0.001, isTrain=False, preprocess=False, batch_size=1, shuffle=False, num_workers=0).loader  # 300个数据

    # 创建网络结构
    kdd99_vae = VAE_KDD99()
    # 创建trainer，并训练网络
    kdd99_vae_trainer = VAE_Kdd99_trainer(net=kdd99_vae, trainloader=kdd99_trian_loader, testloader=kdd99_test_loader, epochs=20)
    kdd99_vae_trainer.train()
    kdd99_vae_trainer.get_normal_parm()
    kdd99_vae_trainer.test()

    # update 网络结构
    time.sleep(DURATION)
    # 加载新的数据集模拟新收集的正常数据
    kdd99_update_loader = KDD99_Loader(ratio=0.001, isTrain=True, preprocess=False, batch_size=10, shuffle=True, num_workers=0).loader
    # 创建trainer
    kdd99_update_trainer = VAE_Kdd99_trainer(net=kdd99_vae_trainer.net, trainloader=kdd99_update_loader, epochs=20)
    kdd99_update_trainer.update_model()
    # 获取更新数据并测试
    kdd99_update_trainer.get_normal_parm()
    kdd99_update_trainer.test()