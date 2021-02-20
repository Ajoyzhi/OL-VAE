from dataset.KDD99.processed.KDD99_loader import KDD99_Loader
# 网络结构
from network.VAE_KDD99 import VAE_KDD99
# trainer
from optim.VAE_KDD99_trainer import VAE_Kdd99_trainer

if __name__ == '__main__':
    """
    测试VAE对KDD99数据的训练和测试过程
    """
    # 创建训练和测试的kdd99_loader
    kdd99_trian_loader = KDD99_Loader(ratio=0.001, isTrain=True, preprocess=False,
                                      batch_size=10, shuffle=False, num_workers=0).loader
    kdd99_test_loader = KDD99_Loader(ratio=0.001, isTrain=False, preprocess=False,
                                     batch_size=10, shuffle=False, num_workers=0).loader

    # 创建kdd99_VAE_Guass网络结构
    kdd99_vae = VAE_KDD99()
    # 创建trainer，并训练网络
    kdd99_vae_trainer = VAE_Kdd99_trainer(net=kdd99_vae, trainloader=kdd99_trian_loader, testloader=kdd99_test_loader, epochs=20)
    kdd99_vae_trainer.train()
    kdd99_vae_trainer.get_normal_parm()
"""
    # 创建kdd99_OLVAE网络解结构
    kdd99_olvae = OL_VAE()
    # 创建trainer，并训练网络
    kdd99_olvae_trainer = OLVAE_Kdd99_trainer(net=kdd99_olvae, trainloader=kdd99_trian_loader, testloader=kdd99_test_loader, epochs=20)
    kdd99_olvae_trainer.train()
    kdd99_olvae_trainer.test()
"""