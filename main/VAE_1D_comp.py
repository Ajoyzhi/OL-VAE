from mytest.VAE_1D_test import test_VAE_1D

"""
    test online VAE and original VAE with 
        1. different distribution
        2. different number of train data and test data
        3. different epoch 
"""
if __name__ == '__main__':
    # 测试不同分布 N(1,1) N(1,10) N(10,1)
    mu = [1.0, 1.0, 10.0]
    std = [1.0, 10.0, 1.0]
    for i in range(len(mu)):
        test_vae_1d = test_VAE_1D(mu=mu[i], std=std[i], epoch=30, train_num=200, test_num=50, train_batch_size=10, logvar=False)
        # 生成数据
        test_vae_1d.get_dataloader()
        # 训练网络并获取参数
        test_vae_1d.get_param()
        # 画图并保存，没有logvar
        test_vae_1d.plot_fig()
        # 保存数据
        test_vae_1d.save_data()
    """
    # 测试不同的训练数据和测试数据 12张图
    train_num = [100, 500, 1000, 2000]
    test_num = [50, 100, 200]
    for i in range(len(train_num)):
        for j in range(len(test_num)):
            test_vae_1d = test_VAE_1D(mu=1.0, std=1.0, epoch=30, train_num=train_num[i], test_num=test_num[j], train_batch_size=100, logvar=False)
            # 生成数据
            test_vae_1d.get_dataloader()
            # 训练网络并获取参数
            test_vae_1d.get_param()
            # 画图并保存，没有logvar
            test_vae_1d.plot_fig()
            # 保存数据
            test_vae_1d.save_data()

    # 测试不同epoch
    epoch = [10, 30, 50, 100, 200]
    for i in range(len(epoch)):
        test_vae_1d = test_VAE_1D(mu=1.0, std=1.0, epoch=epoch[i], train_num=1000, test_num=200, train_batch_size=100, logvar=False)
        # 生成数据
        test_vae_1d.get_dataloader()
        # 训练网络并获取参数
        test_vae_1d.get_param()
        # 画图并保存，没有logvar
        test_vae_1d.plot_fig()
        # 保存数据
        test_vae_1d.save_data()
        """