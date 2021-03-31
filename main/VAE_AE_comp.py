from dataset.KDD99.processed.KDD99_loader import KDD99_Loader
from mytest.VAE_AE_test import VAE_AE_test

if __name__ == '__main__':
    # 创建训练和测试的kdd99_loader
    kdd99_trian_loader = KDD99_Loader(ratio=0.01, isTrain=True, preprocess=False, loadData=False, batch_size=128, shuffle=False,
                                      num_workers=0).loader  # 97个数据
    kdd99_test_loader = KDD99_Loader(ratio=0.001, isTrain=False, preprocess=False, loadData=False, batch_size=1, shuffle=False,
                                     num_workers=0).loader  # 300个数据
    vae_ae_comp = VAE_AE_test(trainloader=kdd99_trian_loader, testloader=kdd99_test_loader,
                              ae_epoch=50, vae_epoch=50, vae_prob_epoch=50, ae_cluster_num=4,
                              vae_prob_sample_num=10, vae_prob_alpha=1)
    vae_ae_comp.get_param()
    vae_ae_comp.save_data()
    vae_ae_comp.my_plot()