from mytest.VAE_SDN_test import VAE_SDN_test

if __name__ == '__main__':
    # 历史数据占比小
    vae_sdn_test = VAE_SDN_test(alpha=0.05)
    # vae_sdn_test.process_data()
    vae_sdn_test.detect()
    vae_sdn_test.get_result()