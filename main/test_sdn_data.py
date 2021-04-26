from dataset.SDN.processd.data_proccess import dataprocess
from dataset.SDN.processd.data_join import datajoin
from dataset.SDN.processd.utils import Utiles
from dataset.SDN.processd.SDN_dataloader import sdnDataloader
from other.path import pro_root, raw_root
import glob

switch_num = 5
# compute rate
for j in range(switch_num):
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
for j in range(switch_num):
    i = j + 1
    fe_com = fe_dst = raw_root + "s" + str(i) + "_fe_pro.csv"
    port_com = raw_root + "s" + str(i) + "_port_pro.csv"
    com_path = pro_root + "combine" + str(i) + ".csv"
    datajoin(src_fe_path=fe_com,
             src_port_path=port_com,
             dst_path=com_path).join()

# test utils:normalize the data
for j in range(switch_num):
    i = j + 1
    src_com = pro_root + "combine" + str(i) + ".csv"
    dst_nor =  pro_root + "normalization" + str(i) + ".csv"
    slice = pro_root
    u = Utiles(combine_path=src_com,
               dst_path=dst_nor,
               slice_path=slice)
    u.pro_data()
    u.slice_data()

"""
# test sdnloader---each switch hs a VAE model
for j in range(switch_num):
    i = j + 1
    src_path = pro_root + str(i) + "/*.csv"
    # print("src_path:", src_path)
    # 检测每个交换机下的训练数据
    file_num = len(glob.glob(src_path))
    # print("src_path file num:", len(file_num))
    for k in range(1, file_num):
        file_path = pro_root + str(i) + "/" + str(k) + ".csv"
        # print("file_path:", file_path)
        sdndataloader = sdnDataloader(src_path=file_path, batch_size=1, shuffle=False, num_worker=0).loader
        for item in sdndataloader:
            data, id, index = item
            print("data:", data, "id:", id, "index:", index)
"""