import numpy as np
import os
from pathlib import Path
from dataset.SDN.processd.data_proccess import output_file

class Utiles():
    def __init__(self, combine_path, dst_path, slice_path):
        self.combpath = combine_path
        self.dstpath = dst_path
        self.slicepath = slice_path

    def pro_data(self):
        # id=(time, dp, import, dst_mac)
        self.id = np.loadtxt(self.combpath, dtype=str, delimiter=',', skiprows=0, usecols=(0, 1, 2, 3))
        # data=(packets_count, bytes_count, packets_rate, bytes_rate, rx_packets, rx_bytes, rx_prate, rx_brate, tx_packets, tx_bytes, tx_prate, tx_brate, delay)
        float_data = np.loadtxt(self.combpath, dtype=float, delimiter=',', skiprows=0,
                               usecols=(4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16))
        # normalization and min-max
        nor_data = normalize(float_data)
        self.data = min_max(nor_data)
        # 保存数据
        self.id_data = np.array(np.hstack((self.id, self.data)))
        output_file(self.dstpath, self.id_data)

    def slice_data(self):
        tmp = []
        row = self.id_data.shape[0]
        count = 0
        for i in range(row):
            if i+1 < row:
                if self.id_data[i][0] == self.id_data[i+1][0]:
                    tmp.append(self.id_data[i])
                else:
                    tmp.append(self.id_data[i])
                    dir_path = Path(self.slicepath  + self.id_data[i][1] + "/")
                    # 如果文件夹不存在，就创建文件夹
                    if not dir_path.exists():
                        os.mkdir(dir_path)
                    path = self.slicepath + self.id_data[i][1] + "/" + str(count) + ".csv"
                    output_file(path, tmp)
                    tmp = []
                    count += 1
            else:
                tmp.append(self.id_data[i])
                dir_path = Path(self.slicepath + self.id_data[i][1] + "/")
                # 如果文件夹不存在，就创建文件夹
                if not dir_path.exists():
                    os.mkdir(dir_path)
                path = self.slicepath + self.id_data[i][1] + "/" + str(count) + ".csv"
                output_file(path, tmp)

def z_score_normalization(x):
    x = (x - np.mean(x)) / np.std(x)
    return x

def min_max_normalization(x):
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x

def normalize(x_data):
    for i in range(x_data.shape[1]):
        x_data[:, i] = z_score_normalization(x_data[:, i])
    return x_data

def min_max(x_data):
    """最大最小化数据"""
    for i in range(x_data.shape[1]):
        x_data[:, i] = min_max_normalization(x_data[:, i])
    return x_data