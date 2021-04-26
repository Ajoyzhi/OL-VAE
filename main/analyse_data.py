from other.path import pro_root
import numpy as np
import operator
import matplotlib.pyplot as plt
from other.path import Picture

swtich_num = 5

def myplot(y, name):
    path = Picture + 'SDN_data/'
    plt.plot(range(len(y)), y)
    plt.title(name)
    plt.savefig(path + name + ".jpg")
    plt.close()

for j in range(swtich_num):
    i = j + 1
    file_path = pro_root + "normalization" + str(i) + ".csv"
    id = np.loadtxt(file_path, dtype=str, delimiter=',', skiprows=0, usecols=(1, 2, 3))
    data = np.loadtxt(file_path, dtype=float, delimiter=',', skiprows=0, usecols=(6,7,10,11,14,15))
    id_unique = np.unique(id, axis=0)
    for id_item in id_unique:
        print("id:", id_item)
        pkt_rate = []
        byte_rate = []
        rx_prate = []
        rx_brate = []
        tx_prate = []
        tx_brate = []
        for i in range(len(id)):
            if (id[i] == id_item).all():
                pkt_rate.append(data[i][0])
                byte_rate.append(data[i][1])
                rx_prate.append(data[i][2])
                rx_brate.append(data[i][3])
                tx_prate.append(data[i][4])
                tx_brate.append(data[i][5])
        string2 = id_item[2].replace(':', '')
        name_pre = id_item[0] + "-" + id_item[1] + "-" + string2 + "-"
        myplot(pkt_rate, name_pre + "pkt_rate")
        myplot(byte_rate, name_pre + "byte_rate")
        myplot(rx_prate, name_pre + "rx_prate")
        myplot(rx_brate, name_pre + "rx_brate")
        myplot(tx_prate, name_pre + "tx_prate")
        myplot(tx_brate, name_pre + "tx_brate")
        print("id:", id_item)



