import csv
import numpy as np

class dataprocess():
    def __init__(self, src_fe_path:str, dst_fe_path:str, src_port_path:str, dst_port_path:str):
        self.src_fe_path = src_fe_path
        self.dst_fe_path = dst_fe_path
        self.src_port_path = src_port_path
        self.dst_port_path = dst_port_path
        self.sleep_time = 30    # 秒

        self.fe_list = []
        self.port_list = []

        # 数据的处理逻辑
        self._load_data()   # 加载数据，初始化array
        self.fe_list = self._slice_by_time(self.fe_array)   # 将数据按照时间分片
        self.port_list = self._slice_by_time(self.port_array)
        self._compute_rate_save()   # 计算增速，并输出到文件中

    def _load_data(self):
        # load fe data
        fe_array = np.loadtxt(self.src_fe_path, dtype=str, delimiter=',', skiprows=0)
        # 先在delay前面插入0,0，作为packet_rate和bytes_rate的占位
        fe_array = np.insert(fe_array, 6, 0.0, axis=1)
        self.fe_array = np.insert(fe_array, 7, 0.0, axis=1)

        # load port data
        port_array = np.loadtxt(self.src_port_path, dtype=str, delimiter=',', skiprows=0)
        # rx_packets_rate
        port_array = np.insert(port_array, 5, 0.0, axis=1)
        # rx_bytes_rate
        port_array = np.insert(port_array, 6, 0.0, axis=1)
        # tx_packet_rate
        port_array = np.insert(port_array, 9, 0.0, axis=1)
        # tx_byte_rate
        self.port_array = np.insert(port_array, 10, 0.0, axis=1)

    def _slice_by_time(self, array):
        list = []
        tmp = []
        row = array.shape[0]
        # 按照时间划分簇
        for i in range(row):
            if i + 1 < row:
                # 比较相邻行的（time）
                if array[i][0] == array[i + 1][0]:
                    tmp.append(array[i])
                else:
                    tmp.append(array[i])
                    list.append(tmp)
                    tmp = []
            else:
                tmp.append(array[i])
                list.append(tmp)
        return list

    def _compute_rate_save(self):
        # fe(time, dp, inport, dst_mac, packets_count, bytes_count, packets_rate, bytes_rate, delay)
        # 比较相邻簇之间的（dp,inport,dst_mac）,计算packet_rate，bytes_rate
        fe_list_len = len(self.fe_list)
        for i in range(fe_list_len):
            if i + 1 < fe_list_len:
                tmp1 = self.fe_list[i]  # 时间相同的一组list
                tmp2 = self.fe_list[i + 1]  # 时间+30s的一组list
            else:
                tmp1 = self.fe_list[i - 1]
                tmp2 = self.fe_list[i]

            for j1 in range(len(tmp1)):
                for j2 in range(len(tmp2)):
                    # (dp,inport,dst_mac)
                    if tmp1[j1][1] == tmp2[j2][1] and tmp1[j1][2] == tmp2[j2][2] and tmp1[j1][3] == tmp2[j2][3]:
                        tmp2[j2][6] = (float(tmp2[j2][4]) - float(tmp1[j1][4])) / self.sleep_time
                        tmp2[j2][7] = (float(tmp2[j2][5]) - float(tmp1[j1][5])) / self.sleep_time
            # 输出第一组数据
            if i == 0:
                output_file(self.dst_fe_path, tmp1)
            output_file(self.dst_fe_path, tmp2)

        # port(time, dp, port, rx_packets, rx_bytes, rx_prate, rx_brate, tx_packets, tx_bytes, tx_prate, tx_brate, delay)
        # 比较相邻簇之间的（dp, port）,计算rx_prate, rx_brate，tx_prate, tx_brate
        port_list_len = len(self.port_list)
        for i in range(port_list_len):
            if i + 1 < port_list_len:
                tmp1 = self.port_list[i]  # 时间相同的一组list
                tmp2 = self.port_list[i + 1]  # 时间+30s的一组list
            else:
                tmp1 = self.port_list[i - 1]
                tmp2 = self.port_list[i]

            for j1 in range(len(tmp1)):
                for j2 in range(len(tmp2)):
                    # (dp,port)
                    if tmp1[j1][1] == tmp2[j2][1] and tmp1[j1][2] == tmp2[j2][2] :
                        tmp2[j2][5] = (float(tmp2[j2][3]) - float(tmp1[j1][3])) / self.sleep_time
                        tmp2[j2][6] = (float(tmp2[j2][4]) - float(tmp1[j1][4])) / self.sleep_time
                        tmp2[j2][9] = (float(tmp2[j2][7]) - float(tmp1[j1][7])) / self.sleep_time
                        tmp2[j2][10] = (float(tmp2[j2][8]) - float(tmp1[j1][8])) / self.sleep_time
            # 输出第一组数据
            if i == 0:
                output_file(self.dst_port_path, tmp1)
            output_file(self.dst_port_path, tmp2)

# 输出到文件
def output_file(file_path, data):
    file = open(file_path, mode='a', newline='')
    csvwriter = csv.writer(file, dialect='excel')
    for item in data:
        csvwriter.writerow(item)
    file.close()
