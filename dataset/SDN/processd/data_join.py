"""
    combine fe and port data by (time, dp, inport)
    load data, slice by time and in each slice,combine the same port
"""
import numpy as np
from dataset.SDN.processd.data_proccess import slice_by_time, output_file
"""
    src_fe_path: fe data with rate path
    src_port_path: port data with rate path
    dst_path: fe and port data combined 
"""
class datajoin():
    def __init__(self, src_fe_path:str, src_port_path:str, dst_path:str):
        self.src_fe_path = src_fe_path
        self.src_port_path = src_port_path
        self.dst_path = dst_path
        # load data
        self.fe_array = np.loadtxt(self.src_fe_path, dtype='str', delimiter=',', skiprows=0)
        self.port_array = np.loadtxt(self.src_port_path, dtype='str', delimiter=',', skiprows=0)
        # divide by time
        self.fe_list = slice_by_time(self.fe_array)
        self.port_list = slice_by_time(self.port_array)
        self.len = len(self.fe_list)
        # equal
        print("fe_list_len:", len(self.fe_list),
              "port_list_len:", len(self.port_list))

    """
        fe_list/poer_list的结构：
            list1(array1, array2, array3,...,)
            list2(array1, array2,...,)
            ...
            listn(array1, array2, array3,...,)
    """
    def join(self):
        for i in range(self.len):
            tmp_write = []
            for j in range(len(self.fe_list[i])):
                fe_tmp = self.fe_list[i][j]
                # deal table-miss
                if fe_tmp[2] != '0':
                    for k in range(len(self.port_list[i])):
                        port_tmp = self.port_list[i][k]
                        if fe_tmp[2] == port_tmp[2]:
                            # rx_pkt,rx_bytes,rx_prate，rx_brate，tx_pkt，tx_byte，tx_prate，tx_brate
                            tmp = np.insert(fe_tmp, 8, [port_tmp[3], port_tmp[4], port_tmp[5], port_tmp[6],
                                              port_tmp[7], port_tmp[8], port_tmp[9], port_tmp[10]])
                            tmp_write.append(tmp)
                            break
                else:
                    tmp = np.insert(fe_tmp, 8, [0.0, 0.0, 0.0, 0.0,
                                                0.0, 0.0, 0.0, 0.0])
                    tmp_write.append(tmp)
            output_file(self.dst_path, tmp_write)
