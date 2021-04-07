from dataset.SDN.processd.data_proccess import dataprocess
from other.path import FE1_src, FE1_dst,FE2_src,FE2_dst,FE3_src,FE3_dst,FE4_src,FE4_dst,FE5_src,FE5_dst
from other.path import PORT1_src, PORT1_dst, PORT2_src, PORT2_dst, PORT3_src,PORT3_dst, PORT4_src, PORT4_dst, PORT5_src, PORT5_dst

dataprocess(FE1_src, FE1_dst, PORT1_src, PORT1_dst)
dataprocess(FE2_src, FE2_dst, PORT2_src, PORT2_dst)
dataprocess(FE3_src, FE3_dst, PORT3_src, PORT3_dst)
dataprocess(FE4_src, FE4_dst, PORT4_src, PORT4_dst)
dataprocess(FE5_src, FE5_dst, PORT5_src, PORT5_dst)