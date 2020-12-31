# 测试kdd99的数值化处理
import time
from dataset.KDD99.processed.util import Util
from other import path
import logging

train_util = Util(path.Train_Src_Path, path.Train_Des_Path, 0.001, True)

start_time = time.time()
# 数据数值化
train_util.get_data()
# 选择特征，按比例加载数据，归一化，并未重新写入文件中
train_util.normalizations()
data = train_util.data
label = train_util.label
final_time = time.time()
# 大概需要7s
print("the time of get and process normal data is: {:.8f}".format(211))


