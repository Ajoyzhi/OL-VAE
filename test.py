# 测试kdd99的数值化处理
import time
from dataset.KDD99.processed.util import Util
from dataset.KDD99.processed.KDD99_loader import KDD99_Loader
from other import path
import logging



"""
train_util = Util(path.Train_Src_Path, path.Train_Number_Path, path.Train_Feature_Path,path.Train_Des_Path, 0.001, True)
start_time = time.time()
# 数据数值化
# train_util.get_data()
# 选择特征
# train_util.select_features()
train_util.normalizations()
time = time.time()-start_time
# 大概需要7s
print("the time of get and process normal data is: {:.2f}".format(time))

"""
kdd99_train_loader = KDD99_Loader(ratio=0.001, isTrain=True, preprocess=False,
                                  batch_size=10, shuffle=False, num_workers=0).loader
for item in kdd99_train_loader:
    print(item)


