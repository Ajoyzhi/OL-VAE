from dataset.KDD99.processed.util import Util
from other.path import Train_Src_Path, Train_Number_Path, Train_Feature_Path, Train_Des_Path

kdd_util = Util(src_path=Train_Src_Path, number_path=Train_Number_Path,
                feature_path=Train_Feature_Path, des_path=Train_Des_Path,
                ratio=0.0001, isTrain=True, FEATURES=9)
kdd_util.get_data()
kdd_util.select_features()
kdd_util.normalizations()