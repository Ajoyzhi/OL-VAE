from torch.utils.data import Dataset
from other import path
from .util import Util
import numpy as np

"""
    1.__init__中实现对数据按比例选择和标准化，可以选择实现对数据的数值化和特征选择
                ratio：加载数据的比例
                isTrain：是否加载训练数据（True）或者测试数据（False）
                preprocess：是否从对数据进行数值化和选择特征（只需要执行一次，即可从path.feature_path中获取所有已处理好的数据）
    2.__getitem__实现对数据集的按索引获取
    3.__len__实现对数据集大小的获取
"""
class Kdd99Dataset(Dataset):
    def __init__(self, ratio: float, isTrain: bool, preprocess: bool, loadData: bool, FEATURE:int=9):
        # 1. Initialize file path or list of file names.
        self.ratio = ratio
        self.isTrain = isTrain
        self.loadData = loadData

        if self.isTrain:
            self.src_path = path.Train_Src_Path
            self.number_path = path.Train_Number_Path
            self.feature_path = path.Train_Feature_Path
            self.des_path = path.Train_Des_Path
        else:
            self.src_path = path.Test_Src_Path
            self.number_path = path.Test_Number_Path
            self.feature_path = path.Test_Feature_Path
            self.des_path = path.Test_Des_Path

        # get data from the destination_path(train\test)
        if self.loadData:
            data_label = np.loadtxt(self.des_path, dtype=float, delimiter=",", skiprows=0)
            temp_data = np.hsplit(data_label, (FEATURE, FEATURE + 1))
            self.data = temp_data[0]
            self.label = temp_data[1]
        else:
            self.util = Util(src_path=self.src_path,
                        number_path=self.number_path,
                        feature_path=self.feature_path,
                        des_path=self.des_path,
                        ratio=self.ratio,
                        isTrain=self.isTrain)
            if preprocess:
                self.preprocess()
            # 按比例获取数据
            self.util.normalizations()

            self.data = self.util.data
            self.label = self.util.label

    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        # 加入数据加载部分的处理逻辑

        return self.data[index], self.label[index], index

    def __len__(self):
        return len(self.data)

    def preprocess(self):
        # 获取数据
        self.util.get_data()
        # 选择特征
        self.util.select_features()