"""
1. 从util中获取data和label
2. 输入data和label，在此处封装为dataset,并定义loader
"""
from torch.utils.data import DataLoader
from .KDD99_dataset import Kdd99Dataset

class KDD99_Loader():
    def __init__(self, data, label, batch_size: int, shuffle=True, num_workers: int = 0):
        self.data = data
        self.label = label

        self.dataset = None
        self.loader = None

        self.batch_szie = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def kdd_loader(self):
        self.dataset = Kdd99Dataset(self.data, self.label)
        self.loader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_szie,
            shuffle=self.shuffle,
            num_workers=self.num_workers
        )
        # 可以再优化
        return self.loader





