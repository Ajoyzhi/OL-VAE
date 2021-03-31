from torch.utils.data import DataLoader
from .KDD99_dataset import Kdd99Dataset

class KDD99_Loader():
    def __init__(self, ratio: float, isTrain: bool, preprocess: bool, loadData:bool, batch_size: int, shuffle=True, num_workers: int = 0, ):
        self.ratio = ratio
        self.isTrain = isTrain
        self.preprocess = preprocess
        self.loadData = loadData

        self.batch_szie = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        self.dataset = Kdd99Dataset(ratio=self.ratio, isTrain=self.isTrain, preprocess=self.preprocess, loadData=self.loadData)
        self.loader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_szie,
            shuffle=self.shuffle,
            num_workers=self.num_workers
        )






