from torch.utils.data import DataLoader
from .SDN_dataset import sdnDataset

class sdnDataloader():
    def __init__(self, src_path: str, batch_size: int=1, shuffle: bool=True, num_worker: int=0):
        self.src_path = src_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_worker = num_worker

        self.dataset = sdnDataset(self.src_path)
        self.loader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_worker
        )