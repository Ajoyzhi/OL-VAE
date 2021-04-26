import torch
from torch.utils.data import Dataset
import numpy as np

"""
    src_path: normalization data
    dst_path: data with the same time 
"""
class sdnDataset(Dataset):
    def __init__(self, src_path):
        self.src_path = src_path
        # load data
        id = np.loadtxt(self.src_path, dtype=str, delimiter=',', skiprows=0, usecols=(0, 1, 2, 3))
        self.id = id.tolist()
        # print("sdndataset_id:", self.id)
        # data = np.loadtxt(self.src_path, dtype=float, delimiter=',', skiprows=0, usecols=(4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16))
        # data = np.loadtxt(self.src_path, dtype=float, delimiter=',', skiprows=0, usecols=(4, 5, 6, 7, 8, 9, 10, 11, 16))
        data = np.loadtxt(self.src_path, dtype=float, delimiter=',', skiprows=0, usecols=(4, 5, 6, 7, 8, 9, 10, 11))
        self.data = torch.Tensor(data)
        # print("sdn_dataset_data:", self.data)

    def __getitem__(self, index):
        # print("dataset_index:", index, "dataset_data:", self.data[index], "dataset_id:", self.id[index])
        return self.data[index], self.id[index], index

    def __len__(self):
        return len(self.data)