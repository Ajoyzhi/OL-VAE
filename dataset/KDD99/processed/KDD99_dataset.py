import torch
from torch.utils.data import Dataset

class Kdd99Dataset(Dataset):
    def __init__(self, data, label):
        # 1. Initialize file path or list of file names.
        self.data = torch.Tensor(data)
        self.label = torch.Tensor(label)

    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)