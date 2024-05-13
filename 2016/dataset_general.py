import os
import os.path as osp
from typing import List
import torch
import numpy as np

# more general version of Videos
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, read_data, transform=None, target_transform=None, size_lim=None):
        """
            data_dir: path to folder with files; string
            read_data: function that reads file; (path_to_file: string) -> tensor
            transform: function that transforms data into input for model; (tensor) -> tensor
            target_transform: function that transforms data into expected output of model; (tensor) -> tensor
            size_lim: limit of how many files to include in dataset; number
        """
        self.data_dir = data_dir
        self.read_data = read_data
        self.transform = transform
        self.target_transform = target_transform
        self.size_lim = size_lim

        filenames = np.random.permutation([fn for fn in os.listdir(self.data_dir) if not fn.startswith('.') and osp.isfile(osp.join(self.data_dir, fn))])
        if self.size_lim==None:
            self.fn_list = filenames
        else:
            self.fn_list = filenames[:self.size_lim]

    def getlist(self):
        res = []
        for fn in self.fn_list:
            data = self.read_data(os.path.join(self.data_dir, fn))
            target = data
            
            if self.transform:
                data = self.transform(data)
            if self.target_transform:
                target = self.target_transform(target)

            res.append(data, target)

        return res


    def __len__(self):
        return len(self.fn_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.fn_list[idx])

        data = self.read_data(file_path)
        target = data

        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            target = self.target_transform(target)
        return data, target
    