import os
import os.path as osp
from typing import List
import torch

# more general version of Videos
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, read_data, transform=None, target_transform=None, size_lim=None):
        self.img_dir = img_dir
        self.read_data = read_data
        self.transform = transform
        self.target_transform = target_transform
        self.size_lim = size_lim
    
    @property
    def fn_list(self) -> List[str]:
        filenames = [fn for fn in os.listdir(self.img_dir) if not fn.startswith('.') and osp.isfile(osp.join(self.img_dir, fn))]
        if self.size_lim==None:
            return filenames
        else:
            return filenames[:self.size_lim]

    def __len__(self):
        return len(self.fn_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.img_dir, self.fn_list[idx])

        data = self.read_data(file_path)
        target = data

        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            target = self.target_transform(target)
        return data, target
    