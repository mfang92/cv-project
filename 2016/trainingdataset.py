import os
import os.path as osp
from typing import List
import torch
import numpy as np
from image_functions import *

# more general version of Videos
class TrainingDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, read_data, transform=None, target_transform=None, size_lim=None, patches_lim=None):
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
        self.patches_lim = patches_lim

        filenames = np.random.permutation([fn for fn in os.listdir(self.data_dir) if not fn.startswith('.') and osp.isfile(osp.join(self.data_dir, fn))])
        if self.size_lim==None:
            self.fn_list = filenames
        else:
            self.fn_list = filenames[:self.size_lim]
        
        self.data = self.getPatches(33)

    def getPatches(self, f_sub):
        """ 
        Returns list of (generated_fn, tensor)
        """
        res = []
        for fn in self.fn_list:
            raw = self.read_data(os.path.join(self.data_dir, fn))
            blurred = torch_blur(raw, 2, 9)
            
            blurred_patches = getPatches(fn, blurred, f_sub) # list of (fn, patch) 
            raw_patches = getPatches(fn, raw, f_sub) # list of (fn, patch) 

            if (self.patches_lim!=None):
                ind = np.random.permutation(np.arange(len(blurred_patches)))[:self.patches_lim]
                blurred_patches = [blurred_patches[i] for i in ind]
                raw_patches = [raw_patches[i] for i in ind]

            for i, patch in enumerate(blurred_patches):
                fn, blurred = patch
                _, raw = raw_patches[i]
                res.append((fn, blurred, raw))

            # for blurred_patch, raw_patch in zip(blurred_patches, raw_patches):
            #     fn, blurred_patch = blurred_patch
            #     _, raw_patch = raw_patch
            #     # res.append((fn, blurred_patch, raw_patch))
            #     res.append((fn, blurred))
        return res

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # fn, blurred, raw  = self.data[idx]

        # if self.transform:
        #     blurred = self.transform(blurred)
        # if self.target_transform:
        #     raw = self.target_transform(raw)
        # return blurred, raw

        fn, blurred, raw  = self.data[idx]

        # turn to 3 channel if black and white
        if blurred.shape[0]==1:
            blurred = blurred.expand(3, -1, -1)
        if raw.shape[0]==1:
            raw = raw.expand(3, -1, -1)
        
        target = raw

        if self.transform:
            blurred = self.transform(blurred)
        if self.target_transform:
            target = self.target_transform(target)
        return blurred, target
    