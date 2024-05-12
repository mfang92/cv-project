from typing import List
import os
import os.path as osp
import torch
import numpy as np
import torchvision
import torch

def downsize_frame(frame, factor=2, sigma = 9):
    frame_blurred = torchvision.transforms.GaussianBlur(kernel_size = 9, sigma = (sigma, sigma))(frame)
    return frame_blurred[:, ::factor, ::factor]

def downsample(vid, factor=2, sigma = 9):
    frames, height, width, channels = vid.shape
    downsized_frames = []

    for i in range(frames):
        frame = vid[i]

        output_frame = downsize_frame(frame, factor=factor, sigma = sigma)
        downsized_frames.append(output_frame)

    downsized_frames = torch.stack(downsized_frames)
    return downsized_frames
    
class Videos(torch.utils.data.Dataset):
    def __init__(self, root: str, transform=None, size_lim=None):
        self.root = root
        self.transform = transform
        self.size_lim=size_lim

    @property
    def raw_videos(self) -> List[str]:
        filenames = [fn for fn in os.listdir(self.root) if not fn.startswith('.') and osp.isfile(osp.join(self.root, fn))]
        if self.size_lim==None:
            return filenames
        else:
            return filenames[:self.size_lim]

    def __getitem__(self, idx: int):
        fn = self.raw_videos[idx]
        original = torch.from_numpy(np.load(osp.join(self.root, fn))).float()

        # # this creates 
        # original = torch.einsum('ijkl -> iljk', original)

        if self.transform is not None:
            original = self.transform(original)

        original = torch.einsum('ijkl -> lijk', original)

        down_sample = downsample(original)
        return down_sample, original

    def __len__(self):
        return len(self.raw_videos)

if __name__ == "__main__":
    root = os.getcwd()
    vid_dir = osp.join(root, "tiny")

    transform = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.299, 0.224, 0.225])
    dataset = Videos(vid_dir, transform=transform)
    print(dataset[0][0].shape, dataset[0][1].shape)