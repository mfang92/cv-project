from typing import List
import os
import os.path as osp
import torch
import numpy as np
import torchvision
import torch
import cv2

def downsize_frame(frame, factor=2, sigma = 9):
    frame_blurred = cv2.GaussianBlur(frame, (sigma, sigma), 0)
    return frame_blurred[:, ::factor, ::factor]

def downsample(vid, factor=2, sigma = 9):
    frames, channels, height, width = vid.shape
    downsized_frames = []

    for i in range(frames):
        if (torch.is_tensor(vid[i])):
            frame = vid[i].numpy()
        else:
            frame = vid[i]

        output_frame = downsize_frame(frame, factor=factor, sigma = sigma)
        downsized_frames.append(output_frame)

    downsized_frames = torch.tensor(np.array(downsized_frames))
    return downsized_frames
    
class Videos(torch.utils.data.Dataset):
    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform

    @property
    def raw_videos(self) -> List[str]:
        filenames = [fn for fn in os.listdir(self.root) if not fn.startswith('.') and osp.isfile(osp.join(self.root, fn))]
        return filenames

    def __getitem__(self, idx: int):
        fn = self.raw_videos[idx]
        original = torch.from_numpy(np.load(osp.join(self.root, fn))).float()

        # this creates 
        original = torch.einsum('ijkl -> iljk', original)

        if self.transform is not None:
            original = self.transform(original)

        down_sample = downsample(original)

        down_sample = torch.einsum('ijkl -> jikl', down_sample)
        original = torch.einsum('ijkl -> jikl', original)
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