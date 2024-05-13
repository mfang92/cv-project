import torch
import torch.nn as nn
import torch

def sub_sample(image, factor):
    """
    image is tensor: (3, h, w)
    """
    return image[:, ::factor, ::factor]

def interpolate(downsampled, factor):
    """
    image is tensor: (3, h, w)
    """
    downsampled = torch.unsqueeze(downsampled, 0) # adds needed dimension for interpolate
    upsampled = nn.functional.interpolate(downsampled, 
                                            scale_factor=(factor, factor), 
                                            mode='bicubic', align_corners=True)
    return upsampled[0, :, :, :] # undoes added dimension
    