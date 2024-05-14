import torch
import torch.nn as nn
import torch
import os
import torchvision

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

def torch_blur(image, sigma, kernel_size):
    # torch implementation:
    return torchvision.transforms.GaussianBlur(kernel_size, sigma = (sigma, sigma))(image)

def getPatches(fn, image, f_sub):
    """
        input: tensor (3, h, w)
        returns list of (generated_fn, tensor)
    """

    depth, height, width = image.shape
    patches = []
    for i in range(0, height//f_sub):
        for j in range(0, width//f_sub):
            patch = image[:, i*f_sub:(i+1)*f_sub, j*f_sub:(j+1)*f_sub]
            base, extension = os.path.splitext(fn)
            name = f"{base}_{i}_{j}_{extension}"
            patches.append((name, patch))

    return patches