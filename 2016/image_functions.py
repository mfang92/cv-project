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

def customLoss(result, target, criterion):
    """
        result: tensor
        target: tensor
    """

    _, _, result_h, result_w = result.shape
    _, _, target_h, target_w  = target.shape

    assert(result_h <= target_h)
    assert(result_w <= target_w)

    h_pad = (target_h - result_h)//2
    w_pad = (target_w - result_w)//2

    if h_pad%2 or w_pad%2:
        assert("must have even padding")

    h_end = -h_pad if h_pad else result_h
    w_end = -w_pad if w_pad else result_w

    return criterion(result, target[:, :, h_pad:h_end, w_pad:w_end])

def customLoss3D(result, target, criterion):
    """
        result: tensor
        target: tensor
    """

    _, _, result_f, result_h, result_w = result.shape
    _, _, target_f, target_h, target_w  = target.shape

    assert(result_h <= target_h)
    assert(result_w <= target_w)
    assert(result_f <= target_f)

    h_pad = (target_h - result_h)//2
    w_pad = (target_w - result_w)//2
    f_pad = (target_f - result_f)//2

    if h_pad%2 or w_pad%2 or f_pad%2:
        assert("must have even padding")

    h_end = -h_pad if h_pad else result_h
    w_end = -w_pad if w_pad else result_w
    f_end = -f_pad if f_pad else result_f

    return criterion(result, target[:, :, f_pad:f_end, h_pad:h_end, w_pad:w_end])