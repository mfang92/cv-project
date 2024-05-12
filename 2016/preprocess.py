"""
gaussian blur
take every other third pixel

33 x 33 sub-images


"""

import os
import os.path as osp
import cv2
import itertools as it
import numpy as np
import random
from torchvision import transforms, io
import torch.nn as nn
import torch
import torchvision
from scipy.ndimage import gaussian_filter

random.seed(0)

"""
Get sub-images from big images.

1. blur with gaussian kernel
2. cut into patches -> new 'dataset'
"""

def apply_blur(images, radius, sigma):
    # torch implementation:
    # blur_fn = transforms.GaussianBlur(kernel_size, sigma = (sigma, sigma))
    # return list(map(blur_fn, images))

    return [gaussian_filter(image, sigma = sigma, radius=radius,) for image in images]

def save_patches(dest, image_names, images, f_sub, upscale_factor):
    """
        images: list of tensors of dimension (3, h, w) 
        f_sub: expected dimension of square sub-images
        upscale_factor: amount to downsize the HR image/ upscale the 
            sub-sampled images by

        requires: upscale_factor must evenly divide f_sub

        Saves list of patches, size (3, f_sub, f_sub) tensors,
        formed from patches of images in 'images'. 
    """
    assert f_sub%upscale_factor == 0, "upscale_factor should divide final sub-image length, f_sub"

    for fn, image in zip(image_names, images):
        height, width, depth = image.shape
        for i in range(0, height//f_sub):
            for j in range(0, width//f_sub):
                patch = image[i*f_sub:(i+1)*f_sub, j*f_sub:(j+1)*f_sub, :]
                # temp = temp.to(torch.int) # this has to happen for color to show on pyplot

                base, extension = osp.splitext(fn)

                name = f"{base}_{i}_{j}_{extension}"
                cv2.imwrite(osp.join(dest, name), patch)

def process_imgs(source, destination, f_sub, upscale_factor):
    """
        Returns the sub-images to be used for training and 
    """
    
    # obtains array of images (tensor)
    img_names = os.listdir(source)
    images = [cv2.imread(osp.join(source, name)) 
              for name in img_names]

    # # gaussian blur
    blurred_images = apply_blur(images, radius=(4, 4, 0), sigma=2) # don't want to blur along color axis

    # # save patches
    save_patches(destination, img_names, blurred_images, f_sub, upscale_factor)


if __name__ == "__main__":
    root = os.getcwd()
    print(root)

    source = "2016/data/set14"
    destination = "2016/data/patches_train"
    f_sub = 33
    upscale_factor = 3

    source = osp.join(root, source)
    destination = osp.join(root, destination)
    process_imgs(source, destination, f_sub, upscale_factor)


    # one-time work:
    # img_names = os.listdir(source)
    # images = [cv2.imread(osp.join(source, name)) 
    #           for name in img_names]
    
    # img = images[5]
    # img = img[:33, :33, :]
    # print(type(img), img.shape)
    # cv2.imwrite(osp.join(destination, "0.png"), img)

