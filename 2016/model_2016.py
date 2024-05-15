"""
https://arxiv.org/pdf/1501.00092

original for single image super resolution:

CNN architecture:
n1 filters of size c x f1 x f1 -> f2
n2 filters of size n1 x f2 x f2 -> f3
c filters of size n2 x f3 x f3


A typical and basic setting is 
f1 = 9, f2 = 1, f3 = 5,
n1 = 64, and n2 = 32 

Trains faster: 9-1-5 
Apparently does best: 9-5-5
Does almost same as 9-5-5, but half number of parameters: 9-3-5

"To avoid border effects during training, all the convolutional layers have no padding"

"""

import torch.nn as nn
import torch.nn.functional as F

class Net_2016(nn.Module):
    def __init__(self, device = None):
        super().__init__()
        # factor = 3 # 2D
        factor = 2 # video
        self.upsample = nn.Upsample(scale_factor=(factor, factor), mode='bicubic')
        self.conv1 = nn.Conv2d(3, 64, (9, 9), padding=(0, 0), device=device) # TODO: change padding
        self.conv2 = nn.Conv2d(64, 32, (3, 3), padding=(0, 0), device=device)
        self.conv3 = nn.Conv2d(32, 3, (5, 5), padding=(0, 0), device=device)

    def takePadding(self, og, x):
        identity = og.clone().detach()
        identity = self.upsample(identity)
        
        _, _, og_h, og_w  = identity.shape
        _, _, result_h, result_w = x.shape

        assert(result_h <= og_h)
        assert(result_w <= og_w)

        h_pad = (og_h - result_h)//2
        w_pad = (og_w - result_w)//2

        if h_pad%2 or w_pad%2:
            assert("must have even padding")

        h_end = -h_pad if h_pad else result_h
        w_end = -w_pad if w_pad else result_w
        identity = identity[:, :, h_pad:h_end, w_pad:w_end]

        return identity, x

    def forward(self, x):
        identity = x.clone().detach() # ReLU
        x = F.relu(self.conv1(x))
        x = self.upsample(x)
        x = F.relu(self.conv2(x))
        x = self.conv3(x) # no ReLU on the third? apparently

        # padding for ReLU
        identity, x = self.takePadding(identity, x)
        return identity + x # residual
    #   return x

if __name__ == "__main__":
    net = Net_2016()
    print(net)
    net.train()