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
    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=(3, 3), mode='bicubic')

        self.conv1 = nn.Conv3d(3, 64, (9, 9), padding=(0, 0))
        self.conv2 = nn.Conv3d(64, 32, (1, 1), padding=(0, 0))
        self.conv3 = nn.Conv3d(32, 3, (5, 5), padding=(0, 0))

    def forward(self, x):
        x = self.upsample(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x) # no ReLU on the third? apparently
        return x

if __name__ == "__main__":
    net = Net_2016()
    print(net)
    net.train()