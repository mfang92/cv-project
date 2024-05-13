import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 3, (1, 3, 3), padding=(0, 1, 1), device=device)
        self.conv2 = nn.Conv3d(3, 3, (3, 1, 1), padding=(1, 0, 0), device=device)

        self.conv3 = nn.Conv3d(3, 3, (1, 3, 3), padding=(0, 1, 1), device=device)
        self.conv4 = nn.Conv3d(3, 3, (3, 1, 1), padding=(1, 0, 0), device=device)

        self.conv5 = nn.Conv3d(3, 3, (1, 3, 3), padding=(0, 1, 1), device=device)
        self.conv6 = nn.Conv3d(3, 3, (3, 1, 1), padding=(1, 0, 0), device=device)

        self.conv7 = nn.Conv3d(3, 3, (1, 3, 3), padding=(0, 1, 1), device=device)
        self.conv8 = nn.Conv3d(3, 3, (3, 1, 1), padding=(1, 0, 0), device=device)

        self.convs = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7, self.conv8, lambda x: x]

        self.upsample = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear')

    def forward(self, x):
        # identity = x.clone()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.upsample(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))

        return x

class VaryNets(Net):
    def __init__(self, device=None, placement = 4, res_net = False):
        super().__init__(device=device)
        self.placement = placement
        self.convs = self.convs[:placement] + [self.convs[8]] + self.convs[placement:8]
        self.res_net = res_net
        print(self.convs)
    
    def forward(self, x):
        identity = x.clone()

        for i in range(9):
            if i != self.placement:
                x = F.relu(self.convs[i](x))
            else:
                x = self.upsample(x)

        return x + self.upsample(identity) if self.res_net else x

if __name__ == "__main__":
    # net = Net()
    # print(net.summary())
    for i in range(9):
        net = VaryNets(placement = i)