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

if __name__ == "__main__":
    net = Net()
    print(net)
    net.train()