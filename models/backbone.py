import torch
import torch.nn as nn
import math
import torch.nn.functional as F


# --- gaussian initialize ---
def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0] * L.kernel_size[1] * L.out_channels
        L.weight.data.normal_(0,math.sqrt(2)/float(n))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)


# --- Simple Conv Block ---
class ConvBlock(nn.Module):
    maml = False
    def __init__(self, indim, outdim, pool=True, padding=1):
        super(ConvBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            pass #TODO
        else:
            self.C = nn.Conv2d(indim,outdim,3,padding=padding)
            self.BN = nn.BatchNorm2d(outdim)
        self.relu = nn.ReLU(inplace=True)

        self.parameterized_layers = [self.C, self.BN, self.relu]

        if pool:
            self.pool = nn.MaxPool2d(2)
            self.parameterized_layers.append(self.pool)
        for layer in self.parameterized_layers:
            init_layer(layer)

        self.trunk = nn.Sequential(*self.parameterized_layers)

    def forward(self, x):
        out = self.trunk(x)
        return out




# ----ConvNetNopool modeule----
class ConvNetNopool(nn.Module):
    def __init__(self, depth):
        super(ConvNetNopool, self).__init__()
        trunk = []
        for i in range(depth):
            indim = 3 if i==0 else 64
            outdim=64
            B = ConvBlock(indim, outdim, pool=(i in [0,1]),
                          padding=0 if i in [0,1] else 1) # only first two layer has pooling and no padding
            trunk.append(B)

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = [64, 19, 19]

    def forward(self, x):
        out = self.trunk(x)
        return out

def Conv4NP():
    return ConvNetNopool(4)

def main():
    model = ConvNetNopool(6)
    x = torch.randn((10,3,84,84))
    y = model(x)
    print(y.shape) #torch.Size([10, 64, 19, 19])
if __name__ == '__main__':
    main()

