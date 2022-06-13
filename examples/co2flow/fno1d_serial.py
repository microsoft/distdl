import numpy as np
import torch, julius
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import operator, math
from functools import reduce
from functools import partial
from timeit import default_timer

torch.manual_seed(0)
np.random.seed(0)


################################################################
#  1D FNO
################################################################

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv1d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = (1 / (in_channels*out_channels))
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes] = self.compl_mul1d(x_ft[:, :, :self.modes], self.weights)

        #Return to physical space
        x = torch.fft.irfft(out_ft)
        return x


class FNO1d(nn.Module):
    def __init__(self, nc_in, nc_out, modes, width, npad=4):
        super(FNO1d, self).__init__()

        self.modes = modes
        self.width = width
        self.padding = npad
        self.fc0 = nn.Conv1d(nc_in, self.width, 1)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Conv1d(self.width, 128, 1)
        self.fc2 = nn.Conv1d(128, nc_out, 1)

    def forward(self, x):   # N C X
    
        x = self.fc0(x)
        if self.padding > 0:
            x = F.pad(x, [self.padding, self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        if self.padding > 0:
            x = x[..., self.padding: -self.padding]
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


###################################################################################################

if __name__ == '__main__':

    import torch

    # Reshape
    nx = 64
    nb = 4
    nc_in = 1
    nc_out = 1
    nkx = nx // 8
    width = 16

    # N C X
    X = torch.randn(nb, nc_in, nx).to('cpu')

    # FNO
    fno = FNO1d(nc_in, nc_out, nkx, width, npad=8).to('cpu')

    # Test
    Y_ = fno(X)
    print("Y.shape: ", Y_.shape)