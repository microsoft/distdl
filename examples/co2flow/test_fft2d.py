import numpy as np
import torch, distdl, math, time
from mpi4py import MPI
from distdl.backends.mpi.partition import MPIPartition
from distdl.nn.repartition import Repartition
from distdl.nn.broadcast import Broadcast
from distdl.utilities.torch import zero_volume_tensor
from distdl.nn import DistributedTranspose
from distdl.utilities.slicing import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
from distdl.functional import ZeroVolumeCorrectorFunction




class RFFT2D(distdl.nn.Module):

    def __init__(self, P_in, P_out):
        super(RFFT2D, self).__init__()

        self.transpose = Repartition(P_in, P_out)
        self.fft = torch.fft.fftn
        self.rfft = torch.fft.rfftn

    def forward(self, x):
        x = self.rfft(x, dim=(3))
        x = self.transpose(x)   # communicate
        x = self.fft(x, dim=(2))
        return x




#######################################################################################################################

if __name__ == '__main__':

    # Init MPI
    P_world = MPIPartition(MPI.COMM_WORLD)
    P_world._comm.Barrier()
    n = P_world.shape[0]

    # Master worker partition with 4 dimensions ( N C X Y )
    root_shape = (1, 1, 1, 1)
    P_root_base = P_world.create_partition_inclusive([0])
    P_root = P_root_base.create_cartesian_topology_partition(root_shape)

    # Distributed paritions
    feat_workers = np.arange(0, n)
    P_x_base = P_world.create_partition_inclusive(feat_workers)
    P_x = P_x_base.create_cartesian_topology_partition((1,1,n,1))   # N C X/n Y
    P_y_base = P_world.create_partition_inclusive(feat_workers)
    P_y = P_y_base.create_cartesian_topology_partition((1,1,1,n))   # N C X Y/n

    # Cuda
    device = torch.device(f'cuda:{P_x.rank}')

    # Data dimensions
    nb = 1
    shape = (64, 64)    # X Y Z T
    num_channel = 16

    # Scatter data
    scatter = Repartition(P_root, P_x)
    gather = Repartition(P_x, P_root)

    # FFT
    fft = RFFT2D(P_x, P_y, num_kx, num_ky)

    # Data
    x = zero_volume_tensor()
    if P_root.active:
        x = torch.randn(nb, num_channel, *shape, dtype=torch.float32)
    x = scatter(x).to(device)

    # Apply fft
    t0 = time.time()
    y = fft(x)
    t1 = time.time()
    print("t = ", t1 - t0)