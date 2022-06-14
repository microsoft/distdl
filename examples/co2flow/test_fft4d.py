import numpy as np, time
import torch, distdl, math
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




class RFFT4D(distdl.nn.Module):

    def __init__(self, P_in, P_out):
        super(RFFT4D, self).__init__()

        self.transpose = Repartition(P_in, P_out)
        self.fft = torch.fft.fftn
        self.rfft = torch.fft.rfftn

    def forward(self, x):
        x = self.rfft(x, dim=(2,4,5))
        x = self.transpose(x)
        x = self.fft(x, dim=(3))
        return x




#######################################################################################################################

if __name__ == '__main__':

    # Init MPI
    P_world = MPIPartition(MPI.COMM_WORLD)
    P_world._comm.Barrier()
    n = P_world.shape[0]

    # Master worker partition with 6 dimensions ( N C X Y Z T )
    root_shape = (1, 1, 1, 1, 1, 1)
    P_root_base = P_world.create_partition_inclusive([0])
    P_root = P_root_base.create_cartesian_topology_partition(root_shape)

    # Distributed paritions
    feat_workers = np.arange(0, n)
    P_x_base = P_world.create_partition_inclusive(feat_workers)
    P_x = P_x_base.create_cartesian_topology_partition((1,1,1,n,1,1))
    P_y_base = P_world.create_partition_inclusive(feat_workers)
    P_y = P_y_base.create_cartesian_topology_partition((1,1,n,1,1,1))

    # Cuda
    device = torch.device(f'cuda:{P_x.rank}')

    # Data dimensions
    nb = 1
    shape = (64, 64, 64, 32)    # X Y Z T
    num_channel = 16
    num_kx = 12
    num_ky = 12
    num_kz = 12
    num_kw = 12

    # Scatter data
    scatter = Repartition(P_root, P_x).to('cuda')
    gather = Repartition(P_x, P_root).to('cuda')

    # FFT
    fft = RFFT4D(P_x, P_y)

    # Data
    x = zero_volume_tensor()
    if P_root.active:
        x = torch.randn(nb, num_channel, *shape, dtype=torch.float32).to('cuda')
    x = scatter(x).to(device)

    # Apply fft
    t0 = time.time()
    y = fft(x, num_kx, num_ky, num_kz, num_kw)
    t1 = time.time()
    print("Time: ", t1 - t0)