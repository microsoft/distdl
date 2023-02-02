import numpy as np
import torch
from mpi4py import MPI
from distdl.config import set_backend

import distdl.utilities.slicing as slicing
from distdl.backends.common.partition import MPIPartition
from distdl.nn.linear_rs import DistributedLinearReduceScatter
from distdl.nn.linear_ag import DistributedLinearAllGather
from distdl.utilities.slicing import compute_subshape
from distdl.utilities.torch import zero_volume_tensor

# Set backend
set_backend(backend_comm="nccl", backend_array="cupy")

# Set up MPI cartesian communicator
P_world = MPIPartition(MPI.COMM_WORLD)
P_world._comm.Barrier()

# Data partition (in, out)
in_shape = (1, 4)
in_size = np.prod(in_shape)
in_workers = np.arange(0, in_size)

P_x_base = P_world.create_partition_inclusive(in_workers)
P_x = P_x_base.create_cartesian_topology_partition(in_shape)

# Input channel dimension is partitioned
batch_size = 4
in_channels = 32
out_channels = 32

# Input
n = 128
x_global_shape = (n, in_channels)
x = zero_volume_tensor(device=P_x.device)
if P_x.active:
    x_local_shape = slicing.compute_subshape(P_x.shape,
                                             P_x.index,
                                             x_global_shape)
    x = torch.zeros(*x_local_shape, device=x.device) + (P_x.rank + 1)
x.requires_grad = True

# Parallel GEMM
linear1 = DistributedLinearReduceScatter(P_x, in_channels, out_channels).to(P_x.device)
linear2 = DistributedLinearReduceScatter(P_x, in_channels, out_channels, 
    P_weight=linear1.P_weight, 
    P_store_bias=linear1.P_store_bias,
    P_apply_bias=linear1.P_apply_bias
    ).to(P_x.device)

y = linear1(x)
z = linear2(y)
print('z.shape: ', z.shape)