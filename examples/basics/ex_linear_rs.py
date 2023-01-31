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
in_channels = 48
out_channels = 72
linear = DistributedLinearReduceScatter(P_x, in_channels, out_channels, checkpointing=True).to(P_x.device)

# Input
num_features = 64
in_channels_local = compute_subshape(P_x.shape[1], P_x.index[1], [in_channels])[0]
x = torch.randn(num_features, in_channels_local).to(P_x.device)

# Parallel GEMM
print('x.shape: ', x.shape)

y = linear(x)
print('y.shape: ', y.shape)