import numpy as np
import torch
from mpi4py import MPI

import distdl.utilities.slicing as slicing
from distdl.backends.common.partition import MPIPartition
from distdl.config import set_backend
from distdl.nn.conv_channel_ag import DistributedChannelAllGatherConv2d
from distdl.nn.conv_channel_rs import DistributedChannelReduceScatterConv2d
from distdl.utilities.torch import zero_volume_tensor

# Set backend
set_backend(backend_comm="mpi", backend_array="numpy")

# Set up MPI cartesian communicator
P_world = MPIPartition(MPI.COMM_WORLD)
P_world._comm.Barrier()

# Data partition
in_shape = (2, 4, 1, 1)  # [ batch, channel, height, width ]
in_size = np.prod(in_shape)
in_workers = np.arange(0, in_size)

P_x_base = P_world.create_partition_inclusive(in_workers)
P_x = P_x_base.create_cartesian_topology_partition(in_shape)

# Input data
x_global_shape = np.array([2, 16, 16, 16])
x = zero_volume_tensor(device=P_x.device)
if P_x.active:
    x_local_shape = slicing.compute_subshape(P_x.shape,
                                             P_x.index,
                                             x_global_shape)
    x = torch.zeros(*x_local_shape, device=x.device) + (P_x.rank + 1)
x.requires_grad = True

# Distributed conv layers: The all-gather version is preferrable to the reduce-scatter version when the
# number of input channels is smaller than the number of output channels.
conv2d_in = DistributedChannelAllGatherConv2d(P_x, 16, 24, (3, 3), padding=(1, 1))
conv2d_out = DistributedChannelReduceScatterConv2d(P_x, 24, 8, (3, 3), padding=(1, 1))

# Forward pass
if P_x.rank == 0:
    print("Forward")
y = conv2d_in(x)
y = conv2d_out(y)

# Backward pass
if P_x.rank == 0:
    print("Backward")
y.sum().backward()

print("y.shape from rank {}: {}".format(P_x.rank, y.shape))
