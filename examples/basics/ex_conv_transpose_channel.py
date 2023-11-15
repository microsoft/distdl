import numpy as np
import torch
from mpi4py import MPI

import distdl.utilities.slicing as slicing
from distdl.backends.common.partition import MPIPartition
from distdl.config import set_backend
from distdl.nn.conv_channel_ag import DistributedChannelAllGatherConvTranspose2d
from distdl.nn.conv_channel_rs import DistributedChannelReduceScatterConvTranspose2d
from distdl.utilities.torch import zero_volume_tensor

# Set backend
set_backend(backend_comm="mpi", backend_array="numpy")

# Set up MPI cartesian communicator
P_world = MPIPartition(MPI.COMM_WORLD)
P_world._comm.Barrier()

# Data partition
in_shape = (1, 4, 1, 1)  # [ batch, channel, height, width ]
in_size = np.prod(in_shape)
in_workers = np.arange(0, in_size)

P_x_base = P_world.create_partition_inclusive(in_workers)
P_x = P_x_base.create_cartesian_topology_partition(in_shape)

# Input data
x_global_shape = np.array([2, 8, 8, 8])
x = zero_volume_tensor(device=P_x.device)
if P_x.active:
    x_local_shape = slicing.compute_subshape(P_x.shape,
                                             P_x.index,
                                             x_global_shape)
    x = torch.zeros(*x_local_shape, device=x.device) + (P_x.rank + 1)
x.requires_grad = True

# Distributed conv layer
if P_x.rank == 0:
    print("Forward")

# Distributed conv transpose layer: The reduce-scatter version is preferrable to the all-gather version when the
# number of output channels is smaller than the number of input channels.
conv2d_in = DistributedChannelAllGatherConvTranspose2d(P_x, 8, 16, (2, 2), stride=(2, 2))   # upsampling effect
conv2d_out = DistributedChannelReduceScatterConvTranspose2d(P_x, 16, 4, (2, 2), stride=(2, 2))

# Apply conv layers
y = conv2d_in(x)
y = conv2d_out(y)

# Backward pass
if P_x.rank == 0:
    print("Backward")
y.sum().backward()

print("x.shape: {}, y.shape: {} from rank {}".format(x.shape, y.shape, P_x.rank))
