import numpy as np
import torch
from mpi4py import MPI
from distdl.config import set_backend

import distdl.utilities.slicing as slicing
from distdl.backends.common.partition import MPIPartition
from distdl.nn.upsampling import DistributedUpsample
from distdl.utilities.torch import zero_volume_tensor

# Set backend. The GPU backend is currently supported. For upsampling
# on GPUs, use the distdl.nn.ConvTranspose layer.
set_backend(backend_comm="mpi", backend_array="numpy")

# Set up MPI cartesian communicator
P_world = MPIPartition(MPI.COMM_WORLD)
P_world._comm.Barrier()

# Data partition
in_shape = (1, 1, 2, 2)     # [ batch, channel, height, width ]
in_size = np.prod(in_shape)
in_workers = np.arange(0, in_size)

P_x_base = P_world.create_partition_inclusive(in_workers)
P_x = P_x_base.create_cartesian_topology_partition(in_shape)

# Input data
x_global_shape = np.array([4, 8, 16, 16])

# Initialize x locally on each worker for its local shape
x = zero_volume_tensor(device=P_x.device)
if P_x.active:
    x_local_shape = slicing.compute_subshape(P_x.shape,
                                             P_x.index,
                                             x_global_shape)
    x = torch.zeros(*x_local_shape, device=x.device) + (P_x.rank + 1)

# Distributed maxpool layer
upsample = DistributedUpsample(P_x, scale_factor=2, mode='linear')
y = upsample(x)

print("x.shape {} and y.shape {} from rank {}".format(x.shape, y.shape, P_x.rank))