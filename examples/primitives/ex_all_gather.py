# This example demonstrates the behavior of the all-gather primitive.
#
# It requires 6 workers to run.
#
# Run with, e.g.,
#     > mpirun -np 6 python ex_all_gather.py

import numpy as np
import torch
from mpi4py import MPI

import distdl.utilities.slicing as slicing
from distdl.backends.common.partition import MPIPartition
from distdl.config import set_backend
from distdl.nn.all_gather import AllGather
from distdl.utilities.torch import zero_volume_tensor

# Set backend
set_backend(backend_comm="mpi", backend_array="numpy")

# Set up MPI cartesian communicator
P_world = MPIPartition(MPI.COMM_WORLD)
P_world._comm.Barrier()

# On the assumption of 1-to-1 mapping between ranks and GPUs
# torch.cuda.set_device(P_world.rank % torch.cuda.device_count())
# P_world.device = torch.cuda.current_device()
# P_world.device = torch.device("cpu")

# Create the input/output partition (using the first worker)
in_shape = (2, 3)
in_size = np.prod(in_shape)
in_workers = np.arange(0, in_size)

P_x_base = P_world.create_partition_inclusive(in_workers)
P_x = P_x_base.create_cartesian_topology_partition(in_shape)

# The all-gather layer operates along partitioned dimensions, not tensor
# dimensions.  Thus, along the dimensions that the reduction applies, the
# subtensors all must be the same size.  Thus, this global shape is evenly
# divisible by the partition.  Later we will have an example for applying the
# reduction on the tensor itself.
x_global_shape = np.array([4, 3])

# Setup the input tensor.  Any worker in P_x will generate its part of the
# input tensor.  Any worker not in P_x will have a zero-volume tensor.
#
# Input tensor will be (on a 2 x 3 partition):
# [ [ 1 | 2 | 3 ]
#   [ 1 | 2 | 3 ]
#   ----------------------------
#   [ 4 | 5 | 6 ]
#   [ 4 | 5 | 6 ] ]

x = zero_volume_tensor(device=P_x.device)

if P_x.active:
    x_local_shape = slicing.compute_subshape(P_x.shape,
                                             P_x.index,
                                             x_global_shape)
    x = torch.zeros(*x_local_shape, device=x.device) + (P_x.rank + 1)

x.requires_grad = True

print(f"P_world.rank {P_world.rank}; P_x.index {P_x.index}; x value: \n{x}\n")

# Create the all-gather layer.  Note, only one of the keep/reduce axes is
# required.  If they are both specified they must be mutually coherent.
# Commented out declarations are equivalent.  `axes_all_gather` is equivalent
# to PyTorch's dimension argument in torch.sum().
#
# Here we all-gather the columns (axis 1), along the rows.
all_gather_cols = AllGather(P_x, axes_all_gather=(1,))

# Output tensor will be (on a 2 x 3 partition):
# [ [ 1 2 3 | 1 2 3 | 1 2 3 ]
#   [ 1 2 3 | 1 2 3 | 1 2 3 ]
#   -------------------------
#   [ 4 5 6 | 4 5 6 | 4 5 6 ]
#   [ 4 5 6 | 4 5 6 | 4 5 6 ] ]
y = all_gather_cols(x)

print(f"P_world.rank {P_world.rank}; P_x.index {P_x.index}; y value: \n{y}\n")

# Setup the adjoint input tensor.  Any worker in P_y will generate its part of
# the adjoint input tensor.  Any worker not in P_y will have a zero-volume
# tensor.
#
# Adjoint input tensor will be (on a 2 x 3 partition):
# [ [ 1 1 1 | 2 2 2 | 3 3 3 ]
#   [ 1 1 1 | 2 2 2| 3 3 3 ]
#   -------------------------
#   [ 4 4 4 | 5 5 5 | 6 6 6 ]
#   [ 4 4 4 | 5 5 5 | 6 6 6] ]
dy = zero_volume_tensor(device=P_x.device)
if P_x.active:
    dy = torch.zeros(*y.shape, device=x.device) + (P_x.rank + 1)

print(f"P_world.rank {P_world.rank}; P_x.index {P_x.index}; dy value: \n{dy}\n")

# Apply the adjoint of the layer.
#
# Adjoint output tensor will be (on a 2 x 2 partition):
# [ [  6 |  6 |  6 ]
#   [  6 |  6 |  6 ]
#   ----------------------------
#   [ 15 | 15 | 15 ]
#   [ 15 | 15 | 15] ]
y.backward(dy)
dx = x.grad

print(f"P_world.rank {P_world.rank}; P_x.index {P_x.index}; dx value: \n{dx}\n")
