# This example demonstrates the behavior of the sum-reduce primitive.
#
# It requires 6 workers to run.
#
# Run with, e.g.,
#     > mpirun -np 6 python ex_sum_reduce.py

import cupy as cp
import numpy as np
import torch
from mpi4py import MPI

import distdl.utilities.slicing as slicing
from distdl.backends.common.partition import MPIPartition
from distdl.backends.common.tensor_comm import assemble_global_tensor_structure
from distdl.nn.sum_reduce import SumReduce
from distdl.utilities.torch import zero_volume_tensor

# Set up MPI cartesian communicator
P_world = MPIPartition(MPI.COMM_WORLD)
P_world._comm.Barrier()

# On the assumption of 1-to-1 mapping between ranks and GPUs
cp.cuda.runtime.setDevice(P_world.rank % cp.cuda.runtime.getDeviceCount())
P_world.device = cp.cuda.runtime.getDevice()

# device = torch.cuda.device(P_world.rank % torch.cuda.device_count())
# torch.cuda.set_device(device)
# print(f"P_world.rank: {P_world.rank}, P_world.device = {P_world.device}, cur_dev: {torch.cuda.current_device()}")

# Create the input/output partition (using the first 2 workers)
in_shape = (2, 3)
in_size = np.prod(in_shape)
in_workers = np.arange(0, in_size)

P_x_base = P_world.create_partition_inclusive(in_workers)
P_x = P_x_base.create_cartesian_topology_partition(in_shape)

# Create the output partition (using the last 6 workers)
out_shape = (2, 1)
out_size = np.prod(out_shape)
out_workers = np.arange(P_world.size-out_size, P_world.size)

P_y_base = P_world.create_partition_inclusive(out_workers)
P_y = P_y_base.create_cartesian_topology_partition(out_shape)

x_global_shape = np.array([5, 6])

# Setup the input tensor.  Any worker in P_x will generate its part of the
# input tensor.  Any worker not in P_x will have a zero-volume tensor.
#
# Input tensor will be (on a 2 x 3 partition):
# [ [ 1 1 | 2 2 | 3 3 ]
#   [ 1 1 | 2 2 | 3 3 ]
#   [ 1 1 | 2 2 | 3 3 ]
#   -------------------------
#   [ 4 4 | 5 5 | 6 6 ]
#   [ 4 4 | 5 5 | 6 6 ]
#   [ 4 4 | 5 5 | 6 6 ] ]

x = zero_volume_tensor(device=P_x.device)

if P_x.active:
    x_local_shape = slicing.compute_subshape(P_x.shape,
                                             P_x.index,
                                             x_global_shape)
    x = cp.zeros(x_local_shape) + (P_x.rank + 1)
    x = torch.as_tensor(x, device=P_x.device)

x.requires_grad = True
print(f"rank {P_world.rank}; index {P_x.index}; value {x}")

# Here we broadcast the columns (axis 1), along the rows.
all_reduce_cols = SumReduce(P_x, P_y, preserve_batch=False)
#
# Output tensor will be (on a 2 x 1 partition):
# [ [  6  6 ]
#   [  6  6 ]
#   [  6  6 ]
#   -------
#   [ 15 15 ]
#   [ 15 15 ] ]
y = all_reduce_cols(x)

print(f"rank {P_world.rank}; index {P_x.index}; value {y}")
# Setup the adjoint input tensor.  Any worker in P_y will generate its part of
# the adjoint input tensor.  Any worker not in P_y will have a zero-volume
# tensor.

y_global_shape = assemble_global_tensor_structure(y, P_y).shape

# Adjoint input tensor will be (on a 2 x 1 partition):
# [ [ 1 1 ]
#   [ 1 1 ]
#   [ 1 1 ]
#   -------
#   [ 2 2 ]
#   [ 2 2 ] ]

dy = zero_volume_tensor(device=P_y.device)

# print(f"P_y.rank: {P_y.rank}, P_y.device: {P_y.device}, P_x.rank: {P_x.rank}, P_x.device: ",
#       f"{P_x.device}, P_world.rank: {P_world.rank}, P_world.device: {P_world.device}\n")

if P_y.active:
    y_local_shape = slicing.compute_subshape(P_y.shape,
                                             P_y.index,
                                             y_global_shape)
    dy = cp.zeros(y_local_shape) + (P_y.rank + 1)
    dy = torch.as_tensor(dy, device=P_y.device)


print(f"rank {P_world.rank}; index {P_y.index}; value {dy}")

# Apply the adjoint of the layer.
#
# Adjoint output tensor will be (on a 2 x 3 partition):
# [ [ 1 1 | 1 1 | 1 1 ]
#   [ 1 1 | 1 1 | 1 1 ]
#   [ 1 1 | 1 1 | 1 1 ]
#   -------------------------
#   [ 2 2 | 2 2 | 2 2 ]
#   [ 2 2 | 2 2 | 2 2 ]
#   [ 2 2 | 2 2 | 2 2 ] ]
y.backward(dy)
dx = x.grad
print(f"rank {P_world.rank}; index {P_x.index}; value {dx}")
