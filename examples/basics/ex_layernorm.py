import numpy as np
import torch
from mpi4py import MPI
from distdl.config import set_backend

import distdl.utilities.slicing as slicing
from distdl.backends.common.partition import MPIPartition
from distdl.nn.layernorm import DistributedLayerNorm
from distdl.nn.repartition import Repartition
from distdl.utilities.torch import zero_volume_tensor

# Set backend
set_backend(backend_comm="mpi", backend_array="cupy")

# Set up MPI cartesian communicator
P_world = MPIPartition(MPI.COMM_WORLD)
P_world._comm.Barrier()

# Root partition
P_root_base = P_world.create_partition_inclusive([0])
P_root = P_root_base.create_cartesian_topology_partition([1, 1, 1, 1])

# Data partition (in, out)
in_shape = (1, 1, 2, 2)
in_size = np.prod(in_shape)
in_workers = np.arange(0, in_size)

P_x_base = P_world.create_partition_inclusive(in_workers)
P_x = P_x_base.create_cartesian_topology_partition(in_shape)

# Dimensions
batchsize = 2
channel_in = 4
num_features = 32
normalized_shape = (num_features, num_features)

# Layer norm
layer_norm_parallel = DistributedLayerNorm(P_x, normalized_shape, elementwise_affine=True)
layer_norm_serial = torch.nn.LayerNorm(normalized_shape).to(P_x.device)

# Data
x = zero_volume_tensor(device=P_x.device)
scatter_x = Repartition(P_root, P_x)
gather_y = Repartition(P_x, P_root)
if P_root.active:
    x = torch.randn(batchsize, channel_in, num_features, num_features).to(P_x.device)*4.0 + 2.0
    ys = layer_norm_serial(x)
    print('x: ', x.mean(), x.var())
x = scatter_x(x)

# Test
yp = layer_norm_parallel(x)

# Gather results
yp = gather_y(yp)
if P_root.active:
    print('y serial: ', ys.mean(), ys.var())
    print('y parallel: ', yp.mean(), yp.var())
