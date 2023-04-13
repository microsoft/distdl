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
set_backend(backend_comm="nccl", backend_array="cupy")

# Set up MPI cartesian communicator
P_world = MPIPartition(MPI.COMM_WORLD)
P_world._comm.Barrier()

# Root partition
P_root_base = P_world.create_partition_inclusive([0])
P_root = P_root_base.create_cartesian_topology_partition([1, 1, 1])

# Data partition (in, out)
in_shape = (1, 1, 1)
in_size = np.prod(in_shape)
in_workers = np.arange(0, in_size)

P_x_base = P_world.create_partition_inclusive(in_workers)
P_x = P_x_base.create_cartesian_topology_partition(in_shape)

# Dimensions
batchsize = 12
num_tokens = 8
num_features = 32
normalized_shape = (num_features)

# Layer norm
layer_norm = DistributedLayerNorm(P_x, normalized_shape, elementwise_affine=True, collect_state=True)

# Scatter/gather data
scatter_x = Repartition(P_root, P_x, preserve_batch=False)
gather_y = Repartition(P_x, P_root, preserve_batch=False)

#mode = 'training'
mode = 'inference'

if mode == 'training':

    # Data
    x = zero_volume_tensor(device=P_x.device)
    if P_root.active:
        x = torch.randn(batchsize, num_tokens, num_features).to(P_x.device)*4.0 + 2.0
        torch.save(x, 'x.dat')
    x = scatter_x(x)

    s = layer_norm.state_dict()
    if P_root.active:
       torch.save(s, 'state.dat')
    print(s.keys())

    # Forward pass
    y = layer_norm(x)

    # Gather result and save
    y = gather_y(y)
    if P_root.active:
        torch.save(y, 'y.dat')

elif mode == 'inference':

    # Load data
    x = zero_volume_tensor(device=P_x.device)
    y = zero_volume_tensor(device=P_x.device)
    if P_root.active:
        x = torch.load('x.dat')
        y = torch.load('y.dat')
    x = scatter_x(x)

    # Load state dict
    s = torch.load('state.dat')
    if P_x.active:
        layer_norm.load_state_dict(s)

    # Forward pass
    y_ = layer_norm(x)

    # Gather result and save
    y_ = gather_y(y_)
    if P_root.active:
        print("Error: ", torch.norm(y - y_) / torch.norm(y))