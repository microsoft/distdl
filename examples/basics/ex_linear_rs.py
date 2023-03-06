import numpy as np
import torch
from mpi4py import MPI
from distdl.config import set_backend

import distdl.utilities.slicing as slicing
from distdl.backends.common.partition import MPIPartition
from distdl.nn.linear_rs_zero import DistributedLinearReduceScatterZero
from distdl.nn.linear_ag_zero import DistributedLinearAllGatherZero
from distdl.utilities.slicing import compute_subshape
from distdl.utilities.torch import zero_volume_tensor
from distdl.nn.repartition import Repartition

# Set backend
set_backend(backend_comm="nccl", backend_array="cupy")

# Set up MPI cartesian communicator
P_world = MPIPartition(MPI.COMM_WORLD)
P_world._comm.Barrier()

# Data partition (in, out)
in_shape = (1, 1, 1)
in_size = np.prod(in_shape)
in_workers = np.arange(0, in_size)

P_root_base = P_world.create_partition_inclusive([0])
P_root = P_root_base.create_cartesian_topology_partition([1, 1, 1])

P_x_base = P_world.create_partition_inclusive(in_workers)
P_x = P_x_base.create_cartesian_topology_partition(in_shape)

# Input channel dimension is partitioned
batch_size = 4
num_tokens = 16
in_channels = 32
out_channels = 48

# Layer
#network = DistributedLinearAllGatherZero(P_x, in_channels, out_channels, collect_state=True).to(P_x.device)
network = DistributedLinearReduceScatterZero(P_x, in_channels, out_channels, collect_state=True).to(P_x.device)

# Scatter/gather data
scatter_x = Repartition(P_root, P_x, preserve_batch=False)
gather_y = Repartition(P_x, P_root, preserve_batch=False)

#mode = 'training'
mode = 'inference'

if mode == 'training':

    # Data
    x = zero_volume_tensor(device=P_x.device)
    if P_root.active:
        x = torch.randn(batch_size, num_tokens, in_channels).to(P_x.device)
        torch.save(x, 'x.dat')
    x = scatter_x(x)

    s = network.state_dict()
    if P_root.active:
       torch.save(s, 'state.dat')
    print(s.keys())

    # Forward pass
    y = network(x)

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
        network.load_state_dict(s)

    # Forward pass
    y_ = network(x)

    # Gather result and save
    y_ = gather_y(y_)
    if P_root.active:
        print("Error: ", torch.norm(y - y_) / torch.norm(y))