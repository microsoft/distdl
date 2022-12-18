import numpy as np
import torch
from mpi4py import MPI
from distdl.config import set_backend

import distdl.utilities.slicing as slicing
from distdl.backends.common.partition import MPIPartition
from distdl.nn.conv_feature_rs import DistributedFeatureReduceScatterConv2d
from distdl.utilities.torch import zero_volume_tensor
from unicron_network import Unicron

# Set backend
set_backend(backend_comm="mpi", backend_array="cupy")

# Set up MPI cartesian communicator
P_world = MPIPartition(MPI.COMM_WORLD)
P_world._comm.Barrier()

# Root partition
root_shape = (1, 1, 1, 1, 1)
root_size = np.prod(root_shape)
root_workers = np.arange(0, root_size)

P_root_base = P_world.create_partition_inclusive(root_workers)
P_root = P_root_base.create_cartesian_topology_partition(root_shape)

# Data partition for unet
p_unet_shape = (1, 8, 1, 1, 1)
p_unet_size = np.prod(p_unet_shape)
p_unet_workers = np.arange(0, p_unet_size)

P_x_base = P_world.create_partition_inclusive(p_unet_workers)
P_x = P_x_base.create_cartesian_topology_partition(p_unet_shape)

# Input data
batch_size = 2
in_channels = 3
features = 256
x = zero_volume_tensor(device=P_x.device)
if P_root.active:
    x = torch.randn(batch_size, in_channels, features, features, features, device=P_x.device)

# Unicron network
levels = 5
base_channels = 32
out_channels = 1
network = Unicron(P_root, P_x, levels, in_channels, base_channels, out_channels, checkpointing=True).to(P_x.device)

y = network(x)
print('y.shape: ', y.shape)
y.sum().backward()