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
from distdl.nn.repartition import Repartition

# Set backend
set_backend(backend_comm="mpi", backend_array="numpy")

# Set up MPI cartesian communicator
P_world = MPIPartition(MPI.COMM_WORLD)
P_world._comm.Barrier()

# Data partition corresponding to [ batch, ..., channel/embedding ]
in_shape = (4, 1, 2)    # [ data-parallel workers, ..., model-parallel workers ]
in_size = np.prod(in_shape)
in_workers = np.arange(0, in_size)

P_root_base = P_world.create_partition_inclusive([0])
P_root = P_root_base.create_cartesian_topology_partition([1, 1, 1])

P_x_base = P_world.create_partition_inclusive(in_workers)
P_x = P_x_base.create_cartesian_topology_partition(in_shape)

# Input/output data dimensions
batch_size = 8
num_tokens = 16
in_channels = 24
hidden_channels = 36
out_channels = 28

# Linear layers: The all-gather version is preferred if out_channels > in_channels.
# Otherwise, the reduce-scatter version is preferred.
linear_in = DistributedLinearAllGather(P_x, in_channels, hidden_channels)
linear_out = DistributedLinearReduceScatter(P_x, hidden_channels, out_channels)

# Scatter/gather data
scatter = Repartition(P_root, P_x, preserve_batch=False)

# Data
x = zero_volume_tensor(device=P_x.device)

# Initialize on root only
if P_root.active:
    x = torch.randn(batch_size, num_tokens, in_channels).to(P_x.device)

# Scatter data to workers
x = scatter(x)

# Forward pass
y = linear_in(x)
y = torch.nn.functional.relu(y)
y = linear_out(y)

print("y.shape from rank {}: {}".format(P_x.rank, y.shape))