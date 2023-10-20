import numpy as np
import torch
from mpi4py import MPI

from distdl.backends.common.partition import MPIPartition
from distdl.config import set_backend
from distdl.nn.linear_ag_zero import DistributedLinearAllGatherZero
from distdl.nn.linear_rs_zero import DistributedLinearReduceScatterZero
from distdl.nn.repartition import Repartition
from distdl.utilities.torch import zero_volume_tensor

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
num_tokens = 64
in_channels = 128
hidden_channels = int(4*128)
out_channels = 128

# Linear layers with 2D data- and weight partitioning (ZeRO-3, FDSP1)
# The all-gather version is preferred if out_channels > in_channels.
# Otherwise, the reduce-scatter version is preferred.
linear_in = DistributedLinearReduceScatterZero(P_x, in_channels, hidden_channels, checkpoint=True)
linear_out = DistributedLinearAllGatherZero(P_x, hidden_channels, out_channels, checkpoint=True)

# For ZeRO-3, the output channel dimension is folded into the batch dimension, such that
# the weight is partitioned along the data-parallel workers. The input channel dimension
# is partitioned along the model-parallel workers.
print("linear_in.weight.shape from rank {}: {}".format(P_x.rank, linear_in.weight.shape))

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
