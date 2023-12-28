import numpy as np
import torch
from mpi4py import MPI

from distdl.backends.common.partition import MPIPartition
from distdl.config import set_backend
from distdl.nn.linear_ag_expert import DistributedExpertAllGather
from distdl.nn.linear_rs_expert import DistributedExpertReduceScatter
from distdl.nn.repartition import Repartition
from distdl.utilities.torch import zero_volume_tensor

# Set backend
set_backend(backend_comm="mpi", backend_array="numpy")

# Set up MPI cartesian communicator
P_world = MPIPartition(MPI.COMM_WORLD)
P_world._comm.Barrier()

# Data partition corresponding to [ experts, capacity, embedding ]
emb_shape = (2, 2, 1, 2)    # [ data/expert dimension, capacity dimension, 1, embedding dimension ]
seq_shape = (2, 2, 2, 1)    # [ data/expert dimension, capacity dimension, sequence dimension, 1 ]
in_size = np.prod(emb_shape)
in_workers = np.arange(0, in_size)

P_root_base = P_world.create_partition_inclusive([0])
P_root = P_root_base.create_cartesian_topology_partition([1, 1, 1, 1])

P_x_base = P_world.create_partition_inclusive(in_workers)
P_emb = P_x_base.create_cartesian_topology_partition(emb_shape)
P_seq = P_x_base.create_cartesian_topology_partition(seq_shape)

# Data dimensions
num_experts = 2
num_tokens = 16
in_channels = 24
hidden_channels = 36
out_channels = 28

# Distributed linear layer with an additional expert dimension for weights and biases. These layers assume that
# inputs/outputs are 3D tensors of shape [ experts, capacity, embedding/channel ]. The all-gather version is
# preferrable if out_channels > in_channels. Otherwise, the reduce-scatter version is preferred.
experts_in = DistributedExpertAllGather(P_emb, num_experts, in_channels, hidden_channels,
                                        collect_state=True, P_expert_seq=None).to(P_emb.device)
experts_out = DistributedExpertReduceScatter(P_emb, num_experts, hidden_channels, out_channels,
                                             collect_state=True, P_expert_seq=None).to(P_emb.device)

# Scatter data
scatter = Repartition(P_root, P_emb, preserve_batch=False)

# Data
x = zero_volume_tensor(device=P_emb.device)

# Initialize on root only
if P_root.active:
    x = torch.randn(num_experts, num_tokens, 1, in_channels).to(P_emb.device)

# Scatter data to workers
x = scatter(x)
print("x.shape from rank {}: {}".format(P_emb.rank, x.shape))

# Forward pass
y = experts_in(x)
y = torch.nn.functional.relu(y)
print("h.shape from rank {}: {}".format(P_emb.rank, y.shape))
y = experts_out(y)

print("y.shape from rank {}: {}".format(P_emb.rank, y.shape))
