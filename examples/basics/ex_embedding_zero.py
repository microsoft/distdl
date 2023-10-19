import numpy as np
import torch
from mpi4py import MPI
from distdl.config import set_backend

from distdl.backends.common.partition import MPIPartition
from distdl.nn.repartition import Repartition
from distdl.nn.embedding_zero import DistributedEmbeddingZero

# Set backend
set_backend(backend_comm="mpi", backend_array="numpy")

# Set up MPI cartesian communicator
P_world = MPIPartition(MPI.COMM_WORLD)
P_world._comm.Barrier()

# Data partition corresponding to [ batch, tokens, embedding ]
in_shape = (2, 1, 4)    # [ data-parallel workers, 1, model-parallel workers ]
in_size = np.prod(in_shape)
in_workers = np.arange(0, in_size)

P_root_base = P_world.create_partition_inclusive([0])
P_root = P_root_base.create_cartesian_topology_partition((1, 1, 1))

P_x_base = P_world.create_partition_inclusive(in_workers)
P_x = P_x_base.create_cartesian_topology_partition(in_shape)

# Scatter data
scatter = Repartition(P_root, P_x)

# Embedding layer
batch_size = 4
num_embeddings = 16
embedding_dim = 32

# Input is range of row indices that we want to extract from the embedding matrix
input_idx = torch.arange(num_embeddings, device=P_x.device)

# Create embedding layer
embedding = DistributedEmbeddingZero(P_x, num_embeddings, embedding_dim)

# Forward pass
y = embedding(input_idx)

print("y.shape from rank {}: {}".format(P_x.rank, y.shape))
