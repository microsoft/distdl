import numpy as np
import torch
from mpi4py import MPI
from distdl.config import set_backend

import distdl.utilities.slicing as slicing
from distdl.backends.common.partition import MPIPartition
from distdl.nn.linear_rs import DistributedLinearReduceScatter
from distdl.nn.linear_ag import DistributedLinearAllGather
from distdl.nn.repartition import Repartition
from distdl.nn.embedding import DistributedEmbedding
from distdl.utilities.slicing import compute_subshape
from distdl.utilities.torch import zero_volume_tensor

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
P_root = P_root_base.create_cartesian_topology_partition((1, 1, 1))

P_x_base = P_world.create_partition_inclusive(in_workers)
P_x = P_x_base.create_cartesian_topology_partition(in_shape)

# Embedding layer
batch_size = 4
num_embeddings = 16
embedding_dim = 32
embedding = DistributedEmbedding(P_x, num_embeddings, embedding_dim)

x = embedding(torch.arange(num_embeddings, device=P_x.device))
print(x.shape)