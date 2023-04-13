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
P_root = P_root_base.create_cartesian_topology_partition((1, 1, 1))

P_x_base = P_world.create_partition_inclusive(in_workers)
P_x = P_x_base.create_cartesian_topology_partition(in_shape)

# Scatter data
scatter = Repartition(P_root, P_x)

# Embedding layer
batch_size = 4
num_embeddings = 16
embedding_dim = 32
input_idx = torch.arange(num_embeddings, device=P_x.device)

#mode = 'training'
mode = 'inference'

if mode == 'training':

    # Create embedding
    embedding = DistributedEmbedding(P_x, num_embeddings, embedding_dim, collect_state=True)
    
    # Store state dict
    s = embedding.state_dict()
    if P_root.active:   
        torch.save(s, 's.dat')
        print(s.keys())

    # Forward pass
    y = embedding(input_idx)

    # Collect and save output
    gather = Repartition(P_x, P_root)
    y = gather(y.view(1, *y.shape))

    # Save output
    if P_root.active:
        torch.save(y, 'y.dat')

elif mode == 'inference':

    # Load input/output
    y = zero_volume_tensor(device=P_x.device)
    if P_root.active:
        y = torch.load('y.dat')

    # Load state dict
    embedding = DistributedEmbedding(P_x, num_embeddings, embedding_dim, collect_state=True)
    s = torch.load('s.dat')
    print(s.keys())
    if P_x.active:
        embedding.load_state_dict(s)

    # Forward pass
    y_ = embedding(input_idx)

    # Collect and save output
    gather = Repartition(P_x, P_root)
    y_ = gather(y_.view(1, *y_.shape))

    # Print error
    if P_root.active:
        print(torch.norm(y - y_) / torch.norm(y_))