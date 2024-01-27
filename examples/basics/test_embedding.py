import sys

import numpy as np
import torch
from mpi4py import MPI

import distdl
from distdl.backends.common.partition import MPIPartition
from distdl.config import set_backend
from distdl.functional import ZeroVolumeCorrectorFunction
from distdl.nn.embedding_zero import DistributedEmbeddingZero
from distdl.nn.embedding import DistributedEmbedding
from distdl.nn.repartition import Repartition
from distdl.nn.sum_reduce import SumReduce
from distdl.nn.broadcast import Broadcast
from distdl.utilities.torch import zero_volume_tensor

##########################################################################

def create_partition(P_x: MPIPartition, dim=None, axis=None) -> MPIPartition:

    # Number of dimensions in partition
    if dim is None:
        dim = P_x.dim

    # Shape of data partition
    partition_shape = [1] * dim
    if axis is not None:
        partition_shape[axis] = P_x.shape[axis]

    # Get corresponding worker indices
    index = [slice(0, 1)] * P_x.dim
    if axis is not None:
        index[axis] = slice(0, P_x.shape[axis])
    workers = distdl.utilities.slicing.worker_layout(P_x.shape)[tuple(index)].reshape(-1).tolist()

    P_data_base = P_x.create_partition_inclusive(workers)
    P_data = P_data_base.create_cartesian_topology_partition(partition_shape)
    P_data_base.deactivate()

    return P_data

##########################################################################

# Set backend
set_backend(backend_comm="mpi", backend_array="numpy")

# Set up MPI cartesian communicator
P_world = MPIPartition(MPI.COMM_WORLD)
P_world._comm.Barrier()

# Get mode from command line
if len(sys.argv) > 0:
    mode = sys.argv[1]  # 'train', 'eval'
else:
    mode = 'train'

# Data partition corresponding to [ batch, ..., channel/embedding ]
if P_world.size > 1:
    n_model = 2
    n_data = P_world.size // n_model
    in_shape = (n_data, 1, n_model)
else:
    in_shape = (1, 1, 1)
if P_world.rank == 0:
    print("Run in {} with P_x: {}".format(mode, in_shape))

in_size = np.prod(in_shape)
in_workers = np.arange(0, in_size)

P_root_base = P_world.create_partition_inclusive([0])
P_root = P_root_base.create_cartesian_topology_partition([1, 1, 1])

P_x_base = P_world.create_partition_inclusive(in_workers)
P_x = P_x_base.create_cartesian_topology_partition(in_shape)

P_data = create_partition(P_x, axis=0)

if P_root.active:
    print("Partitions: ", P_x.shape, P_data.shape, P_root.shape)

# Input/output data dimensions
batch_size = 8
num_tokens = 64
num_embedding = 128

embedding = DistributedEmbeddingZero(P_x, num_tokens, num_embedding, collect_state=True)
mse = torch.nn.MSELoss()

# Load/save state dict
if mode == 'train':
    state = embedding.state_dict()
    if P_root.active:
        torch.save(state, 'embd_state.pt')
else:
    state = torch.load('embd_state.pt')
    embedding.load_state_dict(state)

# Scatter/gather data
scatter_x = Repartition(P_root, P_data, preserve_batch=False)
broadcast_x = Broadcast(P_data, P_x)
scatter_label = Repartition(P_root, P_x, preserve_batch=False)
gather_y = Repartition(P_x, P_root, preserve_batch=False)
sum_reduce_loss = SumReduce(P_x, P_root, preserve_batch=False)

# Data
x = zero_volume_tensor(device=P_x.device, requires_grad=True)
label = zero_volume_tensor(device=P_x.device, requires_grad=True)

if mode == 'train':
    if P_root.active:
        x = torch.randint(0, num_tokens, (batch_size, num_tokens, 1), device=P_x.device)
        label = torch.randn(batch_size, num_tokens, num_embedding, requires_grad=True).to(P_x.device)
        torch.save(x, 'x.pt')
        torch.save(label, 'label.pt')
    x = broadcast_x(scatter_x(x))
    label = scatter_label(label)
else:
    if P_root.active:
        x = torch.load('x.pt')
        label = torch.load('label.pt')
    x = broadcast_x(scatter_x(x))
    label = scatter_label(label)

x = x.squeeze(2)

# Forward pass
y = embedding(x)

# Compute loss
loss = mse(y, label)
loss = sum_reduce_loss(loss) / P_x.size
loss = ZeroVolumeCorrectorFunction.apply(loss)

# Backward pass
loss.backward()

# Do a small update
for p in embedding.parameters():
    p.data -= 0.1 * p.grad
embedding.zero_grad()

# Now do another forward pass
#y = embedding(x)

# Collect logits
y = gather_y(y)

if mode == 'train':
    if P_root.active:
        torch.save(loss, 'loss.pt')
        torch.save(y, 'y.pt')
else:
    if P_root.active:
        loss_ = torch.load('loss.pt')
        y_ = torch.load('y.pt')
        print("Loss error: ", loss.item() - loss_.item())
        print("Loss logits: ", torch.norm(y - y_) / torch.norm(y_))
