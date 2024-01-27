import numpy as np
import torch
from mpi4py import MPI

from distdl.backends.common.partition import MPIPartition
from distdl.config import set_backend
from distdl.nn.linear_ag_zero import DistributedLinearAllGatherZero
from distdl.nn.linear_rs_zero import DistributedLinearReduceScatterZero
from distdl.nn.repartition import Repartition
from distdl.nn.sum_reduce import SumReduce
from distdl.utilities.torch import zero_volume_tensor
from distdl.functional import ZeroVolumeCorrectorFunction
import sys

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

# Input/output data dimensions
batch_size = 8
num_tokens = 64
in_channels = 128
hidden_channels = int(4 * 128)

linear = DistributedLinearReduceScatterZero(P_x, in_channels, hidden_channels, collect_state=True)
# linear = DistributedLinearAllGatherZero(P_x, in_channels, hidden_channels, collect_state=True)
mse = torch.nn.MSELoss()

# Load/save state dict
if mode == 'train':
    state = linear.state_dict()
    if P_root.active:
        torch.save(state, 'linear_state.pt')
else:
    state = torch.load('linear_state.pt')
    linear.load_state_dict(state)

# Scatter/gather data
scatter_x = Repartition(P_root, P_x, preserve_batch=False)
scatter_label = Repartition(P_root, P_x, preserve_batch=False)
gather_y = Repartition(P_x, P_root, preserve_batch=False)
sum_reduce_loss = SumReduce(P_x, P_root, preserve_batch=False)

# Data
x = zero_volume_tensor(device=P_x.device, requires_grad=True)
label = zero_volume_tensor(device=P_x.device, requires_grad=True)

if mode == 'train':
    if P_root.active:
        x = torch.randn(batch_size, num_tokens, in_channels, requires_grad=True).to(P_x.device)
        label = torch.randn(batch_size, num_tokens, hidden_channels, requires_grad=True).to(P_x.device)
        torch.save(x, 'x.pt')
        torch.save(label, 'label.pt')
    x = scatter_x(x)
    label = scatter_label(label)
else:
    if P_root.active:
        x = torch.load('x.pt')
        label = torch.load('label.pt')
    x = scatter_x(x)
    label = scatter_label(label)

# Forward pass
y = linear(x)

# Compute loss
loss = mse(y, label)
loss = sum_reduce_loss(loss) / P_x.size
loss = ZeroVolumeCorrectorFunction.apply(loss)

# Backward pass
loss.backward()

# Do a small update
for p in linear.parameters():
    p.data -= 0.1 * p.grad
linear.zero_grad()

# Now do another forward pass
y = linear(x)

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
