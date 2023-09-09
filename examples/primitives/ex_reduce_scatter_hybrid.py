import numpy as np
import torch
from mpi4py import MPI
from distdl.config import set_backend

import distdl.utilities.slicing as slicing
from distdl.backends.common.partition import MPIPartition
from distdl.nn.reduce_scatter import ReduceScatter
from distdl.utilities.torch import zero_volume_tensor

# Set backend
set_backend(backend_comm="nccl", backend_array="cupy")

# Global MPI communicator
P_world = MPIPartition(MPI.COMM_WORLD)
P_world._comm.Barrier()

num_cluster = 2
num_nodes = 16

# Global cartesian communicator
in_shape = (num_cluster, num_nodes, 1)           # (no. of data centers, no. of workers per data center)
in_size = np.prod(in_shape)
in_workers = np.arange(0, in_size)

P_x_base = P_world.create_partition_inclusive(in_workers)
P_x = P_x_base.create_cartesian_topology_partition(in_shape)

# Inter DC reduce-scatter
inter_dc_reduce_scatter = ReduceScatter(P_x, axes_reduce_scatter=(0,), use_frontend=True)

# Intra DC reduce-scatter
intra_dc_reduce_scatter = ReduceScatter(P_x, axes_reduce_scatter=(1,), use_frontend=False)

# Create some weights
x_global_shape = np.array([2, 6144, 49152])
x = zero_volume_tensor(device=P_x.device)

if P_x.active:
    x = torch.zeros(*x_global_shape, device=x.device) + (P_x.rank + 1)

print("Initial tensor shape: {} on rank {}.".format(x.shape, P_x.rank))

# Burn in
y = intra_dc_reduce_scatter(x)
z = inter_dc_reduce_scatter(y)

# Timings
num_runs = 500
t = torch.zeros(num_runs)

for i in range(num_runs):

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    y = intra_dc_reduce_scatter(x)
    z = inter_dc_reduce_scatter(y)   
    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()

    t[i] = start.elapsed_time(end)
    if P_world.rank == 0:
        print("Iteration {}: {} ms.".format(i, t[i]))
        
if P_world.rank == 0:
    print("Reduce-scatter time: {} +- {} ms.".format(t.mean(), t.std()))
    torch.save(t, "reduce_scatter_hybrid.pt")