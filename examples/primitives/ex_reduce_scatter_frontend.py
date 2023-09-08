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

# Global cartesian communicator
in_shape = (1, 32, 1)           # (servers, 1)
in_size = np.prod(in_shape)
in_workers = np.arange(0, in_size)

P_x_base = P_world.create_partition_inclusive(in_workers)
P_x = P_x_base.create_cartesian_topology_partition(in_shape)
P_x.set_frontend_network(True)

# Inter DC reduce-scatter
reduce_scatter = ReduceScatter(P_x, axes_reduce_scatter=(1,))

# Create some weights
x_global_shape = np.array([1, 12288, 49152])
x = zero_volume_tensor(device=P_x.device)

if P_x.active:
    x = torch.zeros(*x_global_shape, device=x.device) + (P_x.rank + 1)

print("Initial tensor shape: {} on rank {}.".format(x.shape, P_x.rank))

# Burn in
y = reduce_scatter(x)

# Timings
num_runs = 500
t = torch.zeros(num_runs)

for i in range(num_runs):

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    y = reduce_scatter(x)    
    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()

    t[i] = start.elapsed_time(end)
    if P_world.rank == 0:
        print("Iteration {}: {} ms.".format(i, t[i]))

if P_world.rank == 0:
    print("Reduce-scatter time: {} +- {} ms.".format(t.mean(), t.std()))
    torch.save(t, "reduce_scatter_frontend.pt")