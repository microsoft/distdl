import numpy as np
import torch
from mpi4py import MPI
from distdl.config import set_backend

import distdl.utilities.slicing as slicing
from distdl.backends.common.partition import MPIPartition
from distdl.nn.all_gather import AllGather
from distdl.utilities.torch import zero_volume_tensor

# Set backend
set_backend(backend_comm="nccl", backend_array="cupy")

# Global MPI communicator
P_world = MPIPartition(MPI.COMM_WORLD)
P_world._comm.Barrier()

# Global cartesian communicator
in_shape = (32, 1)           # (servers, 1)
in_size = np.prod(in_shape)
in_workers = np.arange(0, in_size)

P_x_base = P_world.create_partition_inclusive(in_workers)
P_x = P_x_base.create_cartesian_topology_partition(in_shape)
P_x.set_frontend_network(False)

# Inter DC all-gather
all_gather = AllGather(P_x, axes_all_gather=(0,))

# Create some weights
x_global_shape = np.array([int(12288*12288), 1])
x = zero_volume_tensor(device=P_x.device)

if P_x.active:
    x_local_shape = slicing.compute_subshape(P_x.shape, P_x.index, x_global_shape)
    x = torch.zeros(*x_local_shape, device=x.device) + (P_x.rank + 1)

print("Initial tensor shape: {} on rank {}.".format(x.shape, P_x.rank))

# Burn in
y = all_gather(x)

# Timings
num_runs = 500
t = torch.zeros(num_runs)

for i in range(num_runs):

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    y = all_gather(x)    
    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()

    t[i] = start.elapsed_time(end)
    if P_world.rank == 0:
        print("Iteration {}: {} ms.".format(i, t[i]))

if P_world.rank == 0:
    print("All-gather time: {} +- {} ms.".format(t.mean(), t.std()))
    torch.save(t, "all_gather_frontend.pt")