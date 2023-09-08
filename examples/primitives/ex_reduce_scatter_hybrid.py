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

# Create inter DC communicators
for i in range(in_shape[1]):
    workers = np.arange(i, i + num_cluster*in_shape[1], in_shape[1])
    P_inter_dc_base = P_world.create_partition_inclusive(workers)

    # If I am a worker belonging to this partition, create the communicator
    if P_inter_dc_base.active:
        P_inter_dc = P_inter_dc_base.create_cartesian_topology_partition((num_cluster, 1, 1))
        P_inter_dc.set_frontend_network(True)  # Set to True for hybrid
    P_inter_dc_base.deactivate()

# Create intra DC communicators
for i in range(in_shape[0]):
    workers = np.arange(i*in_shape[1], (i+1)*in_shape[1])
    P_intra_dc_base = P_world.create_partition_inclusive(workers)

    # If I am a worker belonging to this partition, create the communicator
    if P_intra_dc_base.active:
        P_intra_dc = P_intra_dc_base.create_cartesian_topology_partition((1, num_nodes, 1))
        P_intra_dc.set_frontend_network(False)  # Set to False for hybrid
    P_intra_dc_base.deactivate()


# Every worker should be part of one intra- and one inter-DC communicator
print("Communicators on rank {}: {}, {}".format(P_world.rank, P_inter_dc, P_intra_dc))

# Inter DC reduce-scatter
inter_dc_reduce_scatter = ReduceScatter(P_inter_dc, axes_reduce_scatter=(0,))

# Intra DC reduce-scatter
intra_dc_reduce_scatter = ReduceScatter(P_intra_dc, axes_reduce_scatter=(1,))


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