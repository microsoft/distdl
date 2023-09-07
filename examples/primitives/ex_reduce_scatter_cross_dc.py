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
in_shape = (2, 4)           # (no. of data centers, no. of workers per data center)
in_size = np.prod(in_shape)
in_workers = np.arange(0, in_size)

P_x_base = P_world.create_partition_inclusive(in_workers)
P_x = P_x_base.create_cartesian_topology_partition(in_shape)

# Create inter DC communicators
for i in range(in_shape[1]):
    workers = np.arange(i, i + 2*in_shape[1], in_shape[1])
    P_inter_dc_base = P_world.create_partition_inclusive(workers)

    # If I am a worker belonging to this partition, create the communicator
    if P_inter_dc_base.active:
        P_inter_dc = P_inter_dc_base.create_cartesian_topology_partition((2, 1))
        P_inter_dc.set_frontend_network(True)
    P_inter_dc_base.deactivate()

# Create intra DC communicators
for i in range(in_shape[0]):
    workers = np.arange(i*in_shape[1], (i+1)*in_shape[1])
    P_intra_dc_base = P_world.create_partition_inclusive(workers)

    # If I am a worker belonging to this partition, create the communicator
    if P_intra_dc_base.active:
        P_intra_dc = P_intra_dc_base.create_cartesian_topology_partition((1, 4))
        P_intra_dc.set_frontend_network(False)
    P_intra_dc_base.deactivate()


# Every worker should be part of one intra- and one inter-DC communicator
#print("Communicators on rank {}: {}, {}".format(P_world.rank, P_inter_dc, P_intra_dc))

# Inter DC all-gather
inter_dc_reduce_scatter = ReduceScatter(P_inter_dc, axes_reduce_scatter=(0,))

# Intra DC all-gather
intra_dc_reduce_scatter = ReduceScatter(P_intra_dc, axes_reduce_scatter=(1,))


# Create some weights
x_global_shape = np.array([6, 8])
x = zero_volume_tensor(device=P_x.device)

if P_x.active:
    x = torch.zeros(*x_global_shape, device=x.device) + (P_x.rank + 1)

print("Initial tensor shape: {} on rank {}.".format(x.shape, P_x.rank))

# 1st do inter DC all-gather
x = intra_dc_reduce_scatter(x)

print("Tensor shape after inter-cluster reduce-scatter: {} on rank {}.".format(x.shape, P_x.rank))

# 2nd do intra DC all-gather
x = inter_dc_reduce_scatter(x)

print("Tensor shape after intra-cluster reduce-scatter: {} on rank {}.".format(x.shape, P_x.rank))