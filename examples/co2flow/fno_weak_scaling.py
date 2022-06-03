import distdl, torch, os, time, h5py
import numpy as np
import azure.storage.blob
from mpi4py import MPI
from pfno import ParallelFNO4d, DistributedRelativeLpLoss
from distdl.backends.mpi.partition import MPIPartition
from sleipner_dataset import DistributedSleipnerDataset3D
from distdl.nn.repartition import Repartition
from distdl.utilities.torch import zero_volume_tensor

###################################################################################################
# Set up MPI and worker partitions

# Init MPI
P_world = MPIPartition(MPI.COMM_WORLD)
P_world._comm.Barrier()
nworkers = P_world.shape[0]

# Master worker partition with 6 dimensions ( N C X Y Z T )
root_shape = (1, 1, 1, 1, 1, 1)
P_root_base = P_world.create_partition_inclusive([0])
P_root = P_root_base.create_cartesian_topology_partition(root_shape)

# Distributed partitions
feat_workers = np.arange(0, nworkers)
P_feat_base = P_world.create_partition_inclusive(feat_workers)
P_x = P_feat_base.create_cartesian_topology_partition((1,1,nworkers,1,1,1))
P_y = P_feat_base.create_cartesian_topology_partition((1,1,1,nworkers,1,1))

# Cuda
device = torch.device(f'cuda:{P_x.rank}')

# Reproducibility
torch.manual_seed(P_x.rank + 123)
np.random.seed(P_x.rank + 123)

###################################################################################################
# Set up network and data

# Data dimensions
nb = 1
n = 32
shape = (nworkers*n, n, n, n)    # X Y Z T
num_train = 1
num_valid = 1

# Scatter data to workers
scatter = Repartition(P_root, P_x)

# Network dimensions
channel_in = 2
channel_hidden = 4
channel_out = 1
num_k = (nworkers*n,n, n, n//2)

# FNO
pfno = ParallelFNO4d(
    P_world, 
    P_root,
    P_x,
    P_y,
    channel_in,
    channel_hidden,
    channel_out,
    shape,
    num_k,
    balance=True
).to(device)

# Create some random input data
x = zero_volume_tensor()
if P_root.active:
    x = torch.randn(nb, channel_in, *shape)

# Distribute among workers
x = scatter(x).to(device)
print("Rank: ", P_x.rank, "; x.shape: ", x.shape)

###################################################################################################
# Measure runtimes

# Evaluate network
t0 = time.time()
y = pfno(x)

# Compute gradient and backprop
fval = torch.norm(y)**2
fval.backward()

t1 = time.time()
dt = t1 - t0
print("Runtime from rank, ", P_x.rank, ": ", dt)