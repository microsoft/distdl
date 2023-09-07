#!/bin/bash

# Load modules
module load openmpi-4.1.5

# GPUs, NUMA node, NIC:IB device
# [0, 1], 0, [ib0: mlx5_0:1, eth4: mlx5_1:1]
# [2, 3], 1, [ib1: mlx5_2:1, eth5: mlx5_3:1]
# [4, 5], 2, [ib2: mlx5_4:1, eth6: mlx5_5:1]
# [6, 7], 3, [ib3: mlx5_6:1, eth7: mlx5_7:1]

# No. of MPI processes
NP=$1
FILE=$2

# Set default values if no argument is passed
if [ -z "$NP" ]; then
    NP=2
fi

HOSTFILE="${HOME}/distdl/examples/basics/hostfile"

# Get my IP address
IP=$(hostname -I | cut -d' ' -f1)

# Run with ROCE
#--hostfile $HOSTFILE \
#    -x NCCL_DEBUG=INFO \
#    -x LD_LIBRARY_PATH=/opt/rocm-custom/rccl/lib \
#    -x NCCL_RINGS="N0 0 1 2 3 N1 N2 4 5 6 7 N3|N3 7 6 5 4 N2 N1 3 2 1 0 N0|N1 2 3 0 1 N0 N3 6 7 4 5 N2|N2 5 4 7 6 N3 N0 1 0 3 2 N1" \
#    --hostfile $HOSTFILE \

mpiexec \
    -np $NP \
    -mca coll_hcoll_enable 0 \
    -mca pml ob1 \
    -mca btl ^openib \
    -mca btl_tcp_if_include eth0 \
    -x NCCL_SOCKET_IFNAME=eth0 \
    -x NCCL_IB_HCA=mlx5_0:1,mlx5_2:1,mlx5_4:1,mlx5_6:1 \
    -x NCCL_IB_DISABLE=0 \
    -x NCCL_IB_CUDA_SUPPORT=1 \
    -x NCCL_IB_QPS_PER_CONNECTION=2 \
    -x NCCL_ROCE_HCA=mlx5_1:1,mlx5_3:1,mlx5_5:1,mlx5_7:1 \
    -x NCCL_ROCE_DISABLE=0 \
    -x NCCL_ROCE_CUDA_SUPPORT=1 \
    -x NCCL_ROCE_QPS_PER_CONNECTION=2 \
    -x NCCL_ROCE_GID_INDEX=3 \
    -x NCCL_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -x NCCL_NCHANNELS_PER_NET_PEER=1 \
    -x NCCL_NET_GDR_LEVEL=SYS \
    -x NCCL_DEBUG=INFO \
    -x NCCL_ALGO=Ring \
    -x LD_LIBRARY_PATH=/opt/rocm-custom/lib/ \
    --map-by node \
    --bind-to numa \
    /home/pwitte/anaconda3/envs/pytorch/bin/python3 $FILE