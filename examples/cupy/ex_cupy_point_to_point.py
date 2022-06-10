from mpi4py import MPI
import numpy as np
import cupy as cp

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

with cp.cuda.Device(rank):

    # Send
    cp.cuda.get_current_stream().synchronize()
    if rank == 0:
        data = cp.random.randn(200, 200, 200)
        print("Rank ", rank, ": ", data.device)
        comm.Send(data, dest=1, tag=13)

    # Receive
    elif rank == 1:
        data = cp.zeros((200, 200, 200), dtype=np.float64)
        print("Rank ", rank, ": ", data.device)
        comm.Recv(data, source=0, tag=13)