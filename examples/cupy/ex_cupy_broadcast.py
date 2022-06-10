from mpi4py import MPI
import cupy as cp

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

with cp.cuda.Device(rank):

    # Create arrays on gpu
    if rank == 0:
        data = cp.ones((100,100,100))
        print("Source data from rank {} on device {}".format(rank, data.device))
    else:
        data = cp.empty((100,100,100))
        print("Destination on rank {} on device {}".format(rank, data.device))


    # Make sure GPU buffer is ready
    cp.cuda.get_current_stream().synchronize()

    # Broadcast data
    comm.Bcast(data, root=0)
