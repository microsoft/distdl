from distdl.backends.mpi.buffer import MPIBufferManager
import cupy as cp

# Create buffer manager (when network is created)
num_buffers = 2
buffer_dtype = cp.dtype('float64')
buffer_manager = MPIBufferManager()

# ...