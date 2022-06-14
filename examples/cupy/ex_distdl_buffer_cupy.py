from distdl.backends.mpi.buffer_cupy import MPIBufferManager
import cupy as cp
import numpy as np

# Create buffer manager (when network is created)
num_buffers = 2
buffer_dtype = cp.dtype('float64')
buffer_manager = MPIBufferManager()

# Request (empty) buffers (when network is called for 1st time)
buffers = buffer_manager.request_buffers(num_buffers, buffer_dtype)

# Allocate shape to each buffer
tensor_shape = (3, 2)
buffers[0].allocate_view(tensor_shape)
buffers[1].allocate_view(tensor_shape)

# Inspect buffers (should be empty or have random stuff in it)
print(buffers[0].get_view(tensor_shape))
print(buffers[1].get_view(tensor_shape))

# Create some input as numpy tensor
x = cp.random.randn(*tensor_shape)
y = cp.random.randn(*tensor_shape)

# Copy our data to the buffer
cp.copyto(buffers[0].get_view(tensor_shape), x)
cp.copyto(buffers[1].get_view(tensor_shape), y)

# Inspect buffers again (should contain our arrays now)
print(buffers[0].get_view(tensor_shape))
print(buffers[1].get_view(tensor_shape))

# Resize the first buffer
new_tensor_shape = (4, 2)
buffers[0].expand(np.prod(new_tensor_shape))

print(buffers[0].get_view(new_tensor_shape))

# Populate new buffer
z = cp.random.randn(*new_tensor_shape)
cp.copyto(buffers[0].get_view(new_tensor_shape), z)
print(buffers[0].get_view(new_tensor_shape))