from distdl.backends.mpi.buffer import MPIBufferManager
import numpy as np

# Create buffer manager (when network is created)
num_buffers = 2
buffer_dtype = np.dtype('float64')
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
x = np.random.randn(*tensor_shape)
y = np.random.randn(*tensor_shape)

# Copy our data to the buffer
np.copyto(buffers[0].get_view(tensor_shape), x)
np.copyto(buffers[1].get_view(tensor_shape), y)

# Inspect buffers again (should contain our arrays now)
print(buffers[0].get_view(tensor_shape))
print(buffers[1].get_view(tensor_shape))

# Resize the first buffer
new_tensor_shape = (4, 2)
buffers[0].expand(np.prod(new_tensor_shape))

# Populate new buffer
z = np.random.randn(*new_tensor_shape)
np.copyto(buffers[0].get_view(new_tensor_shape), z)
print(buffers[0].get_view(new_tensor_shape))