import os
import distdl.logger as logger
import distdl.backends

# Environment variable names
BACKEND_COMM_ENV = "DISTDL_BACKEND_COMM"
BACKEND_ARRAY_ENV = "DISTDL_BACKEND_ARRAY"

# Get default communication protocol
def get_default_comm():

    # Check if backend communication protocol is set as 
    # an environment variable. If not, set to default (mpi).
    supported_backend_comm = ['mpi', 'nccl']
    if BACKEND_COMM_ENV in os.environ:
        if os.environ[BACKEND_COMM_ENV] in supported_backend_comm:
            backend_comm = os.environ[BACKEND_COMM_ENV]
        else:
            logger.warning("Specified backend communication protocol does not exist. Default to mpi.")
            backend_comm = "mpi"
    else:
        backend_comm = "mpi"
    return backend_comm


# Get default array representation
def get_default_array():

    # Check if backend array representation is set.
    # If not, default to numpy.
    supported_backend_array = ['numpy', 'cupy', 'torch']
    if BACKEND_ARRAY_ENV in os.environ:
        if os.environ[BACKEND_ARRAY_ENV] in supported_backend_array:
            backend_array = os.environ[BACKEND_ARRAY_ENV]
        else:
            logger.warning("Specified backend array representation does not exist. Default to numpy.")
            backend_array = "numpy"
    else:
        backend_array = "numpy"
    return backend_array


def set_backend(backend_comm=None, backend_array=None):

    # Get default config
    if backend_comm is None:
        backend_comm = get_default_comm()
    if backend_array is None:
        backend_array = get_default_array()

    backend_config = '_'.join([backend_comm, backend_array])

    if backend_config in distdl.backends.supported_backends and \
        distdl.backends.supported_backends[backend_config] is not None:
        distdl.backends.backend = distdl.backends.supported_backends[backend_config]
    else:
        logger.warning("Selected backend not supported. Default to mpi-numpy.")
        distdl.backends.backend = distdl.backends.supported_backends['mpi_numpy']



# TODO : decide what to do with the device
# # Devices should be initialized only once, so this flag takes care of that
# device_initialized = False

# # Initialize device based on backend array representation
# def init_device(requested_device=None, rank=None):
#     logger.info(f"Requested device: {requested_device}")

#     if array == "CUPY":
#         cp.cuda.runtime.setDevice(rank % cp.cuda.runtime.getDeviceCount())
#         return cp.cuda.runtime.getDevice()
#     elif array == "NUMPY":
#         return torch.device("cpu")
#     elif array == "TORCH" and requested_device == None:
#         torch.cuda.set_device(rank % torch.cuda.device_count())
#         return torch.cuda.current_device()
#     elif array == "TORCH" and requested_device == "cuda":
#         torch.cuda.set_device(rank % torch.cuda.device_count())
#         return torch.cuda.current_device()
#     elif array == "TORCH" and requested_device == "cpu":
#         return torch.device("cpu")
#     else:
#         logger.warning("Invalid protocols are requested.")
#         return torch.device("cpu")

# # Get the current device and initialize if not set
# def get_current_device(requested_device=None, rank=None):
#     global device_initialized

#     if device_initialized == False:
#         init_device(requested_device=requested_device, rank=rank)
#         device_initialized = True

#     if array == "CUPY":
#         return cp.cuda.runtime.getDevice()
#     elif array == "NUMPY":
#         return torch.device("cpu")
#     elif array == "TORCH" and requested_device == "cpu":
#         return torch.device("cpu")
#     elif array == "TORCH" and requested_device == "cuda":
#         return torch.cuda.current_device()
#     elif array == "TORCH" and requested_device == None:
#         return torch.cuda.current_device()
#     else:
#         return torch.device("cpu")