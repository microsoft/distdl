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