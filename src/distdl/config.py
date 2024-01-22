import os

import distdl.backends
import distdl.logger as logger
from distdl.backends import supported_backends

# Environment variable names
BACKEND_COMM_ENV = "DISTDL_BACKEND_COMM"
BACKEND_ARRAY_ENV = "DISTDL_BACKEND_ARRAY"
PRE_HOOK_CHECK_INPUT_CHANGED_ENV = "DISTDL_CHECK_INPUT_CHANGED"


# Get default communication protocol
def get_default_comm():

    # Check if backend communication protocol is set as
    # an environment variable. If not, set to default (mpi).
    supported_backend_comm = ['mpi', 'nccl']
    if BACKEND_COMM_ENV in os.environ:
        if os.environ[BACKEND_COMM_ENV] in supported_backend_comm:
            backend_comm = os.environ[BACKEND_COMM_ENV]
        else:
            logger.logger.warning("Specified backend communication protocol does not exist. Default to mpi.")
            backend_comm = "mpi"
    else:
        backend_comm = "mpi"
    return backend_comm


# Get default array representation
def get_default_array():

    # Check if backend array representation is set.
    # If not, default to numpy.
    supported_backend_array = ['numpy', 'cupy']
    if BACKEND_ARRAY_ENV in os.environ:
        if os.environ[BACKEND_ARRAY_ENV] in supported_backend_array:
            backend_array = os.environ[BACKEND_ARRAY_ENV]
        else:
            logger.logger.warning("Specified backend array representation does not exist. Default to numpy.")
            backend_array = "numpy"
    else:
        backend_array = "numpy"
    return backend_array


def get_default_pre_hook_setting():
    supported_settings = ['0', '1', 'False', 'True']
    if PRE_HOOK_CHECK_INPUT_CHANGED_ENV in os.environ:
        if os.environ[PRE_HOOK_CHECK_INPUT_CHANGED_ENV] in supported_settings:
            check_input_changed = bool(int(os.environ[PRE_HOOK_CHECK_INPUT_CHANGED_ENV]))
        else:
            logger.logger.warning("Specified setting for check input changed does not exist. Default to False.")
    else:
        check_input_changed = True
    return check_input_changed


def set_backend(backend_comm=None, backend_array=None, check_input_changed=None):

    # Get default config
    if backend_comm is None:
        backend_comm = get_default_comm()
    if backend_array is None:
        backend_array = get_default_array()
    if check_input_changed is None:
        check_input_changed = get_default_pre_hook_setting()
    distdl.config.check_input_changed = check_input_changed

    backend_config = '_'.join([backend_comm, backend_array])

    if backend_config in supported_backends and supported_backends[backend_config] is not None:
        distdl.backends.backend = supported_backends[backend_config]
    else:
        logger.logger.warning("Selected backend not supported. Default to mpi-numpy.")
        distdl.backends.backend = supported_backends['mpi_numpy']
