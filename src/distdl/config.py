import os, importlib
from pickle import FALSE
import traceback as tb
from enum import Enum
import distdl.logger as logger
import torch

# Environment variable names
FRONTEND_ENV_NAME = "DISTDL_FRONTEND"
BACKEND_ENV_NAME = "DISTDL_BACKEND"
ARRAY_ENV_NAME = "DISTDL_ARRAY"

# Devices should be initialized only once, so this flag takes care of that
device_initialized = False

# Fields for backend and array representation
name = None # mpi_mpi_numpy
array = None

# Create default configuration for distdl
def get_default_config():

    # Check if front end communication protocol is specified as
    # an environment variable. If not, set to default protocol (MPI)
    # Remove because only 1 option anyway
    if FRONTEND_ENV_NAME in os.environ:
        frontend_comm_env = os.environ[FRONTEND_ENV_NAME]
        if frontend_comm_env == "MPI":
            frontend_comm = frontend_comm_env
        else:
            logger.warning("Specified frontend communication protocol does not exist. Default to MPI.")
            frontend_comm = "MPI"
    else:
        frontend_comm = "MPI"

    # Check if backend communication protocol is set as 
    # an environment variable. If not, set to default (MPI).
    if BACKEND_ENV_NAME in os.environ:
        backend_comm_env = os.environ[BACKEND_ENR_NAME]
        if backend_comm_env == "MPI":
            backend_comm = backend_comm_env
        elif backend_comm_env == "NCCL":
            backend_comm = backend_comm_env
        else:
            logger.warning("Specified backend communication protocol does not exist. Default to MPI.")
            backend_comm = "MPI"
    else:
        backend_comm = "MPI"

    # Check if backend array representation is set.
    # If not, default to NUMPY.
    if ARRAY_ENV_NAME in os.environ:
        backend_array_env = os.environ[ARRAY_ENV_NAME]
        if backend_array_env == "NUMPY":
            backend_array = backend_array_env
        elif backend_array_env == "CUPY":
            backend_array = backend_array_env
        elif backend_array_env == "TORCH":
            backend_array = backend_array_env
        else:
            logger.warning("Specified backend array representation does not exist. Default to NUMPY.")
            backend_array = "NUMPY"
    else:
        backend_array = "NUMPY"

    return frontend_comm, backend_comm, backend_array

# Change to dictonary: Map dict to string
def set_config(frontend_comm="MPI", backend_comm="MPI", backend_array="NUMPY"):
    global name
    global array

    if(frontend_comm == "MPI" and
       backend_comm == "MPI" and
       backend_array == "CUPY"):
        name = "mpi_mpi_cupy"
        array = backend_array
        logger.info("Configuration MPI_MPI_CUPY has been selected.")

    elif(frontend_comm == "MPI" and
         backend_comm == "MPI" and
         backend_array == "NUMPY"):
        name = "mpi_mpi_numpy"
        array = backend_array
        logger.info("Configuration MPI_MPI_NUMPY has been selected.")

    elif(frontend_comm == "MPI" and
         backend_comm == "MPI" and
         backend_array == "TORCH"):
        name = "mpi_mpi_torch"
        array = backend_array
        logger.info("Configuration MPI_MPI_TORCH has been selected.")

    elif(frontend_comm == "MPI" and
         backend_comm == "NCCL" and
         backend_array == "CUPY"):
        name = "mpi_nccl_cupy"
        array = backend_array
        logger.info("Configuration MPI_NCCL_CUPY has been selected.")

    else:
        logger.error("Invalid Configuration has been selected.")
        tb.print_exc()
        name = "mpi_mpi_numpy"
        array = backend_array


def load_backend_module(current_backend=None):
    if current_backend is None:
        backends = importlib.import_module('distdl.backends')
    else:
        backends = importlib.reload(current_backend)
    return backends


# Initialize device based on backend array representation
def init_device(requested_device=None, rank=None):
    logger.info(f"Requested device: {requested_device}")

    if array == "CUPY":
        cp.cuda.runtime.setDevice(rank % cp.cuda.runtime.getDeviceCount())
        return cp.cuda.runtime.getDevice()
    elif array == "NUMPY":
        return torch.device("cpu")
    elif array == "TORCH" and requested_device == None:
        torch.cuda.set_device(rank % torch.cuda.device_count())
        return torch.cuda.current_device()
    elif array == "TORCH" and requested_device == "cuda":
        torch.cuda.set_device(rank % torch.cuda.device_count())
        return torch.cuda.current_device()
    elif array == "TORCH" and requested_device == "cpu":
        # Right now, if the user wants to create the buffer manager as Torch tensors
        # on cpu, they should do something like this:
        # P_world = MPIPartition(MPI.COMM_WORLD, device="cpu")
        return torch.device("cpu")
    else:
        logger.warning("Invalid protocols are requested.")
        return torch.device("cpu")

# Get the current device and initialize if not set
def get_current_device(requested_device=None, rank=None):
    global device_initialized

    if device_initialized == False:
        init_device(requested_device=requested_device, rank=rank)
        device_initialized = True

    if array == "CUPY":
        return cp.cuda.runtime.getDevice()
    elif array == "NUMPY":
        return torch.device("cpu")
    elif array == "TORCH" and requested_device == "cpu":
        return torch.device("cpu")
    elif array == "TORCH" and requested_device == "cuda":
        return torch.cuda.current_device()
    elif array == "TORCH" and requested_device == None:
        return torch.cuda.current_device()
    else:
        return torch.device("cpu")