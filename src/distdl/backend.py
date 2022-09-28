# TODO: this source should move to backends package

from inspect import trace
import os
from pickle import FALSE
import traceback as tb
from enum import Enum
import distdl.logger as logger
import distdl.backends.mpi_mpi_numpy as mpi_mpi_numpy
import distdl.backends.mpi_mpi_cupy as mpi_mpi_cupy
import distdl.backends.mpi_mpi_torch as mpi_mpi_torch
import distdl.backends.mpi_nccl_cupy as mpi_nccl_cupy
import distdl.utilities.dtype as dtype_utils
import cupy as cp
import torch

# Communication protocal for the frontend. This includes
# the communications of tensor shapes, partition sizes etc.
# Currently only MPI is supported.
class FrontEndProtocol(Enum):
    MPI = 1

# Communication protocal for the backend, which is used
# to communicate tensors in the forward/backward evaluation
# of torch modules.
class BackendProtocol(Enum):
    MPI = 1
    NCCL = 2

# Array/tensor representation that is used in the backend. 
# The frontend always uses torch tensors. If packages other
# than torch are used in the backend, tensors are converted
# to the other format for calling the communication primitives.
class ModelProtocol(Enum):
    CUPY = 1
    NUMPY = 2
    TORCH = 3


# default options
backend = None
_model_protocol = ModelProtocol.CUPY
_backend_protocol = BackendProtocol.MPI
_frontend_protocol = FrontEndProtocol.MPI

# Environment variable names
MODEL_ENVAR_NAME = "DISTDL_MODEL"
BACKEND_ENVAR_NAME = "DISTDL_BACKEND"
FRONTEND_ENVAR_NAME = "DISTDL_FRONTEND"

# Devices should be initialized only once, so this flag takes care of that
device_initialized = False

# The backend is stored as a global variable. If no backend is set by the user,
# a default backend (mpi-mpi-cupy) will be created.
def get_backend():
    global backend
    if backend == None:
        logger.warning("Uninitialized backend detected!")
        # tb.print_stack()
        _init_distdl()
    return backend


# TODO: handle mapping configuration from user input
# Currently we have n-to-1 relationship between ranks and GPUs
# Each rank works on only one GPU at a time, and each GPU may be
# the defult device of multiple ranks
def init_device(requested_device=None, rank=None):
    global backend
    logger.info(f"Requested device: {requested_device}")

    if backend == None:
        _init_distdl()

    if _model_protocol == ModelProtocol.CUPY:
        cp.cuda.runtime.setDevice(rank % cp.cuda.runtime.getDeviceCount())
        return cp.cuda.runtime.getDevice()
    elif _model_protocol == ModelProtocol.NUMPY:
        return torch.device("cpu")
    elif _model_protocol == ModelProtocol.TORCH and requested_device == None:
        torch.cuda.set_device(rank % torch.cuda.device_count())
        return torch.cuda.current_device()
    elif _model_protocol == ModelProtocol.TORCH and requested_device == "cuda":
        torch.cuda.set_device(rank % torch.cuda.device_count())
        return torch.cuda.current_device()
    elif _model_protocol == ModelProtocol.TORCH and requested_device == "cpu":
        # Right now, if the user wants to create the buffer manager as Torch tensors
        # on cpu, they should do something like this:
        # P_world = MPIPartition(MPI.COMM_WORLD, device="cpu")
        return torch.device("cpu")
    else:
        logger.warning("Invalid protocols are requested.")
        return torch.device("cpu")


def get_current_device(requested_device=None, rank=None):
    global device_initialized

    if device_initialized == False:
        init_device(requested_device=requested_device, rank=rank)
        device_initialized = True

    if _model_protocol == ModelProtocol.CUPY:
        return cp.cuda.runtime.getDevice()
    elif _model_protocol == ModelProtocol.NUMPY:
        return torch.device("cpu")
    elif _model_protocol == ModelProtocol.TORCH and requested_device == "cpu":
        return torch.device("cpu")
    elif _model_protocol == ModelProtocol.TORCH and requested_device == "cuda":
        return torch.cuda.current_device()
    elif _model_protocol == ModelProtocol.TORCH and requested_device == None:
        return torch.cuda.current_device()
    else:
        return torch.device("cpu")


def init_distdl(frontend_protocol=None, backend_protocol=None, model_protocol=None):
    global _backend_protocol, _frontend_protocol, _model_protocol
    global backend

    if frontend_protocol != None:
        _frontend_protocol = frontend_protocol
    else:
        _frontend_protocol = FrontEndProtocol.MPI

    if backend_protocol != None:
        _backend_protocol = backend_protocol
    else:
        _backend_protocol = BackendProtocol.MPI

    if model_protocol != None:
        _model_protocol = model_protocol
    else:
        _model_protocol = ModelProtocol.CUPY

    if(_frontend_protocol == FrontEndProtocol.MPI and
       _backend_protocol == BackendProtocol.MPI and
       _model_protocol == ModelProtocol.CUPY):
        backend = mpi_mpi_cupy
        logger.info("Configuration MPI_MPI_CUPY has been selected.")
    elif(_frontend_protocol == FrontEndProtocol.MPI and
         _backend_protocol == BackendProtocol.MPI and
         _model_protocol == ModelProtocol.NUMPY):
        backend = mpi_mpi_numpy
        logger.info("Configuration MPI_MPI_NUMPY has been selected.")
    elif(_frontend_protocol == FrontEndProtocol.MPI and
         _backend_protocol == BackendProtocol.MPI and
         _model_protocol == ModelProtocol.TORCH):
        backend = mpi_mpi_torch
        logger.info("Configuration MPI_MPI_TORCH has been selected.")
    elif(_frontend_protocol == FrontEndProtocol.MPI and
         _backend_protocol == BackendProtocol.NCCL and
         _model_protocol == ModelProtocol.CUPY):
        backend = mpi_nccl_cupy
        logger.info("Configuration MPI_NCCL_CUPY has been selected.")
    else:
        # Invalid configuration
        logger.error("Invalid Configuration has been selected.")
        tb.print_exc()
        backend = mpi_mpi_numpy


def _init_distdl():
    try:
        # Selecting the backend based on env vars
        if os.environ[MODEL_ENVAR_NAME] == "cupy":
            init_distdl(model_protocol=ModelProtocol.CUPY)
        elif os.environ[MODEL_ENVAR_NAME] == "numpy":
            init_distdl(model_protocol=ModelProtocol.NUMPY)
        elif os.environ[MODEL_ENVAR_NAME] == "torch":
            init_distdl(model_protocol=ModelProtocol.TORCH)
        else:
            logger.warning("No Configuration has been specified in env vars, Numpy will be selected.")
            init_distdl(model_protocol=ModelProtocol.NUMPY)
    except:
        logger.error("Invalid model protocol was specified in env vars, Numpy will be selected.")
        init_distdl(model_protocol=ModelProtocol.NUMPY)


def convert_torch_to_model_dtype(dtype):
    if _model_protocol == ModelProtocol.CUPY:
        return dtype_utils.torch_to_cupy_dtype_dict[dtype]
    if _model_protocol == ModelProtocol.NUMPY:
        return dtype_utils.torch_to_numpy_dtype_dict[dtype]
    if _model_protocol == ModelProtocol.TORCH:
        return dtype
    logger.error("Selected model doesn't exist!")


def convert_model_to_torch_dtype(dtype):
    if _model_protocol == ModelProtocol.CUPY:
        return dtype_utils.cupy_to_torch_dtype_dict[dtype]
    if _model_protocol == ModelProtocol.NUMPY:
        return dtype_utils.numpy_to_torch_dtype_dict[dtype]
    if _model_protocol == ModelProtocol.TORCH:
        return dtype
    logger.error("Selected model doesn't exist!")


def convert_intID_to_model_dtype_dict(intID):
    if _model_protocol == ModelProtocol.CUPY:
        return dtype_utils.intID_to_cupy_dtype_dict[intID]
    if _model_protocol == ModelProtocol.NUMPY:
        return dtype_utils.intID_to_numpy_dtype_dict[intID]
    if _model_protocol == ModelProtocol.TORCH:
        return dtype_utils.intID_to_torch_dtype_dict[intID]
    logger.error("Selected model doesn't exist!")


def convert_model_to_intID_dtype_dict(dtype):
    if _model_protocol == ModelProtocol.CUPY:
        return dtype_utils.cupy_to_intID_dtype_dict[dtype]
    if _model_protocol == ModelProtocol.NUMPY:
        return dtype_utils.numpy_to_intID_dtype_dict[dtype]
    if _model_protocol == ModelProtocol.TORCH:
        return dtype_utils.torch_to_intID_dtype_dict[dtype]
    logger.error("Selected model doesn't exist!")
