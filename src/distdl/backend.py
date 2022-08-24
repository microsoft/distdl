# TODO: this source should move to backends package

import os
from enum import Enum
import distdl.backends.mpi_mpi_numpy as mpi_mpi_numpy
import distdl.backends.mpi_mpi_cupy as mpi_mpi_cupy
import distdl.backends.mpi_mpi_torch as mpi_mpi_torch
import cupy as cp
import torch


class FrontEndProtocol(Enum):
    MPI = 1


class BackendProtocol(Enum):
    MPI = 1
    NCCL = 2


class ModelProtocol(Enum):
    CUPY = 1
    NUMPY = 2
    TORCH = 3


class ModelDevice(Enum):
    CPU = 1
    GPU = 1


# defult options
backend = None
_model_protocol = ModelProtocol.CUPY
_backend_protocol = BackendProtocol.MPI
_frontend_protocol = FrontEndProtocol.MPI

# Environment variable names
MODEL_ENVAR_NAME = "DISTDL_MODEL"
BACKEND_ENVAR_NAME = "DISTDL_BACKEND"
FRONTEND_ENVAR_NAME = "DISTDL_FRONTEND"


def get_backend():
    global backend
    if backend == None:
        _init_distdl()
    return backend


# TODO: handle mapping configuration from user input
# Currently we have n-to-1 relationship between ranks and GPUs
# Each rank works on only one GPU at a time, and each GPU may be
# the defult device of multiple ranks
def get_device(requested_device=None, rank=None):
    global backend
    if backend == None:
        _init_distdl()

    if _model_protocol == ModelProtocol.CUPY:
        # print(f"1 - backend: {backend}, model: {_model_protocol}")
        cp.cuda.runtime.setDevice(rank % cp.cuda.runtime.getDeviceCount())
        return cp.cuda.runtime.getDevice()
    elif _model_protocol == ModelProtocol.NUMPY:
        # print(f"2 - backend: {backend}, model: {_model_protocol}")
        return torch.device("cpu")
    elif _model_protocol == ModelProtocol.TORCH and requested_device == None:
        # print(f"3 - backend: {backend}, model: {_model_protocol}")
        torch.cuda.set_device(rank % torch.cuda.device_count())
        return torch.cuda.current_device()
    elif _model_protocol == ModelProtocol.TORCH and requested_device == ModelDevice.CPU:
        # print(f"4 - backend: {backend}, model: {_model_protocol}")
        return torch.device("cpu")
    elif _model_protocol == ModelProtocol.TORCH and requested_device == ModelDevice.GPU:
        # print(f"5 - backend: {backend}, model: {_model_protocol}")
        torch.cuda.set_device(rank % torch.cuda.device_count())
        return torch.cuda.current_device()
    else:
        # print(f"6 - backend: {backend}, model: {_model_protocol}")
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
        print("Configuration MPI_MPI_CUPY has been selected.")
    elif(_frontend_protocol == FrontEndProtocol.MPI and
         _backend_protocol == BackendProtocol.MPI and
         _model_protocol == ModelProtocol.NUMPY):
        backend = mpi_mpi_numpy
        print("Configuration MPI_MPI_NUMPY has been selected.")
    elif(_frontend_protocol == FrontEndProtocol.MPI and
         _backend_protocol == BackendProtocol.MPI and
         _model_protocol == ModelProtocol.TORCH):
        backend = mpi_mpi_torch
        print("Configuration MPI_MPI_TORCH has been selected.")
    else:
        # Invalid configuration
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
            init_distdl(model_protocol=ModelProtocol.NUMPY)

    except:
        init_distdl(model_protocol=ModelProtocol.NUMPY)
