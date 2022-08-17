# TODO: this source should move to backends package

from enum import Enum
import os
import distdl.backends.mpi_mpi_numpy as mpi_mpi_numpy
import distdl.backends.mpi_mpi_cupy as mpi_mpi_cupy
import distdl.backends.mpi_mpi_torch as mpi_mpi_torch
import torch
import cupy as cp
    
class FrontEndProtocol(Enum):
    MPI = 1

class BackendProtocol(Enum):
    MPI = 1
    NCCL = 1
    
class ModelProtocol(Enum):
    CUPY = 1
    NUMPY = 1
    TORCH = 1
    
backend = None
model_protocol = ModelProtocol.CUPY
backend_protocol = BackendProtocol.MPI
frontend_protocol = FrontEndProtocol.MPI

def get_backend():
    global backend
    if backend == None:
        init()
    return backend


def get_device(requested_device=None, rank=None):
    global backend
    if backend == None:
        init()
    
    if model_protocol == ModelProtocol.CUPY:
        print(f"1 - backend: {backend}, requested device: {requested_device}")
        # TODO: handle mapping configuration from user input
        cp.cuda.runtime.setDevice(rank % cp.cuda.runtime.getDeviceCount())
        return cp.cuda.runtime.getDevice()
    elif model_protocol == ModelProtocol.NUMPY:
        print(f"2 - backend: {backend}, requested device: {requested_device}")
        return torch.device("cpu")
    elif model_protocol == ModelProtocol.TORCH and requested_device == None:
        print(f"3 - backend: {backend}, requested device: {requested_device}")
        torch.cuda.set_device(rank % torch.cuda.device_count())
        return torch.cuda.current_device()
    elif model_protocol == ModelProtocol.TORCH and requested_device == "cpu":
        print(f"4 - backend: {backend}, requested device: {requested_device}")
        return torch.device("cpu")
    elif model_protocol == ModelProtocol.TORCH and requested_device == "cuda":
        print(f"5 - backend: {backend}, requested device: {requested_device}")
        torch.cuda.set_device(rank % torch.cuda.device_count())
        return torch.cuda.current_device()
    else:
        print(f"6 - backend: {backend}, requested device: {requested_device}")
        return torch.device("cpu")


def set_backend(requested_backend=None, device=None):
    global backend
    if(requested_backend != None):
        backend = requested_backend
        print(f"Backend: {backend} selected.")
    else:
        if device == None:
            print("---- Either parameters should have value, using Cupy as the backend ----")
            backend = mpi_mpi_cupy
        elif type(device) == cp.cuda.device.Device:
            backend = mpi_mpi_cupy
            print("---- Using Cupy as the backend ----")
        elif type(device) == torch.device and device.type == 'cuda':
            backend = mpi_mpi_torch
            print("---- Using Torch as the backend ----")
        elif type(device) == torch.device and device.type == 'cpu':
            backend = mpi_mpi_numpy
            print("---- Using Cupy as the backend ----")
        else:
            print("No valid device type detected, setting the backend to Numpy.")
            backend = mpi_mpi_numpy


def init():
    try:
        if os.environ["DISTDL_BACKEND"] == "Cupy":
            set_backend(requested_backend=mpi_mpi_cupy)
        elif os.environ["DISTDL_BACKEND"] == "Numpy":
            set_backend(requested_backend=mpi_mpi_numpy)
        elif os.environ["DISTDL_BACKEND"] == "Torch":
            set_backend(requested_backend=mpi_mpi_torch)
        else:
            set_backend(requested_backend=mpi_mpi_numpy)

    except:
        set_backend(requested_backend=mpi_mpi_numpy)
