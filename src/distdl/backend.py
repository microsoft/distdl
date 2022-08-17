# TODO: this source should move to backends package

import os
import distdl.backends.mpi_mpi_numpy as mpi_mpi_numpy
import distdl.backends.mpi_mpi_cupy as mpi_mpi_cupy
import distdl.backends.mpi_mpi_torch as mpi_mpi_torch
import torch
import cupy as cp

backend = None


def get_backend():
    global backend
    if backend == None:
        init()
    return backend


def get_device(requested_device=None, rank=None):
    global backend
    if backend == None:
        init()
    if backend == mpi_mpi_cupy:
        # TODO: handle mapping configuration from user input
        cp.cuda.runtime.setDevice(rank % cp.cuda.runtime.getDeviceCount())
        return cp.cuda.runtime.getDevice()
    elif backend == mpi_mpi_numpy:
        return torch.device("cpu")
    elif backend == mpi_mpi_torch and requested_device == None:
        torch.cuda.set_device(rank % torch.cuda.device_count())
        return torch.cuda.current_device()
    elif backend == mpi_mpi_torch and requested_device == "cpu":
        return torch.device("cpu")
    elif backend == mpi_mpi_torch and requested_device == "cuda":
        torch.cuda.set_device(rank % torch.cuda.device_count())
        return torch.cuda.current_device()
    else:
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
