# TODO: this source should move to backends package

import os
import distdl.backends.mpi_mpi_numpy as mpi_mpi_numpy
import distdl.backends.mpi_mpi_cupy as mpi_mpi_cupy
import distdl.backends.mpi_mpi_torch as mpi_mpi_torch
import torch
import cupy as cp

backend = mpi_mpi_numpy

try:
    if os.environ["DISTDL_BACKEND"] == "Cupy":
        backend = mpi_mpi_cupy
        print("---- Using Cupy as the backend ----")
    elif os.environ["DISTDL_BACKEND"] == "Numpy":
        backend = mpi_mpi_numpy
        print("---- Using Numpy as the backend ----")
    elif os.environ["DISTDL_BACKEND"] == "Torch":
        backend = mpi_mpi_torch
        print("---- Using Torch as the backend ----")
    else:
        backend = mpi_mpi_numpy
        print("---- Invalid Backend. Using Numpy for now. ----")

except:
    backend = mpi_mpi_numpy
    print("---- No backends specified, using Numpy for now ----")

# TODO: do we need sth like this?
def set_backend(device):
    global backend
    if type(device) == cp.cuda.device.Device:
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

