# TODO: this source should move to backends package

import os
import distdl.backends.mpi_mpi_numpy as mpi_mpi_numpy
import distdl.backends.mpi_mpi_cupy as mpi_mpi_cupy
import distdl.backends.mpi_mpi_torch as mpi_mpi_torch

backend = mpi_mpi_numpy

try:
    if os.environ["DISTDL_BACKEND"] == "Cupy":
        backend = mpi_mpi_cupy
        print("---- Using Cupy ----")
    elif os.environ["DISTDL_BACKEND"] == "Numpy":
        backend = mpi_mpi_numpy
        print("---- Using Numpy ----")
    elif os.environ["DISTDL_BACKEND"] == "Torch":
        backend = mpi_mpi_torch
        print("---- Using Torch ----")
    else:
        backend = mpi_mpi_numpy
        print("---- No backends specified, using Numpy for now ----")
      
except:
    backend = mpi_mpi_numpy
    print("---- Invalid Backend. Using Numpy for now. ----")