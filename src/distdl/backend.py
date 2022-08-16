import os
import distdl.backends.mpi_mpi_numpy
import distdl.backends.mpi_mpi_cupy
import distdl.backends.mpi_mpi_torch

backend = distdl.backends.mpi_mpi_numpy

try:
    if os.environ["DISTDL_BACKEND"] == "Cupy":
        backend = distdl.backends.mpi_mpi_cupy
        print("---- Using Cupy ----")
    elif os.environ["DISTDL_BACKEND"] == "Numpy":
        backend = distdl.backends.mpi_mpi_numpy
        print("---- Using Numpy ----")
    elif os.environ["DISTDL_BACKEND"] == "Torch":
        backend = distdl.backends.mpi_mpi_torch
        print("---- Using Torch ----")
    else:
        backend = distdl.backends.mpi_mpi_numpy
        print("---- No backends specified, using Numpy for now ----")
      
except:
    backend = distdl.backends.mpi_mpi_numpy
    print("---- Invalid Backend. Using Numpy for now. ----")