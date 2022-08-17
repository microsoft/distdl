__version__ = '0.5.0-dev'

import os
import distdl.nn  # noqa: F401

from . import backends  # noqa: F401
from . import nn  # noqa: F401
from . import utilities  # noqa: F401
from . import backend

from .backend import set_backend

try:
    if os.environ["DISTDL_BACKEND"] == "Cupy":
        set_backend(requested_backend=backends.mpi_mpi_cupy)
        print("---- Using Cupy as the backend ----")
    elif os.environ["DISTDL_BACKEND"] == "Numpy":
        set_backend(requested_backend=backends.mpi_mpi_numpy)
        print("---- Using Numpy as the backend ----")
    elif os.environ["DISTDL_BACKEND"] == "Torch":
        set_backend(requested_backend=backends.mpi_mpi_torch)
        print("---- Using Torch as the backend ----")
    else:
        set_backend(requested_backend=backends.mpi_mpi_numpy)
        print("---- Invalid Backend. Using Numpy for now. ----")

except:
    set_backend(requested_backend=backends.mpi_mpi_numpy)
    print("---- No backends specified, using Numpy for now ----")
