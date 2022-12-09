from .. import config
from . import common

print("Set backend to {}".format(config.name))

# Select backend
if config.name == 'mpi_numpy':
    from . import mpi_mpi_numpy as backend # noqa: F401
elif config.name == 'mpi_cupy':
    from . import mpi_mpi_cupy as backend  # noqa: F401
elif config.name == 'nccl_cupy':
    from . import mpi_nccl_cupy as backend # noqa: F401
elif config.name == 'mpi_torch':
    from . import mpi_mpi_torch as backend # noqa: F401