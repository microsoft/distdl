from . import common
from .. import logger
import importlib


# Supported distdl backends
supported_backends = {
    'mpi_numpy': None,
    'mpi_cupy': None,
    'nccl_cupy': None,
    'mpi_torch': None
}

# Load backends that are locally supported
for backend in supported_backends.keys():
    try:
        supported_backends[backend] = importlib.import_module('distdl.backends.' + backend)
    except:
        logger.warning("Could not load {} backend.".format(backend))