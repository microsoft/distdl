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
for key in supported_backends.keys():
    try:
        supported_backends[key] = importlib.import_module('distdl.backends.' + key)
    except:
        logger.warning("Could not load {} backend.".format(key))