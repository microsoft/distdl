import importlib

from .. import logger
from . import common  # noqa: F401

# Supported distdl backends
supported_backends = {
    'mpi_numpy': None,
    'mpi_cupy': None,
    'nccl_cupy': None
}

# Load backends that are locally supported
for backend in supported_backends.keys():
    try:
        supported_backends[backend] = importlib.import_module('distdl.backends.' + backend)
    except Exception as e:
        logger.warning("Could not load {} backend.".format(backend))
        logger.warning(e)
