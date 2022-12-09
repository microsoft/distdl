__version__ = '0.5.0-dev'

from .logger import logger
from . import config

def init_distdl(backend_comm="MPI", backend_array="NUMPY"):
    global backends
    config.set_config(backend_comm, backend_array)
    backends = config.load_backend_module(current_backend=backends)

# Select and import backend
backends = None
init_distdl(*config.get_default_config())

# Import remaining modules
from . import nn
from . import utilities