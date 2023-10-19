__version__ = '0.5.0-dev'

from .logger import logger
from . import backends
from . import config

# Select default backend
config.set_backend()

# Import remaining modules
from . import nn
from . import utilities
