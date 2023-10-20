__version__ = '0.5.0-dev'

# Import remaining modules
from . import backends  # noqa: F401
from . import config
from . import nn  # noqa: F401
from . import utilities  # noqa: F401
from .logger import logger  # noqa: F401

# Select default backend
config.set_backend()
