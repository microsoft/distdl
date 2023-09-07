from . import functional
from . import nccl_comm

from ..common import buffer_allocator
from ..common import CartesianPartition
from ..common import Partition

from ..common import assemble_global_tensor_structure
from ..common import assemble_global_tensor_structure_along_axis
from ..common import broadcast_tensor_structure

from ..common import tensor_decomposition as tensor_decomposition
from ..common import tensor_comm as tensor_comm

from .buffer_cupy import MPICupyBufferManager as BufferManager
from .buffer_cupy import MPIExpandableCupyBuffer as ExpandableBuffer
from .device import set_device, get_device

__name__ = 'nccl_cupy'

# Dictionary of active nccl communicators
comm_store_backend = {}
comm_store_frontend = {}