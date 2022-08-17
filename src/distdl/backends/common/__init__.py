from mpi4py import MPI as _MPI

from . import tensor_comm  # noqa: F401
from . import tensor_decomposition  # noqa: F401
#
# Expose the partition types
from .partition import MPICartesianPartition as CartesianPartition 
from .partition import MPIPartition as Partition
#
#
from .tensor_comm import assemble_global_tensor_structure
from .tensor_comm import broadcast_tensor_structure

operation_map = {
    "min": _MPI.MIN,
    "max": _MPI.MAX,
    "prod": _MPI.PROD,
    "sum": _MPI.SUM,
}
