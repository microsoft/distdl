from mpi4py import MPI as _MPI

from . import tensor_comm
from . import tensor_decomposition

from .partition import MPICartesianPartition as CartesianPartition 
from .partition import MPIPartition as Partition

from .tensor_comm import assemble_global_tensor_structure
from .tensor_comm import broadcast_tensor_structure

operation_map = {
    "min": _MPI.MIN,
    "max": _MPI.MAX,
    "prod": _MPI.PROD,
    "sum": _MPI.SUM,
}