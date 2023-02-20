import os, sys, pytest
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from adjoint_test import check_adjoint_test_tight
import numpy as np

BACKEND_COMM = "mpi"
BACKEND_ARRAY = "numpy"

adjoint_parametrizations = []

# Main functionality
adjoint_parametrizations.append(
    pytest.param(
        np.arange(0, 6), [2, 3],  # P_x_ranks, P_x_topo
        [4, 3],  # x_global_shape
        [2, 3], # y_global_shape
        (0,),  # axes_reduce_scatter
        6,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-2D-0D_reduction",
        marks=[pytest.mark.mpi(min_size=6)]
        )
    )

adjoint_parametrizations.append(
    pytest.param(
        np.arange(0, 6), [2, 3],  # P_x_ranks, P_x_topo
        [2, 9],  # x_global_shape
        [2, 3], # y_global_shape
        (1,),  # axes_reduce_scatter
        6,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-2D-1D_reduction",
        marks=[pytest.mark.mpi(min_size=6)]
        )
    )

adjoint_parametrizations.append(
    pytest.param(
        np.arange(0, 4), [4],  # P_x_ranks, P_x_topo
        [16],  # x_global_shape
        [4], # y_global_shape
        (0,),  # axes_gather
        4,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-2D-0D_reduction",
        marks=[pytest.mark.mpi(min_size=4)]
        )
    )

# For example of indirect, see https://stackoverflow.com/a/28570677
@pytest.mark.parametrize("P_x_ranks, P_x_shape,"
                         "x_global_shape,"
                         "y_global_shape,"
                         "axes_reduce_scatter,"
                         "comm_split_fixture",
                         adjoint_parametrizations,
                         indirect=["comm_split_fixture"])
def test_all_sum_reduce_adjoint(barrier_fence_fixture,
                                comm_split_fixture,
                                P_x_ranks, P_x_shape,
                                x_global_shape,
                                y_global_shape,
                                axes_reduce_scatter):

    import numpy as np
    import torch

    from distdl.backends.common.partition import MPIPartition
    from distdl.config import set_backend
    from distdl.nn.reduce_scatter import ReduceScatter
    from distdl.utilities.torch import zero_volume_tensor
    import distdl.utilities.slicing as slicing

    set_backend(backend_comm=BACKEND_COMM, backend_array=BACKEND_ARRAY)

    # Isolate the minimum needed ranks
    base_comm, active = comm_split_fixture
    if not active:
        return
    P_world = MPIPartition(base_comm)

    # Create the partitions
    P_x_base = P_world.create_partition_inclusive(P_x_ranks)
    P_x = P_x_base.create_cartesian_topology_partition(P_x_shape)


    # # have different shape.  Then, the output size will also be different, which
    # # we will have to get from `y` itself.
    x_local_shape = slicing.compute_subshape(P_x.shape, P_x.index, x_global_shape)
    y_local_shape = slicing.compute_subshape(P_x.shape, P_x.index, y_global_shape)

    # Layer
    layer = ReduceScatter(P_x, axes_reduce_scatter)
    layer = layer.to(P_x.device)

    x = zero_volume_tensor(device=P_x.device)
    if P_x.active:
        x = 10*torch.ones(*x_local_shape, device=P_x.device)
    x.requires_grad = True

    dy = zero_volume_tensor(device=P_x.device)
    if P_x.active:
        # Adjoint Input
        dy = 0.1*torch.ones(*y_local_shape, device=P_x.device)

    # y = F @ x
    y = layer(x)

    # dx = F* @ dy
    y.backward(dy)
    dx = x.grad

    x = x.detach()
    dx = dx.detach()
    dy = dy.detach()
    y = y.detach()

    check_adjoint_test_tight(P_world, x, dx, y, dy)

    P_world.deactivate()
    P_x_base.deactivate()
    P_x.deactivate()