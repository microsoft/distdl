import numpy as np
import pytest
import torch
from adjoint_test import check_adjoint_test_tight

BACKEND_COMM = "mpi"
BACKEND_ARRAY = "numpy"

adjoint_parametrizations = []

# Main functionality
adjoint_parametrizations.append(
    pytest.param(
        np.arange(4, 8), [2, 2, 1],  # P_x_ranks, P_x_shape
        np.arange(0, 12), [2, 2, 3],  # P_y_ranks, P_y_shape
        [1, 7, 5],  # x_global_shape
        False,  # transpose_src
        12,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-overlap-3D",
        marks=[pytest.mark.mpi(min_size=12)]
    )
)

adjoint_parametrizations.append(
    pytest.param(
        np.arange(0, 4), [2, 2, 1],  # P_x_ranks, P_x_shape
        np.arange(4, 16), [2, 2, 3],  # P_y_ranks, P_y_shape
        [1, 7, 5],  # x_global_shape
        False,  # transpose_src
        16,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-disjoint-3D",
        marks=[pytest.mark.mpi(min_size=16)]
    )
)

adjoint_parametrizations.append(
    pytest.param(
        np.arange(0, 4), [2, 2, 1],  # P_x_ranks, P_x_shape
        np.arange(5, 17), [2, 2, 3],  # P_y_ranks, P_y_shape
        [1, 7, 5],  # x_global_shape
        False,  # transpose_src
        17,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-disjoint-inactive-3D",
        marks=[pytest.mark.mpi(min_size=17)]
    )
)

# Sequential functionality
adjoint_parametrizations.append(
    pytest.param(
        np.arange(0, 1), [1],  # P_x_ranks, P_x_shape
        np.arange(0, 1), [1],  # P_y_ranks, P_y_shape
        [1, 7, 5],  # x_global_shape
        False,  # transpose_src
        1,  # passed to comm_split_fixture, required MPI ranks
        id="sequential-identity",
        marks=[pytest.mark.mpi(min_size=1)]
    )
)

# Main functionality, single source
adjoint_parametrizations.append(
    pytest.param(
        np.arange(2, 3), [1],  # P_x_ranks, P_x_shape
        np.arange(0, 3), [1, 1, 3],  # P_y_ranks, P_y_shape
        [1, 7, 5],  # x_global_shape
        False,  # transpose_src
        3,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-overlap-3D-single_source",
        marks=[pytest.mark.mpi(min_size=3)]
    )
)

adjoint_parametrizations.append(
    pytest.param(
        np.arange(3, 4), [1],  # P_x_ranks, P_x_shape
        np.arange(0, 3), [1, 1, 3],  # P_y_ranks, P_y_shape
        [1, 7, 5],  # x_global_shape
        False,  # transpose_src
        4,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-disjoint-3D-single_source",
        marks=[pytest.mark.mpi(min_size=4)]
    )
)

adjoint_parametrizations.append(
    pytest.param(
        np.arange(4, 5), [1],  # P_x_ranks, P_x_shape
        np.arange(0, 3), [1, 1, 3],  # P_y_ranks, P_y_shape
        [1, 7, 5],  # x_global_shape
        False,  # transpose_src
        5,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-disjoint-inactive-3D-single_source",
        marks=[pytest.mark.mpi(min_size=5)]
    )
)

# Main functionality, transposed
adjoint_parametrizations.append(
    pytest.param(
        np.arange(4, 8), [2, 2, 1],  # P_x_ranks, P_x_shape
        np.arange(0, 12), [3, 2, 2],  # P_y_ranks, P_y_shape
        [1, 7, 5],  # x_global_shape
        True,  # transpose_src
        12,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-overlap-3D",
        marks=[pytest.mark.mpi(min_size=12)]
    )
)

adjoint_parametrizations.append(
    pytest.param(
        np.arange(0, 4), [2, 2, 1],  # P_x_ranks, P_x_shape
        np.arange(4, 16), [3, 2, 2],  # P_y_ranks, P_y_shape
        [1, 7, 5],  # x_global_shape
        True,  # transpose_src
        16,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-disjoint-3D",
        marks=[pytest.mark.mpi(min_size=16)]
    )
)

adjoint_parametrizations.append(
    pytest.param(
        np.arange(0, 4), [2, 2, 1],  # P_x_ranks, P_x_shape
        np.arange(5, 17), [3, 2, 2],  # P_y_ranks, P_y_shape
        [1, 7, 5],  # x_global_shape
        True,  # transpose_src
        17,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-disjoint-inactive-3D",
        marks=[pytest.mark.mpi(min_size=17)]
    )
)


# For example of indirect, see https://stackoverflow.com/a/28570677
@pytest.mark.parametrize("P_x_ranks, P_x_shape,"
                         "P_y_ranks, P_y_shape,"
                         "x_global_shape,"
                         "transpose_src,"
                         "comm_split_fixture",
                         adjoint_parametrizations,
                         indirect=["comm_split_fixture"])
def test_broadcast_adjoint(barrier_fence_fixture,
                           comm_split_fixture,
                           P_x_ranks, P_x_shape,
                           P_y_ranks, P_y_shape,
                           x_global_shape,
                           transpose_src):

    import numpy as np
    import torch

    from distdl.backends.common.partition import MPIPartition
    from distdl.config import set_backend
    from distdl.nn.broadcast import Broadcast
    from distdl.utilities.torch import zero_volume_tensor

    set_backend(backend_comm=BACKEND_COMM, backend_array=BACKEND_ARRAY)

    # Isolate the minimum needed ranks
    base_comm, active = comm_split_fixture
    if not active:
        return
    P_world = MPIPartition(base_comm)

    # Create the partitions
    P_x_base = P_world.create_partition_inclusive(P_x_ranks)
    P_x = P_x_base.create_cartesian_topology_partition(P_x_shape)

    P_y_base = P_world.create_partition_inclusive(P_y_ranks)
    P_y = P_y_base.create_cartesian_topology_partition(P_y_shape)

    # TODO #93: Change this to create a subtensor so we test when local tensors
    # have different shape.  Then, the output size will also be different, which
    # we will have to get from `y` itself.
    x_local_shape = np.asarray(x_global_shape)

    layer = Broadcast(P_x, P_y, transpose_src=transpose_src, preserve_batch=False)
    layer = layer.to(P_x.device)

    x = zero_volume_tensor(device=P_x.device)
    if P_x.active:
        x = torch.randn(*x_local_shape, device=P_x.device)
    x.requires_grad = True

    dy = zero_volume_tensor(device=P_x.device)
    if P_y.active:
        # Adjoint Input
        dy = torch.randn(*x_local_shape, device=P_x.device)

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
    P_y_base.deactivate()
    P_y.deactivate()


deadlock_parametrizations = []

# These cases test for a situation where mpi_comm_create_group deadlocked
deadlock_parametrizations.append(
    pytest.param(
        np.arange(1, 3), [1, 2],  # P_x_ranks, P_x_shape
        np.arange(0, 4), [2, 2],  # P_y_ranks, P_y_shape
        4,  # passed to comm_split_fixture, required MPI ranks
        id="deadlock-2D",
        marks=[pytest.mark.mpi(min_size=4)]
    )
)

deadlock_parametrizations.append(
    pytest.param(
        np.arange(2, 6), [1, 2, 2],  # P_x_ranks, P_x_shape
        np.arange(0, 8), [2, 2, 2],  # P_y_ranks, P_y_shape
        8,  # passed to comm_split_fixture, required MPI ranks
        id="deadlock-3D",
        marks=[pytest.mark.mpi(min_size=8)]
    )
)

deadlock_parametrizations.append(
    pytest.param(
        np.arange(4, 12), [1, 2, 2, 2],  # P_x_ranks, P_x_shape
        np.arange(0, 16), [2, 2, 2, 2],  # P_y_ranks, P_y_shape
        16,  # passed to comm_split_fixture, required MPI ranks
        id="deadlock-4D",
        marks=[pytest.mark.mpi(min_size=16)]
    )
)


@pytest.mark.parametrize("P_x_ranks, P_x_shape,"
                         "P_w_ranks, P_w_shape,"
                         "comm_split_fixture",
                         deadlock_parametrizations,
                         indirect=["comm_split_fixture"])
def test_potentially_deadlocked_send_recv_pairs(barrier_fence_fixture,
                                                comm_split_fixture,
                                                P_x_ranks, P_x_shape,
                                                P_w_ranks, P_w_shape):

    from distdl.backends.common.partition import MPIPartition
    from distdl.config import set_backend
    from distdl.nn.broadcast import Broadcast

    set_backend(backend_comm=BACKEND_COMM, backend_array=BACKEND_ARRAY)

    # Isolate the minimum needed ranks
    base_comm, active = comm_split_fixture
    if not active:
        return
    P_world = MPIPartition(base_comm)

    # Create the partitions
    P_x_base = P_world.create_partition_inclusive(P_x_ranks)
    P_x = P_x_base.create_cartesian_topology_partition(P_x_shape)

    P_w_base = P_world.create_partition_inclusive(P_w_ranks)
    P_w = P_w_base.create_cartesian_topology_partition(P_w_shape)

    layer = Broadcast(P_x, P_w)  # noqa F841
    layer = layer.to(P_x.device)

    P_world.deactivate()
    P_x_base.deactivate()
    P_x.deactivate()
    P_w_base.deactivate()
    P_w.deactivate()


dtype_parametrizations = []

# Main functionality
dtype_parametrizations.append(
    pytest.param(
        torch.float32, True,  # dtype, test_backward,
        np.arange(4, 8), [2, 2, 1],  # P_x_ranks, P_x_shape
        np.arange(0, 12), [2, 2, 3],  # P_y_ranks, P_y_shape
        [1, 7, 5],  # x_global_shape
        False,  # transpose_src
        12,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-dtype-float32",
        marks=[pytest.mark.mpi(min_size=12)]
    )
)

# Test that it works with ints as well, can't compute gradient here
dtype_parametrizations.append(
    pytest.param(
        torch.int32, False,  # dtype, test_backward,
        np.arange(4, 8), [2, 2, 1],  # P_x_ranks, P_x_shape
        np.arange(0, 12), [2, 2, 3],  # P_y_ranks, P_y_shape
        [1, 7, 5],  # x_global_shape
        False,  # transpose_src
        12,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-dtype-int32",
        marks=[pytest.mark.mpi(min_size=12)]
    )
)

# Also test doubles
dtype_parametrizations.append(
    pytest.param(
        torch.float64, True,  # dtype, test_backward,
        np.arange(4, 8), [2, 2, 1],  # P_x_ranks, P_x_shape
        np.arange(0, 12), [2, 2, 3],  # P_y_ranks, P_y_shape
        [1, 7, 5],  # x_global_shape
        False,  # transpose_src
        12,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-dtype-float64",
        marks=[pytest.mark.mpi(min_size=12)]
    )
)


# For example of indirect, see https://stackoverflow.com/a/28570677
@pytest.mark.parametrize("dtype, test_backward,"
                         "P_x_ranks, P_x_shape,"
                         "P_y_ranks, P_y_shape,"
                         "x_global_shape,"
                         "transpose_src,"
                         "comm_split_fixture",
                         dtype_parametrizations,
                         indirect=["comm_split_fixture"])
def test_broadcast_dtype(barrier_fence_fixture,
                         comm_split_fixture,
                         dtype, test_backward,
                         P_x_ranks, P_x_shape,
                         P_y_ranks, P_y_shape,
                         x_global_shape,
                         transpose_src):

    import numpy as np
    import torch

    from distdl.backends.common.partition import MPIPartition
    from distdl.config import set_backend
    from distdl.nn.broadcast import Broadcast
    from distdl.utilities.torch import zero_volume_tensor

    set_backend(backend_comm=BACKEND_COMM, backend_array=BACKEND_ARRAY)

    # Isolate the minimum needed ranks
    base_comm, active = comm_split_fixture
    if not active:
        return
    P_world = MPIPartition(base_comm)

    # Create the partitions
    P_x_base = P_world.create_partition_inclusive(P_x_ranks)
    P_x = P_x_base.create_cartesian_topology_partition(P_x_shape)

    P_y_base = P_world.create_partition_inclusive(P_y_ranks)
    P_y = P_y_base.create_cartesian_topology_partition(P_y_shape)

    # TODO #93: Change this to create a subtensor so we test when local tensors
    # have different shape.  Then, the output size will also be different, which
    # we will have to get from `y` itself.
    x_local_shape = np.asarray(x_global_shape)

    layer = Broadcast(P_x, P_y, transpose_src=transpose_src, preserve_batch=False)
    layer = layer.to(P_x.device)

    x = zero_volume_tensor(device=P_x.device)
    if P_x.active:
        x = 10 * torch.randn(*x_local_shape).to(dtype)
        x = x.to(P_x.device)

    x.requires_grad = test_backward

    # y = F @ x
    y = layer(x)

    # If we are not in the output partition, there is no data to test the type
    # against.
    if P_y.active:
        assert y.dtype == dtype

    if test_backward:
        dy = zero_volume_tensor(device=P_x.device)
        if P_y.active:
            # Adjoint Input
            dy = 10 * torch.randn(*x_local_shape).to(dtype)
            dy = dy.to(P_x.device)

        # dx = F* @ dy
        y.backward(dy)
        dx = x.grad

        if P_x.active:
            assert dx.dtype == dtype

    P_world.deactivate()
    P_x_base.deactivate()
    P_x.deactivate()
    P_y_base.deactivate()
    P_y.deactivate()
