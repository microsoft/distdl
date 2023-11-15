import numpy as np
import pytest
from adjoint_test import check_adjoint_test_tight

BACKEND_COMM = "mpi"
BACKEND_ARRAY = "numpy"

adjoint_parametrizations = []

# Main functionality
adjoint_parametrizations.append(
    pytest.param(
        np.arange(0, 2), [1, 2, 1],  # P_x_ranks, P_x_shape
        [1, 2, 1],     # x_global_shape
        [1, 2, 1],    # y_global_shape
        2,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-co3_ci2",
        marks=[pytest.mark.mpi(min_size=2)]
    )
)

adjoint_parametrizations.append(
    pytest.param(
        np.arange(0, 4), [1, 4, 1],  # P_x_ranks, P_x_shape
        [1, 12, 10],     # x_global_shape
        [1, 8, 10],    # y_global_shape
        4,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-co2_ci3",
        marks=[pytest.mark.mpi(min_size=4)]
    )
)


# For example of indirect, see https://stackoverflow.com/a/28570677
@pytest.mark.parametrize("P_x_ranks, P_x_shape,"
                         "x_global_shape,"
                         "y_global_shape,"
                         "comm_split_fixture",
                         adjoint_parametrizations,
                         indirect=["comm_split_fixture"])
def test_channel_conv1d_adjoint_input(barrier_fence_fixture,
                                      comm_split_fixture,
                                      P_x_ranks, P_x_shape,
                                      x_global_shape,
                                      y_global_shape
                                      ):

    import numpy as np
    import torch

    from distdl.backends.common.partition import MPIPartition
    from distdl.config import set_backend
    from distdl.nn.conv_channel_ag import DistributedChannelAllGatherConv1d
    from distdl.utilities.slicing import compute_subshape
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

    x_global_shape = np.asarray(x_global_shape)

    layer = DistributedChannelAllGatherConv1d(P_x,
                                              x_global_shape[1],
                                              y_global_shape[1],
                                              kernel_size=[3],
                                              padding=[1],
                                              device=P_x.device,
                                              bias=False
                                              ).to(P_world.device)

    x = zero_volume_tensor(x_global_shape[0], device=P_world.device)
    if P_x.active:
        x_local_shape = compute_subshape(P_x.shape, P_x.index, x_global_shape)
        y_local_shape = compute_subshape(P_x.shape, P_x.index, y_global_shape)
        x = torch.randn(*x_local_shape, device=P_world.device)
    x.requires_grad = True

    y = layer(x)

    dy = zero_volume_tensor(y_local_shape[0], device=P_world.device)
    if P_x.active:
        dy = torch.randn(*y_local_shape, device=P_world.device)

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


# For example of indirect, see https://stackoverflow.com/a/28570677
@pytest.mark.parametrize("P_x_ranks, P_x_shape,"
                         "x_global_shape,"
                         "y_global_shape,"
                         "comm_split_fixture",
                         adjoint_parametrizations,
                         indirect=["comm_split_fixture"])
def test_channel_conv1d_adjoint_weight(barrier_fence_fixture,
                                       comm_split_fixture,
                                       P_x_ranks, P_x_shape,
                                       x_global_shape,
                                       y_global_shape
                                       ):

    import numpy as np
    import torch

    from distdl.backends.common.partition import MPIPartition
    from distdl.config import set_backend
    from distdl.nn.conv_channel_ag import DistributedChannelAllGatherConv1d
    from distdl.utilities.slicing import compute_subshape
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

    x_global_shape = np.asarray(x_global_shape)

    layer = DistributedChannelAllGatherConv1d(P_x,
                                              x_global_shape[1],
                                              y_global_shape[1],
                                              kernel_size=[3],
                                              padding=[1],
                                              device=P_x.device,
                                              bias=False
                                              ).to(P_world.device)

    x = zero_volume_tensor(x_global_shape[0], device=P_world.device)
    if P_x.active:
        x_local_shape = compute_subshape(P_x.shape, P_x.index, x_global_shape)
        y_local_shape = compute_subshape(P_x.shape, P_x.index, y_global_shape)
        x = torch.randn(*x_local_shape, device=P_world.device)
    x.requires_grad = True

    y = layer(x)

    dy = zero_volume_tensor(y_local_shape[0], device=P_world.device)
    if P_x.active:
        dy = torch.randn(*y_local_shape, device=P_world.device)

    y.backward(dy)

    W = zero_volume_tensor()
    dW = zero_volume_tensor()
    if P_x.active:
        W = layer.weight.detach()
        dW = layer.weight.grad.detach()

    dy = dy.detach()
    y = y.detach()

    check_adjoint_test_tight(P_world, W, dW, y, dy)

    P_world.deactivate()
    P_x_base.deactivate()
    P_x.deactivate()


# For example of indirect, see https://stackoverflow.com/a/28570677
@pytest.mark.parametrize("P_x_ranks, P_x_shape,"
                         "x_global_shape,"
                         "y_global_shape,"
                         "comm_split_fixture",
                         adjoint_parametrizations,
                         indirect=["comm_split_fixture"])
def test_channel_conv1d_adjoint_bias(barrier_fence_fixture,
                                     comm_split_fixture,
                                     P_x_ranks, P_x_shape,
                                     x_global_shape,
                                     y_global_shape,
                                     ):

    import numpy as np
    import torch

    from distdl.backends.common.partition import MPIPartition
    from distdl.config import set_backend
    from distdl.nn.conv_channel_ag import DistributedChannelAllGatherConv1d
    from distdl.utilities.slicing import compute_subshape
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

    x_global_shape = np.asarray(x_global_shape)

    layer = DistributedChannelAllGatherConv1d(P_x,
                                              x_global_shape[1],
                                              y_global_shape[1],
                                              kernel_size=[3],
                                              padding=[1],
                                              device=P_x.device,
                                              bias=True
                                              ).to(P_world.device)
    layer.weight.data.fill_(0)

    x = zero_volume_tensor(x_global_shape[0], device=P_world.device)
    if P_x.active:
        x_local_shape = compute_subshape(P_x.shape, P_x.index, x_global_shape)
        y_local_shape = compute_subshape(P_x.shape, P_x.index, y_global_shape)
        x = torch.randn(*x_local_shape, device=P_world.device)
    x.requires_grad = True

    y = layer(x)

    dy = zero_volume_tensor(y_local_shape[0], device=P_world.device)
    if P_x.active:
        dy = torch.randn(*y_local_shape, device=P_world.device)

    y.backward(dy)

    b = zero_volume_tensor()
    db = zero_volume_tensor()
    if P_x.active and layer.use_bias:
        b = layer.bias.detach()
        db = layer.bias.grad.detach()

    dy = dy.detach()
    y = y.detach()

    check_adjoint_test_tight(P_world, b, db, y, dy)

    P_world.deactivate()
    P_x_base.deactivate()
    P_x.deactivate()
