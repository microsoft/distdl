import os, torch

import numpy as np
import pytest
from adjoint_test import check_adjoint_test_tight

torch.set_printoptions(precision=8)
torch.manual_seed(0)

BACKEND_COMM = "nccl"
BACKEND_ARRAY = "cupy"

adjoint_parametrizations = []

# Main functionality
adjoint_parametrizations.append(
    pytest.param(
        np.arange(0, 1), [1, 1, 1],  # P_x_ranks, P_x_shape
        [1, 8, 12],  # x_global_shape
        [1, 8, 16],  # y_global_shape
        1,  # passed to comm_split_fixture, required MPI ranks
        id="serial",
        marks=[pytest.mark.mpi(min_size=1)]
        )
    )

adjoint_parametrizations.append(
    pytest.param(
        np.arange(0, 4), [1, 1, 4],  # P_x_ranks, P_x_shape
        [1, 8, 12],  # x_global_shape
        [1, 8, 16],  # y_global_shape
        4,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-3d-a",
        marks=[pytest.mark.mpi(min_size=4)]
        )
    )

adjoint_parametrizations.append(
    pytest.param(
        np.arange(0, 8), [2, 1, 4],  # P_x_ranks, P_x_shape
        [2, 4, 12],  # x_global_shape
        [2, 4, 16],  # y_global_shape
        8,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-3d-b",
        marks=[pytest.mark.mpi(min_size=8)]
        )
    )

adjoint_parametrizations.append(
    pytest.param(
        np.arange(0, 4), [4, 1, 1],  # P_x_ranks, P_x_shape
        [4, 4, 12],  # x_global_shape
        [4, 4, 16],  # y_global_shape
        4,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-3d-c",
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
def test_linear_adjoint_input(barrier_fence_fixture,
                              comm_split_fixture,
                              P_x_ranks, P_x_shape,
                              x_global_shape,
                              y_global_shape):

    import numpy as np
    import torch

    from distdl.backends.common.partition import MPIPartition
    from distdl.utilities.slicing import compute_subshape
    from distdl.utilities.torch import zero_volume_tensor
    from distdl.config import set_backend
    from distdl.nn.linear_rs_expert import DistributedExpertReduceScatter

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
    y_global_shape = np.asarray(y_global_shape)

    layer = DistributedExpertReduceScatter(P_x,
                                           P_x.shape[0],
                                           x_global_shape[-1],
                                           y_global_shape[-1],
                                           bias=False)
    layer = layer.to(P_world.device)

    x = zero_volume_tensor(x_global_shape[0], device=P_world.device)
    if P_x.active:
        x_local_shape = compute_subshape(P_x.shape,
                                         P_x.index,
                                         x_global_shape)
        x = torch.randn(*x_local_shape, device=P_world.device)
    x.requires_grad = True

    y = layer(x)

    dy = zero_volume_tensor(x_global_shape[0], device=P_world.device)
    if P_x.active:
        dy = torch.randn(*y.shape, device=P_world.device)

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
def test_linear_adjoint_weight(barrier_fence_fixture,
                               comm_split_fixture,
                               P_x_ranks, P_x_shape,
                               x_global_shape,
                               y_global_shape):

    import numpy as np
    import torch

    from distdl.backends.common.partition import MPIPartition
    from distdl.utilities.slicing import compute_subshape
    from distdl.utilities.torch import zero_volume_tensor
    from distdl.config import set_backend
    from distdl.nn.linear_rs_expert import DistributedExpertReduceScatter

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
    y_global_shape = np.asarray(y_global_shape)

    layer = DistributedExpertReduceScatter(P_x,
                                           P_x.shape[0],
                                           x_global_shape[-1],
                                           y_global_shape[-1],
                                           bias=False)
    layer = layer.to(P_x.device)

    x = zero_volume_tensor(x_global_shape[0], device=P_x.device)
    if P_x.active:
        x_local_shape = compute_subshape(P_x.shape,
                                         P_x.index,
                                         x_global_shape)
        x = torch.randn(*x_local_shape, device=P_x.device)
    x.requires_grad = True

    y = layer(x)

    dy = zero_volume_tensor(x_global_shape[0], device=P_x.device)
    if P_x.active:
        dy = torch.randn(*y.shape, device=P_x.device)

    y.backward(dy)

    W = zero_volume_tensor(device=P_x.device)
    dW = zero_volume_tensor(device=P_x.device)
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
def test_linear_adjoint_bias(barrier_fence_fixture,
                             comm_split_fixture,
                             P_x_ranks, P_x_shape,
                             x_global_shape,
                             y_global_shape):

    import numpy as np
    import torch

    from distdl.backends.common.partition import MPIPartition
    from distdl.utilities.slicing import compute_subshape
    from distdl.utilities.torch import zero_volume_tensor
    from distdl.config import set_backend
    from distdl.nn.linear_rs_expert import DistributedExpertReduceScatter

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
    y_global_shape = np.asarray(y_global_shape)

    layer = DistributedExpertReduceScatter(P_x,
                                           P_x.shape[0],
                                           x_global_shape[-1],
                                           y_global_shape[-1],
                                           bias=True)
    layer = layer.to(P_x.device)
    layer.weight.data.fill_(0)

    x = zero_volume_tensor(x_global_shape[0], device=P_x.device)
    if P_x.active:
        x_local_shape = compute_subshape(P_x.shape,
                                         P_x.index,
                                         x_global_shape)
        # For this test, we are only testing to see if the adjoint works
        # correctly for the bias term.  But the adjoint test only works on the
        # Jacobian of the linear layer.  The Jacobian block for b is 0 for x and
        # W, so killing x makes the forward operator equal to its Jacobian and
        # we can test to see that adjoint is computed correctly.
        x = torch.zeros(*x_local_shape, device=P_x.device)
    x.requires_grad = True

    y = layer(x)

    dy = zero_volume_tensor(x_global_shape[0], device=P_x.device)
    if P_x.active:
        dy = torch.randn(*y.shape, device=P_x.device)

    y.backward(dy)

    b = zero_volume_tensor(device=P_x.device)
    db = zero_volume_tensor(device=P_x.device)
    if layer.P_bias.active:
        b = layer.bias.detach()
        db = layer.bias.grad.detach()

    dy = dy.detach()
    y = y.detach()

    check_adjoint_test_tight(P_world, b, db, y, dy)

    P_world.deactivate()
    P_x_base.deactivate()
    P_x.deactivate()