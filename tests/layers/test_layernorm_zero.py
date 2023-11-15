import numpy as np
import pytest
import torch

from distdl.nn.layernorm_zero import DistributedLayerNormZero
from distdl.nn.repartition import Repartition
from distdl.utilities.torch import zero_volume_tensor

ERROR_THRESHOLD = 1e-4
BACKEND_COMM = "mpi"
BACKEND_ARRAY = "numpy"

parametrizations_affine = []

parametrizations_affine.append(
    pytest.param(
        np.arange(0, 4), [2, 1, 2],  # P_x_ranks, P_x_shape,
        (4, 3, 12),  # input_shape
        (3, 12),    # normalized_shape
        True,  # elementwise_affine
        4,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-batch-norm-affine-batch-0",
        marks=[pytest.mark.mpi(min_size=4)]
        )
    )

parametrizations_affine.append(
    pytest.param(
        np.arange(0, 4), [2, 1, 2],  # P_x_ranks, P_x_shape,
        (4, 3, 12),  # input_shape
        (12,),    # normalized_shape
        True,  # elementwise_affine
        4,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-batch-norm-affine-batch-1",
        marks=[pytest.mark.mpi(min_size=4)]
        )
    )

parametrizations_affine.append(
    pytest.param(
        np.arange(0, 4), [1, 2, 2],  # P_x_ranks, P_x_shape,
        (3, 4, 12),  # input_shape
        (4, 12),    # normalized_shape
        True,  # elementwise_affine
        4,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-batch-norm-affine-batch-2",
        marks=[pytest.mark.mpi(min_size=4)]
        )
    )

parametrizations_affine.append(
    pytest.param(
        np.arange(0, 2), [2, 1, 1],  # P_x_ranks, P_x_shape,
        (4, 3, 8),  # input_shape
        (3, 8),    # normalized_shape
        True,  # elementwise_affine
        2,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-batch-norm-affine-batch-3",
        marks=[pytest.mark.mpi(min_size=4)]
        )
    )

parametrizations_affine.append(
    pytest.param(
        np.arange(0, 4), [2, 2],  # P_x_ranks, P_x_shape,
        (4, 8),  # input_shape
        (4, 8),    # normalized_shape
        True,  # elementwise_affine
        4,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-batch-norm-affine-batch-4",
        marks=[pytest.mark.mpi(min_size=4)]
        )
    )

parametrizations_affine.append(
    pytest.param(
        np.arange(0, 4), [1, 4],  # P_x_ranks, P_x_shape,
        (4, 8),  # input_shape
        (4, 8),    # normalized_shape
        True,  # elementwise_affine
        4,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-batch-norm-affine-batch-5",
        marks=[pytest.mark.mpi(min_size=4)]
        )
    )

parametrizations_affine.append(
    pytest.param(
        np.arange(0, 4), [4, 1],  # P_x_ranks, P_x_shape,
        (4, 8),  # input_shape
        (4, 8),    # normalized_shape
        True,  # elementwise_affine
        4,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-batch-norm-affine-batch-6",
        marks=[pytest.mark.mpi(min_size=4)]
        )
    )

parametrizations_affine.append(
    pytest.param(
        np.arange(0, 4), [2, 2],  # P_x_ranks, P_x_shape,
        (4, 8),  # input_shape
        (8,),    # normalized_shape
        True,  # elementwise_affine
        4,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-batch-norm-affine-batch-7",
        marks=[pytest.mark.mpi(min_size=4)]
        )
    )

parametrizations_affine.append(
    pytest.param(
        np.arange(0, 4), [1, 4],  # P_x_ranks, P_x_shape,
        (4, 8),  # input_shape
        (8,),    # normalized_shape
        True,  # elementwise_affine
        4,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-batch-norm-affine-batch-8",
        marks=[pytest.mark.mpi(min_size=4)]
        )
    )

parametrizations_affine.append(
    pytest.param(
        np.arange(0, 4), [4, 1],  # P_x_ranks, P_x_shape,
        (4, 8),  # input_shape
        (8,),    # normalized_shape
        True,  # elementwise_affine
        4,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-batch-norm-affine-batch-9",
        marks=[pytest.mark.mpi(min_size=4)]
        )
    )


# TODO This test is broken when normalized shape has more than 1 dimension
@pytest.mark.parametrize("P_x_ranks, P_x_shape,"
                         "input_shape,"
                         "normalized_shape,"
                         "elementwise_affine,"
                         "comm_split_fixture",
                         parametrizations_affine,
                         indirect=["comm_split_fixture"])
def test_batch_norm_with_training(barrier_fence_fixture,
                                  P_x_ranks, P_x_shape,
                                  input_shape,
                                  normalized_shape,
                                  elementwise_affine,
                                  comm_split_fixture):

    from distdl.backends.common.partition import MPIPartition
    from distdl.config import set_backend

    torch.manual_seed(0)

    set_backend(backend_comm=BACKEND_COMM, backend_array=BACKEND_ARRAY)

    # Isolate the minimum needed ranks
    base_comm, active = comm_split_fixture
    if not active:
        return
    P_world = MPIPartition(base_comm)

    # Create the partitions
    num_dimensions = len(input_shape)
    P_root_base = P_world.create_partition_inclusive([0])
    P_root = P_root_base.create_cartesian_topology_partition([1] * num_dimensions)
    P_x_base = P_world.create_partition_inclusive(P_x_ranks)
    P_x = P_x_base.create_cartesian_topology_partition(P_x_shape)

    # Create the input
    if P_root.active:
        input_train = torch.rand(input_shape, dtype=torch.float32, device=P_x.device)
        input_eval = torch.rand(input_shape, dtype=torch.float32, device=P_x.device)
        output = torch.rand(input_shape, dtype=torch.float32, device=P_x.device)
    else:
        input_train = zero_volume_tensor(device=P_x.device)
        input_eval = zero_volume_tensor(device=P_x.device)
        output = zero_volume_tensor(device=P_x.device)

    # Scatter/gather data
    scatter = Repartition(P_root, P_x)
    gather = Repartition(P_x, P_root)

    # Sequential layer
    seq_ln = torch.nn.LayerNorm(normalized_shape,
                                elementwise_affine=elementwise_affine
                                ).to(P_x.device)

    # Train sequential layer
    if P_root.active:
        seq_ln.train()
        seq_out1 = seq_ln(input_train)
        seq_loss = ((seq_out1 - output)**2).sum()
        seq_loss.backward()

        # Do a manual weight update (this is what optimizer does):
        with torch.no_grad():
            for p in seq_ln.parameters():
                p.copy_(p + 0.1*p.grad)

    # Evaluate sequential network
    if P_root.active:
        seq_ln.eval()
        seq_out2 = seq_ln(input_eval)
        seq_out2 = seq_out2.detach()
    if P_root.active:
        print("Seq weight: ", seq_ln.weight, P_root.rank)

    # Create distributed network
    dist_ln = DistributedLayerNormZero(P_x, normalized_shape,
                                       elementwise_affine=elementwise_affine
                                       ).to(P_x.device)

    # Train distributed network
    dist_ln.train()
    dist_out1 = gather(dist_ln(scatter(input_train)))
    dist_loss = ((dist_out1 - output)**2).sum()
    assert dist_loss.requires_grad
    dist_loss.backward()

    # Do a manual gradient update
    if P_x.active:
        with torch.no_grad():
            for p in dist_ln.parameters():
                p.copy_(p + 0.1*p.grad)

    # if P_root.active:
    #     print("Seq weight: ", seq_ln.weight, P_root.rank)
    # print("Dist weight: ", dist_ln.weight, P_x.rank)

    # Evaluate distributed network
    dist_ln.eval()
    dist_out2 = gather(dist_ln(scatter(input_eval)))
    dist_out2 = dist_out2.detach()

    # Compare the distributed and sequential networks
    if P_world.rank == 0:

        # Set the absolute tolerance to ~sqrt(e_mach), or the default
        # Pytorch got their defaults from NumPy, but NumPy defaults to 64-bit
        # floats, not 32-bit floats as torch does.  Consequently, the default
        # torch atol is actually tighter than one can expect from two fp-equal
        # floating point numbers.  The NumPy default of 1e-8 is closer to
        # sqrt(e_mach) for 64-bit numbers.  So we set the 32-bit tolerance to
        # sqrt(1e-7), as our usual choice, 1e-5, is too tight.
        if seq_out1.dtype == torch.float64:
            atol = 1e-8
        elif seq_out1.dtype == torch.float32:
            import math
            atol = math.sqrt(1e-7)
        else:
            # torch default
            atol = 1e-8

        assert dist_out1.shape == seq_out1.shape
        assert torch.allclose(dist_out1, seq_out1, rtol=ERROR_THRESHOLD, atol=atol)
        assert dist_loss.shape == seq_loss.shape
        assert torch.allclose(dist_loss, seq_loss, rtol=ERROR_THRESHOLD, atol=atol)
        assert dist_out2.shape == seq_out2.shape
        # assert torch.allclose(dist_out2, seq_out2, rtol=ERROR_THRESHOLD, atol=atol)

    P_world.deactivate()
    P_root_base.deactivate()
    P_root.deactivate()
    P_x_base.deactivate()
    P_x.deactivate()


@pytest.mark.parametrize("P_x_ranks, P_x_shape,"
                         "input_shape,"
                         "normalized_shape,"
                         "elementwise_affine,"
                         "comm_split_fixture",
                         parametrizations_affine,
                         indirect=["comm_split_fixture"])
def test_batch_norm_without_training(barrier_fence_fixture,
                                     P_x_ranks, P_x_shape,
                                     input_shape,
                                     normalized_shape,
                                     elementwise_affine,
                                     comm_split_fixture):

    from distdl.backends.common.partition import MPIPartition
    from distdl.config import set_backend

    torch.manual_seed(0)

    set_backend(backend_comm=BACKEND_COMM, backend_array=BACKEND_ARRAY)

    # Isolate the minimum needed ranks
    base_comm, active = comm_split_fixture
    if not active:
        return
    P_world = MPIPartition(base_comm)

    # Create the partitions
    num_dimensions = len(input_shape)
    P_root_base = P_world.create_partition_inclusive([0])
    P_root = P_root_base.create_cartesian_topology_partition([1] * num_dimensions)
    P_x_base = P_world.create_partition_inclusive(P_x_ranks)
    P_x = P_x_base.create_cartesian_topology_partition(P_x_shape)

    # Create the input
    if P_root.active:
        input_train = torch.rand(input_shape, dtype=torch.float32, device=P_x.device)
        output = torch.rand(input_shape, dtype=torch.float32, device=P_x.device)
    else:
        input_train = zero_volume_tensor(device=P_x.device)
        output = zero_volume_tensor(device=P_x.device)

    # Scatter/gather data
    scatter = Repartition(P_root, P_x)
    gather = Repartition(P_x, P_root)

    # Sequential layer
    seq_ln = torch.nn.LayerNorm(normalized_shape).to(P_x.device)

    # Train sequential layer
    if P_root.active:
        seq_ln.eval()
        seq_out = seq_ln(input_train)
        seq_loss = ((seq_out - output)**2).sum()

    # Create distributed network
    dist_ln = DistributedLayerNormZero(P_x, normalized_shape,
                                       elementwise_affine=elementwise_affine
                                       ).to(P_x.device)

    # Train distributed network
    dist_ln.eval()
    dist_out = gather(dist_ln(scatter(input_train)))
    dist_loss = ((dist_out - output)**2).sum()

    # Compare the distributed and sequential networks
    if P_world.rank == 0:

        # Set the absolute tolerance to ~sqrt(e_mach), or the default
        # Pytorch got their defaults from NumPy, but NumPy defaults to 64-bit
        # floats, not 32-bit floats as torch does.  Consequently, the default
        # torch atol is actually tighter than one can expect from two fp-equal
        # floating point numbers.  The NumPy default of 1e-8 is closer to
        # sqrt(e_mach) for 64-bit numbers.  So we set the 32-bit tolerance to
        # sqrt(1e-7), as our usual choice, 1e-5, is too tight.
        if seq_out.dtype == torch.float64:
            atol = 1e-8
        elif seq_out.dtype == torch.float32:
            import math
            atol = math.sqrt(1e-7)
        else:
            # torch default
            atol = 1e-8

        assert dist_out.shape == seq_out.shape
        assert torch.allclose(dist_out, seq_out, rtol=ERROR_THRESHOLD, atol=atol)
        assert dist_loss.shape == seq_loss.shape
        assert torch.allclose(dist_loss, seq_loss, rtol=ERROR_THRESHOLD, atol=atol)

    P_world.deactivate()
    P_root_base.deactivate()
    P_root.deactivate()
    P_x_base.deactivate()
    P_x.deactivate()
