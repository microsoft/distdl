import numpy as np
import pytest
import torch

from distdl.nn.embedding_zero import DistributedEmbeddingZero
from distdl.nn.repartition import Repartition
from distdl.utilities.torch import zero_volume_tensor

ERROR_THRESHOLD = 1e-4

BACKEND_COMM = "mpi"
BACKEND_ARRAY = "numpy"
ZERO_ENABLED = True

parametrizations_affine = []

parametrizations_affine.append(
    pytest.param(
        np.arange(0, 1), [1, 1],  # P_x_ranks, P_x_shape,
        (4, 12),  # embedding_shape
        1,  # passed to comm_split_fixture, required MPI ranks
        id="serial-embedding",
        marks=[pytest.mark.mpi(min_size=1)]
    )
)

parametrizations_affine.append(
    pytest.param(
        np.arange(0, 4), [1, 4],  # P_x_ranks, P_x_shape,
        (4, 12),  # embedding_shape
        4,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-embedding",
        marks=[pytest.mark.mpi(min_size=4)]
    )
)


@pytest.mark.parametrize("P_x_ranks, P_x_shape,"
                         "embedding_shape,"
                         "comm_split_fixture",
                         parametrizations_affine,
                         indirect=["comm_split_fixture"])
def test_batch_norm_with_training(barrier_fence_fixture,
                                  P_x_ranks, P_x_shape,
                                  embedding_shape,
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
    num_dimensions = len(embedding_shape)
    P_root_base = P_world.create_partition_inclusive([0])
    P_root = P_root_base.create_cartesian_topology_partition([1] * num_dimensions)
    P_x_base = P_world.create_partition_inclusive(P_x_ranks)
    P_x = P_x_base.create_cartesian_topology_partition(P_x_shape)

    # Input/output data
    input_range = torch.arange(embedding_shape[0]).to(P_x.device)
    if P_root.active:
        weight = torch.rand(embedding_shape, dtype=torch.float32, device=P_x.device)
        output = torch.rand(embedding_shape, dtype=torch.float32, device=P_x.device)
    else:
        weight = zero_volume_tensor(device=P_x.device)
        output = zero_volume_tensor(device=P_x.device)

    # Scatter/gather data
    scatter = Repartition(P_root, P_x)
    gather = Repartition(P_x, P_root)

    # Sequential layer
    if P_root.active:
        seq_emb = torch.nn.Embedding(*embedding_shape, _weight=torch.clone(weight)).to(P_x.device)

    # Train sequential layer
    if P_root.active:
        seq_emb.train()
        seq_out1 = seq_emb(input_range)
        seq_loss = ((seq_out1 - output)**2).sum()
        seq_loss.backward()
        seq_grads = [p.grad.detach().cpu() for p in seq_emb.parameters()]

        # Do a manual weight update (this is what optimizer does):
        with torch.no_grad():
            for p in seq_emb.parameters():
                p.copy_(p + 0.1 * p.grad)

    # Evaluate sequential network
    if P_root.active:
        seq_emb.eval()
        seq_out2 = seq_emb(input_range).detach()

    # Create distributed network
    weight_local = scatter(weight)
    dist_emb = DistributedEmbeddingZero(P_x, *embedding_shape, _weight=weight_local).to(P_x.device)

    # Train distributed network
    dist_emb.train()
    dist_out1 = gather(dist_emb(input_range))
    dist_loss = ((dist_out1 - output)**2).sum()
    assert dist_loss.requires_grad
    dist_loss.backward()

    # Do a manual gradient update
    if dist_emb.P_x.active:
        dist_grads = []
        with torch.no_grad():
            for p in dist_emb.parameters():
                p.copy_(p + 0.1 * p.grad)

    # Evaluate distributed network
    dist_emb.eval()
    dist_out2 = gather(dist_emb(input_range)).detach()

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
        for dist_grad, seq_grad in zip(dist_grads, seq_grads):
            assert dist_grad.shape == seq_grad.shape
            assert torch.allclose(dist_grad, seq_grad, rtol=ERROR_THRESHOLD, atol=atol)
        assert dist_out2.shape == seq_out2.shape
        assert torch.allclose(dist_out2, seq_out2, rtol=ERROR_THRESHOLD, atol=atol)

    P_world.deactivate()
    P_root_base.deactivate()
    P_root.deactivate()
    P_x_base.deactivate()
    P_x.deactivate()


@pytest.mark.parametrize("P_x_ranks, P_x_shape,"
                         "embedding_shape,"
                         "comm_split_fixture",
                         parametrizations_affine,
                         indirect=["comm_split_fixture"])
def test_batch_norm_without_training(barrier_fence_fixture,
                                     P_x_ranks, P_x_shape,
                                     embedding_shape,
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
    num_dimensions = len(embedding_shape)
    P_root_base = P_world.create_partition_inclusive([0])
    P_root = P_root_base.create_cartesian_topology_partition([1] * num_dimensions)
    P_x_base = P_world.create_partition_inclusive(P_x_ranks)
    P_x = P_x_base.create_cartesian_topology_partition(P_x_shape)

    # Input/output data
    input_range = torch.arange(embedding_shape[0]).to(P_x.device)
    if P_root.active:
        weight = torch.rand(embedding_shape, dtype=torch.float32, device=P_x.device)
        output = torch.rand(embedding_shape, dtype=torch.float32, device=P_x.device)
    else:
        weight = zero_volume_tensor(device=P_x.device)
        output = zero_volume_tensor(device=P_x.device)

    # Scatter/gather data
    scatter = Repartition(P_root, P_x)
    gather = Repartition(P_x, P_root)

    # Sequential layer
    if P_root.active:
        seq_emb = torch.nn.Embedding(*embedding_shape, _weight=torch.clone(weight)).to(P_x.device)

    # Train sequential layer
    if P_root.active:
        seq_emb.eval()
        seq_out1 = seq_emb(input_range)
        seq_loss = ((seq_out1 - output)**2).sum()

    # Create distributed network
    weight_local = scatter(weight)
    dist_emb = DistributedEmbeddingZero(P_x, *embedding_shape, _weight=weight_local).to(P_x.device)

    # Train distributed network
    dist_emb.eval()
    dist_out1 = gather(dist_emb(input_range))
    dist_loss = ((dist_out1 - output)**2).sum()

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

    P_world.deactivate()
    P_root_base.deactivate()
    P_root.deactivate()
    P_x_base.deactivate()
    P_x.deactivate()
