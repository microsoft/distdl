__all__ = ["SumReduceFunction"]

import threading
# import time
import cupy as cp
import torch
from mpi4py import MPI

from distdl.utilities.torch import zero_volume_tensor

# A better idea is to implement a progress engine for this purpose


def reduce_function(partition, src, dst):
    partition._comm.Reduce(src, dst, root=0, op=MPI.SUM)

class SumReduceFunction(torch.autograd.Function):
    r"""MPI-based functional implementation of a distributed sum-reduce layer.

    Implements the required `forward()` and adjoint (`backward()`) operations
    for a distributed SumReduce layer using the PyTorch autograd interface.

    This implementation uses MPI for data movement, accessed through the
    ``mpi4py`` MPI wrappers.

    Warning
    -------
    This implementation currently requires that tensors have data stored in main
    memory (CPU) only, not auxiliary memories such as those on GPUs.

    Warning
    -------
    The ``mpi4py`` interface currently used requires NumPy views of the tensors.

    """

    @staticmethod
    def forward(ctx, input, P_send, P_recv, preserve_batch,
                input_tensor_structure, output_tensor_structure,
                scale_backward):
        r"""Forward function of distributed sum-reduction layer.

        This method implements the forward sum-reduction operation using the
        ``MPI_Ireduce`` function.

        Any given worker may participate in two MPI reductions, one on the
        ``P_send`` partition and one on the ``P_recv`` partition.  The
        communication pattern and function selection is to avoid potential
        deadlocks due to potential overlaps in these partitions.

        When the current worker is active in its ``P_send`` partition, it
        *always* has data that must be reduced.  Therefore it will always send
        data (through a sum-reduce) to the root of that partition.

        If the current worker is active in ``P_recv`` then it is guaranteed to be the root
        worker of ``P_recv`` and there are two potential scenerios.

        1. If the ``P_send`` and ``P_recv`` partitions are distinct, the
           current worker will receive reduced tensor data as the root of an
           additional ``MPI_Ireduce``.
        2. If the ``P_send`` and ``P_recv`` partitions are the same, the
           reduction is completed by the *first* ``MPI_Ireduce`` and the second
           is not necessary, and in fact will cause a deadlock.

        When the current worker is inactive in the ``P_recv`` partition, it will
        output a zero-volume tensor, potentially preserving a non-zero batch
        size.

        Parameters
        ----------
        ctx :
            PyTorch context.
        input : `torch.tensor`
            Input tensor.
        P_send : Partition
            Sending partition current worker is a part of.
        P_recv : Partition
            Receiving partition current worker is a part of.
        preserve_batch : bool
            Indicates if batch size should be preserved for zero-volume outputs.
        input_tensor_structure : tuple
            Tuple containing properties of the input tensor (dimension, shape,
            requires_grad).
        output_tensor_structure : tuple
            Tuple containing properties of the output tensor (dimension, shape,
            requires_grad).
        scale_backward : Union[int, slice]
            Scale the backward pass by the number of workers along the given dimension(s).

        Returns
        -------
        output :
            Output tensor.

        """
        device = input.device
        ctx.P_send = P_send
        ctx.P_recv = P_recv
        ctx.preserve_batch = preserve_batch
        ctx.input_tensor_structure = input_tensor_structure
        ctx.output_tensor_structure = output_tensor_structure
        ctx.scale_backward = scale_backward
        ctx.device = device

        # This allows all ranks to use the same exit path, so that we can be
        # sure that all requests have cleared.
        if preserve_batch:
            output = zero_volume_tensor(input.shape[0], device=device, dtype=input_tensor_structure.dtype)
        else:
            output = zero_volume_tensor(device=device, dtype=input_tensor_structure.dtype)

        # By design, the roots are always 0 in the cross-communicators
        # If I receive data (either from a remote worker or just from myself)
        # I need to reduce that data.  If I send and receive to myself, this
        # is OK, as the reduction accounts for the copy, unlike the broadcast
        # below.
        if P_send.active:
            reduced_data_send = torch.zeros(input_tensor_structure.shape, dtype=input_tensor_structure.dtype, device=device)
            P_send._nccl.reduce(input.detach().contiguous(), reduced_data_send, op='sum', root=0, stream=None)

        # If I sent data in the forward, I have to receive it here.
        if P_send != P_recv and P_recv.active:
            reduced_data_recv = torch.zeros(output_tensor_structure.shape, dtype=output_tensor_structure.dtype, device=device)
            P_recv._nccl.reduce(reduced_data_recv, reduced_data_recv, op='sum', root=0, stream=None)

        # If we had to receive data, we need to tensorify it.
        if P_recv.active:
            if P_send == P_recv:
                output = reduced_data_send
                output.requires_grad_(output_tensor_structure.requires_grad)
            else:
                output = reduced_data_recv
                output.requires_grad_(output_tensor_structure.requires_grad)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        r"""Backward function of distributed sum-reduction layer.

        This method implements the adjoint of the Jacobian of the sum-reduce
        operation, a sum-reduce, using the ``MPI_Ibcast`` function.

        The roles of the respective send and receive partitions are reversed in
        the adjoint algorithm.  Any worker that was the source of reduced data
        in the forward algorithm will be the destination of broadcast data in
        the adjoint.

        Any given worker may participate in two MPI broadcasts, one on the
        ``P_recv`` partition and one on the ``P_send`` partition.  The
        communication pattern and function selection is to avoid potential
        deadlocks due to potential overlaps in these partitions.

        When the current worker is active in its ``P_recv`` partition, it
        *always* has data that it must share.  It will only be active in
        ``P_recv`` if it is the root worker of that partition, therefore, it
        will send tensor data as the root of an ``MPI_Ibcast``.

        When the current worker is active in its ``P_send`` partition, there are
        multiple potential scenerios.

        1. If it is *active* in a ``P_recv`` partition and ``P_recv`` is the
           *same* partition as ``P_send``, then the input subtensor can simply
           be cloned for the output.
        2. If it is *active* in a ``P_recv`` partition and ``P_recv`` is a
           *different* partition from ``P_send``, then it will receive tensor
           data from the root of an ``MPI_Ibcast``.
        3. If it is *inactive* in a ``P_recv`` partition, then it will receive
           tensor data from the root of an ``MPI_Ibcast``.

        When the current worker is inactive in the ``P_send`` partition, it will
        output a zero-volume tensor, potentially preserving a non-zero batch
        size.

        Parameters
        ----------
        ctx :
            PyTorch context.
        grad_output : `torch.tensor`
            Input tensor.

        Returns
        -------
        grad_input :
            Output tensor.
        """

        P_send = ctx.P_send
        P_recv = ctx.P_recv
        preserve_batch = ctx.preserve_batch
        input_tensor_structure = ctx.input_tensor_structure
        device = ctx.device

        assert grad_output.device == device

        # This allows all ranks to use the same exit path, so that we can be
        # sure that all requests have cleared.
        if preserve_batch:
            grad_input = zero_volume_tensor(grad_output.shape[0], device=device, dtype=input_tensor_structure.dtype)
        else:
            grad_input = zero_volume_tensor(device=device, dtype=input_tensor_structure.dtype)

        # If I received the reduction in the forward call, I broadcast my data
        if P_recv.active:
            if ctx.scale_backward is not None:
                grad_output.div_(P_recv.shape[ctx.scale_backward])
            P_recv._nccl.broadcast(grad_output.detach().contiguous(), root=0, stream=None)

        # If I just receive, receive the broadcast
        if P_send.active:
            # If I both sent and received reduction data, then I copy the "input"
            if P_send == P_recv:
                if ctx.scale_backward is not None:
                    grad_output.div_(P_send.shape[ctx.scale_backward])
                grad_input = grad_output
            else:
                grad_input = torch.zeros(input_tensor_structure.shape, dtype=input_tensor_structure.dtype, device=device)
                P_send._nccl.broadcast(grad_input, root=0, stream=None)

                grad_input.requires_grad_(input_tensor_structure.requires_grad)

        return grad_input, None, None, None, None, None, None
