__all__ = ["ReduceScatterFunction"]

import threading
import time
import cupy as cp
import numpy as np
import torch
from mpi4py import MPI
from einops import rearrange

from distdl.utilities.torch import zero_volume_tensor
from distdl.utilities.slicing import get_rearrange_ordering

def reorg(x_vec, P_x, axis, in_shape):

    # Reshape to p x (n1 x n2 x ... x nN)
    num_partitions = P_x.shape[axis]
    new_shape =  [num_partitions] + list(in_shape)
    x = x_vec.reshape(new_shape)

    # Permute back to orig shape
    num_dims = len(P_x.shape)
    expanded_order, new_order = get_rearrange_ordering(num_dims, axis)
    operation = expanded_order + ' -> ' + new_order
    return rearrange(x, operation)


class ReduceScatterFunction(torch.autograd.Function):
    r"""MPI-based functional implementation of a distributed reduce-scatter layer.

    Implements the required `forward()` and adjoint (`backward()`) operations
    for a distributed ReduceScatter layer using the PyTorch autograd interface.

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
    def forward(ctx, input, P_reducescatter,
                input_tensor_structure, output_tensor_structure, axes):
        r"""Forward function of distributed reduce-scatter layer.

        This method implements the forward reduce-scatter operation using the
        ``MPI_Ireduce_scatter`` function on the communicator defined by ``P_reducescatter``.

        When the current worker is inactive in the ``P_reducescatter`` partition, it will
        output a zero-volume tensor.

        Parameters
        ----------
        ctx :
            PyTorch context.
        input : `torch.tensor`
            Input tensor.
        P_reducescatter : Partition
            Partition reduce-scatter happens within.
        input_tensor_structure : tuple
            Tuple containing properties of the input tensor (dimension, shape,
            requires_grad).
        output_tensor_structure : tuple
            Tuple containing properties of the output tensor (dimension, shape,
            requires_grad).
        axes : tuple
            Axes along which to reduce-scatter.

        Returns
        -------
        output :
            Output tensor.

        """

        device = input.device
        ctx.P_reducescatter = P_reducescatter
        ctx.input_tensor_structure = input_tensor_structure
        ctx.output_tensor_structure = output_tensor_structure
        ctx.device = device
        ctx.axes = axes

        output = zero_volume_tensor(device=device, dtype=output_tensor_structure.dtype)

        # There is no need to specificy a root.
        if P_reducescatter.active:
            
            # Allocate output array
            scattered_data = torch.zeros(torch.Size(output_tensor_structure.shape), dtype=output_tensor_structure.dtype, device=device)

            # Re-order input array
            expanded_order, new_order = get_rearrange_ordering(len(input.shape), axes[0])
            operation = new_order + ' -> ' + expanded_order
            input_flat = rearrange(input, operation, p=P_reducescatter.shape[axes]).reshape(-1)

            # Reduce-scatter operation
            count = np.prod(scattered_data.shape).item()
            P_reducescatter._nccl.reduce_scatter(input_flat, scattered_data, count, op='sum', stream=None)

        # If we had to receive data, we need to tensorify it.
        if P_reducescatter.active:
            output = scattered_data
            output.requires_grad_(output_tensor_structure.requires_grad)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        r"""Backward function of distributed reduce-scatter layer.

        This method implements the adjoint of the Jacobian of the
        reduce-scatter operation, the all-gather operation, using the
        ``MPI_Iallgather`` function.

        When the current worker is inactive in the ``P_reducescatter`` partition,
        it will output a zero-volume tensor.

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

        P_reducescatter = ctx.P_reducescatter
        input_tensor_structure = ctx.input_tensor_structure
        output_tensor_structure = ctx.output_tensor_structure
        device = ctx.device
        axes = ctx.axes

        grad_input = zero_volume_tensor(device=device, dtype=input_tensor_structure.dtype)

        # All-gather operation
        if P_reducescatter.active:
            
            gathered_data = torch.zeros(np.prod(input_tensor_structure.shape), dtype=input_tensor_structure.dtype, device=device)
            count = np.prod(grad_output.shape).item()
            P_reducescatter._nccl.all_gather(grad_output.detach().contiguous(), gathered_data, count, stream=None)

        # If we had to receive data, we need to tensorify it.
        if P_reducescatter.active:
            grad_input = reorg(gathered_data, P_reducescatter, axes[0], output_tensor_structure.shape)
            grad_input.requires_grad_(input_tensor_structure.requires_grad)



        return grad_input, None, None, None, None
