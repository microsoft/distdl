__all__ = ["ReduceScatterFunction"]

import threading
import time
import cupy as cp
import numpy as np
import torch
from mpi4py import MPI

from distdl.utilities.dtype import torch_to_cupy_dtype_dict
from distdl.utilities.torch import zero_volume_tensor


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
                input_tensor_structure, output_tensor_structure, slices):
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
        slices : tuple
            Tuple of slices in cartesian and flattened form for reshaping the input/output.

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
        ctx.slices = slices

        output = zero_volume_tensor(device=device)

        # There is no need to specificy a root.
        if P_reducescatter.active:
            
            # Allocate output array
            cupy_dtype = torch_to_cupy_dtype_dict[input_tensor_structure.dtype]
            scattered_data = cp.zeros(output_tensor_structure.shape, dtype=cupy_dtype)

            # Re-order input array
            input_flat = cp.zeros(np.prod(input.shape), dtype=cupy_dtype)
            for cart, flat in zip(*slices):
                input_flat[flat] = cp.array(input[cart].detach().reshape(-1))

            count = cp.prod(cp.array(scattered_data.shape)).item()
            P_reducescatter._nccl.reduce_scatter(input_flat, scattered_data, count, op='sum', stream=None)

        # If we had to receive data, we need to tensorify it.
        if P_reducescatter.active:
            output = torch.as_tensor(scattered_data, dtype=input_tensor_structure.dtype,
                                     device=device)
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
        slices = ctx.slices

        grad_input = zero_volume_tensor(device=device)

        # All-gather operation
        if P_reducescatter.active:
            cupy_dtype = torch_to_cupy_dtype_dict[input_tensor_structure.dtype]
            gathered_data = cp.zeros(np.prod(input_tensor_structure.shape), dtype=cupy_dtype)
            grad_output_cupy = cp.asarray(grad_output.detach())
            count = cp.prod(cp.array(grad_output_cupy.shape)).item()
            P_reducescatter._nccl.all_gather(grad_output_cupy, gathered_data, count, stream=None)

        # If we had to receive data, we need to tensorify it.
        if P_reducescatter.active:

            # Re-order flat output array from all-gather to correct cartesian shape
            grad_input = torch.zeros(torch.Size(input_tensor_structure.shape), 
                dtype=input_tensor_structure.dtype, device=device)
                
            for cart, flat in zip(*slices):
                grad_input[cart] = torch.tensor(gathered_data[flat].reshape(output_tensor_structure.shape), 
                    device=device)
                
            grad_input.requires_grad_(input_tensor_structure.requires_grad)

        return grad_input, None, None, None, None
