__all__ = ["ReduceScatterFunction"]

import cupy as cp
import numpy as np
import torch
from einops import rearrange
from mpi4py import MPI

from distdl.utilities.dtype import torch_to_cupy_dtype_dict
from distdl.utilities.slicing import get_rearrange_ordering
from distdl.utilities.torch import distdl_padding_to_torch_padding
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
                input_tensor_structure, output_tensor_structure, axes, scale_backward):
        r"""Forward function of distributed reduce-scatter layer.

        This method implements the forward reduce-scatter operation using the
        ``MPI_Reduce_scatter`` function on the communicator defined by ``P_reducescatter``.

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
        scale_backward : Union[int, slice]
            Scale the backward pass by the number of workers along the given dimension(s).

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
        ctx.scale_backward = scale_backward
        ctx.remainder = 0

        output = zero_volume_tensor(device=device, dtype=output_tensor_structure.dtype)
        output_tensor_shape = output_tensor_structure.shape.copy()

        # There is no need to specificy a root.
        if P_reducescatter.active:

            # Get operator to rearrange tensor to be contiguous along the reduce-scatter axis
            expanded_order, new_order = get_rearrange_ordering(len(input.shape), axes[0])
            operation = new_order + ' -> ' + expanded_order

            # If input shape does not split evenly along no. of partitions, we need to zero-pad
            ctx.remainder = input_tensor_structure.shape[axes[0]] % P_reducescatter.shape[axes[0]]
            if ctx.remainder > 0:

                # Split tensor along reduce-scatter axis
                local_shapes = [input_tensor_structure.shape[axes[0]] // P_reducescatter.shape[axes[0]]] * \
                    P_reducescatter.shape[axes[0]]
                for i in range(ctx.remainder):
                    local_shapes[i] += 1
                input_list = list(torch.split(input, local_shapes, dim=axes[0]))

                # Re-order each sub-tensor
                for i in range(P_reducescatter.shape[axes[0]]):
                    input_list[i] = rearrange(input_list[i], operation, p=1)

                # Zero-pad sub-tensors that are too small
                for i in range(ctx.remainder, P_reducescatter.shape[axes[0]]):
                    padding = [0] * 2 * P_reducescatter.dim
                    padding[2 * axes[0]] = 1
                    padding = distdl_padding_to_torch_padding(tuple(padding))
                    input_list[i] = torch.nn.functional.pad(input_list[i], padding, mode='constant', value=0)

                # Concatenate and flatten
                input_flat = torch.cat(input_list, dim=0).reshape(-1)

                # Update output shape for ranks that received zero-padded data
                if P_reducescatter.rank >= ctx.remainder:
                    output_tensor_shape[axes[0]] += 1
            else:
                input_flat = rearrange(input, operation, p=P_reducescatter.shape[axes]).reshape(-1)

            # Allocate output array
            cupy_dtype = torch_to_cupy_dtype_dict[input_tensor_structure.dtype]
            scattered_data = cp.zeros(output_tensor_shape, dtype=cupy_dtype)
            input_flat = cp.asarray(input_flat.detach(), dtype=cupy_dtype)

            # Reduce-scatter primitive
            P_reducescatter._comm.Reduce_scatter(input_flat, scattered_data, op=MPI.SUM)

        # If we had to receive data, we need to tensorify it.
        if P_reducescatter.active:
            output = torch.as_tensor(scattered_data, dtype=input_tensor_structure.dtype,
                                     device=device)
            output.requires_grad_(output_tensor_structure.requires_grad)

            # If we're one of the workers having received zero-padded data, remove padding
            if ctx.remainder != 0 and P_reducescatter.rank >= ctx.remainder:
                s = [slice(None)] * (P_reducescatter.dim)
                s[axes[0]] = slice(0, -1)
                output = output[s]

        return output

    @staticmethod
    def backward(ctx, grad_output):
        r"""Backward function of distributed reduce-scatter layer.

        This method implements the adjoint of the Jacobian of the
        reduce-scatter operation, the all-gather operation, using the
        ``MPI_Allgather`` function.

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
        remainder = ctx.remainder

        grad_input = zero_volume_tensor(device=device)
        input_tensor_shape = np.array(input_tensor_structure.shape)
        output_tensor_shape = np.array(output_tensor_structure.shape)

        # Scale by number of workers along the given dimension(s)
        if ctx.scale_backward is not None:
            grad_output.div_(np.prod(P_reducescatter.shape[ctx.scale_backward]))

        # All-gather operation
        if P_reducescatter.active:

            # If output shape does not evenly divide by number of partitions, we need to zero-pad input
            if remainder != 0:
                if P_reducescatter.rank >= remainder:
                    padding = [0] * 2 * P_reducescatter.dim
                    padding[2 * axes[0]] = 1
                    padding = distdl_padding_to_torch_padding(tuple(padding))
                    grad_output = torch.nn.functional.pad(grad_output, padding, mode='constant', value=0)
                output_tensor_shape = grad_output.shape
                input_tensor_shape[axes[0]] = output_tensor_shape[axes[0]] * P_reducescatter.shape[axes[0]]

            # Allocate output tensor
            cupy_dtype = torch_to_cupy_dtype_dict[input_tensor_structure.dtype]
            gathered_data = cp.zeros(np.prod(input_tensor_shape), dtype=cupy_dtype)
            grad_output_cupy = cp.asarray(grad_output.detach().contiguous())

            # All-gather
            P_reducescatter._comm.Allgather(grad_output_cupy, gathered_data)

        # If we had to receive data, we need to tensorify it.
        if P_reducescatter.active:
            grad_input.requires_grad_(input_tensor_structure.requires_grad)

            # Re-order flat output array from all-gather to correct cartesian shape
            gathered_data = torch.asarray(gathered_data, dtype=output_tensor_structure.dtype, device=device)

            # Reshape vectorized all-gather output to tensor.
            gathered_cart_shape = [P_reducescatter.shape[axes[0]]] + list(output_tensor_shape)
            grad_input = gathered_data.reshape(gathered_cart_shape)

            # Dimension ordering for rearrange.E.g.,  p a, b, c -> a, (p b), c
            in_shape_char, out_shape_char = get_rearrange_ordering(P_reducescatter.dim, axes[0])

            # If we zero-padded, we need to remove the padding now
            if remainder > 0:

                # Split tensor into its original inputs, so we can
                # remove padding from inputs that were zero-padded
                grad_input_list = list(torch.split(grad_input, 1, dim=0))

                # Remove padding
                s = [slice(None)] * (P_reducescatter.dim + 1)
                s[axes[0] + 1] = slice(0, -1)
                for i in range(remainder, P_reducescatter.shape[axes[0]]):
                    grad_input_list[i] = grad_input_list[i][s]

                # Rearrange and undo splitting
                for i in range(P_reducescatter.shape[axes[0]]):
                    grad_input_list[i] = rearrange(grad_input_list[i], in_shape_char + ' -> ' + out_shape_char)
                grad_input = torch.cat(tuple(grad_input_list), dim=axes[0])
            else:
                grad_input = rearrange(grad_input, in_shape_char + ' -> ' + out_shape_char)

        return grad_input, None, None, None, None, None
