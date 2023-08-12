__all__ = ["AllGatherFunction"]

import threading
import time
import cupy as cp
import numpy as np
import torch
from mpi4py import MPI
from einops import rearrange

from distdl.utilities.dtype import torch_to_cupy_dtype_dict
from distdl.utilities.torch import zero_volume_tensor, distdl_padding_to_torch_padding
from distdl.utilities.slicing import get_rearrange_ordering


class AllGatherFunction(torch.autograd.Function):
    r"""MPI-based functional implementation of a distributed all-gather layer.

    Implements the required `forward()` and adjoint (`backward()`) operations
    for a distributed AllGather layer using the PyTorch autograd interface.

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
    def forward(ctx, input, P_allgather,
                input_tensor_structure, output_tensor_structure, axes):
        r"""Forward function of distributed all-gather layer.

        This method implements the forward all-gather operation using the
        ``MPI_Allgather`` function on the communicator defined by ``P_allgather``.

        When the current worker is inactive in the ``P_allgather`` partition, it will
        output a zero-volume tensor.

        Parameters
        ----------
        ctx :
            PyTorch context.
        input : `torch.tensor`
            Input tensor.
        P_allgather : Partition
            Partition all-gather happens within.
        input_tensor_structure : tuple
            Tuple containing properties of the input tensor (dimension, shape,
            requires_grad).
        output_tensor_structure : tuple
            Tuple containing properties of the output tensor (dimension, shape,
            requires_grad).
        axes : tuple
            Axes along which to all-gather.
            
        Returns
        -------
        output :
            Output tensor.

        """

        device = input.device
        ctx.P_allgather = P_allgather
        ctx.input_tensor_structure = input_tensor_structure
        ctx.output_tensor_structure = output_tensor_structure
        ctx.device = device
        ctx.axes = axes
        ctx.remainder = 0

        output = zero_volume_tensor(device=device, dtype=output_tensor_structure.dtype)
        input_tensor_shape = np.array(input_tensor_structure.shape)
        output_tensor_shape = np.array(output_tensor_structure.shape)

        # There is no need to specificy a root.
        if P_allgather.active:

            # If output shape does not evenly divide by number of partitions, we need to zero-pad the input
            ctx.remainder = output_tensor_shape[axes[0]] % P_allgather.shape[axes[0]]
            if ctx.remainder != 0:
                if P_allgather.rank >= ctx.remainder:
                    padding = [0]*2*P_allgather.dim
                    padding[2*axes[0]] = 1
                    padding = distdl_padding_to_torch_padding(tuple(padding))
                    input = torch.nn.functional.pad(input, padding, mode='constant', value=0)

                # Update input/ouput shapes after padding
                input_tensor_shape = input.shape
                output_shape = list(output_tensor_shape)
                output_shape[axes[0]] = input_tensor_shape[axes[0]] * P_allgather.shape[axes[0]]
                output_tensor_shape = torch.Size(output_shape)

            # Allocate flattened output array
            cupy_dtype = torch_to_cupy_dtype_dict[input_tensor_structure.dtype]
            gathered_data = cp.zeros(np.prod(output_tensor_shape), dtype=cupy_dtype)

            # All-gather
            input_cupy = cp.asarray(input.detach(), dtype=cupy_dtype)
            P_allgather._comm.Allgather(input_cupy, gathered_data)

        # If we had to receive data, we need to tensorify it.
        if P_allgather.active:
            
            # Re-order flat output array from all-gather to correct cartesian shape
            gathered_cart_shape = [P_allgather.shape[axes[0]]] + list(input_tensor_shape)
            output = torch.tensor(gathered_data, device=device).reshape(gathered_cart_shape)

            # Dimension ordering for rearrange.E.g.,  p a, b, c -> a, (p b), c
            in_shape_char, out_shape_char = get_rearrange_ordering(P_allgather.dim, axes[0])

            # If we zero-padded, we need to remove the padding now from the gathered tensor.
            if ctx.remainder > 0:

                # Split tensor into its original inputs, so we can
                # remove padding from inputs that were zero-padded
                output_list = list(torch.split(output, 1, dim=0))

                # Remove padding
                s = [slice(None)]*(P_allgather.dim+1)
                s[axes[0]+1] = slice(0, -1)
                for i in range(ctx.remainder, P_allgather.shape[axes[0]]):
                    output_list[i] = output_list[i][s]
            
                # Rearrange
                for i in range(P_allgather.shape[axes[0]]):
                    output_list[i] = rearrange(output_list[i], in_shape_char + ' -> ' + out_shape_char)

                # Undo splitting
                output = torch.cat(tuple(output_list), dim=axes[0])

            else:
                output = rearrange(output, in_shape_char + ' -> ' + out_shape_char)
            output.requires_grad_(output_tensor_structure.requires_grad)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        r"""Backward function of distributed all-gather layer.

        This method implements the adjoint of the Jacobian of the
        all-gather operation, the reduce-scatter operation, using the
        ``MPI_Reduce_scatter`` function.

        When the current worker is inactive in the ``P_allgather`` partition,
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

        P_allgather = ctx.P_allgather
        input_tensor_structure = ctx.input_tensor_structure
        output_tensor_structure = ctx.output_tensor_structure
        device = ctx.device
        axes = ctx.axes
        remainder = ctx.remainder

        grad_input = zero_volume_tensor(device=device, dtype=input_tensor_structure.dtype)
        input_tensor_shape = np.array(input_tensor_structure.shape)

        # All-gather operation
        if P_allgather.active:

            # Re-order input array
            expanded_order, new_order = get_rearrange_ordering(len(grad_output.shape), axes[0])
            operation = new_order + ' -> ' + expanded_order

            # If input shape does not split evenly along no. of partitions, we need to zero-pad
            if remainder > 0:

                # Split tensor along reduce-scatter axis
                local_shapes = [output_tensor_structure.shape[axes[0]] // P_allgather.shape[axes[0]]] * \
                    P_allgather.shape[axes[0]]
                for i in range(ctx.remainder):
                    local_shapes[i] += 1
                grad_output_list = list(torch.split(grad_output, local_shapes, dim=axes[0]))

                # Re-order each sub-tensor
                for i in range(P_allgather.shape[axes[0]]):
                    grad_output_list[i] = rearrange(grad_output_list[i], operation, p=1)

                # Zero-pad sub-tensors that are too small
                for i in range(ctx.remainder, P_allgather.shape[axes[0]]):
                    padding = [0]*2*P_allgather.dim
                    padding[2*axes[0]] = 1
                    padding = distdl_padding_to_torch_padding(tuple(padding))
                    grad_output_list[i] = torch.nn.functional.pad(grad_output_list[i], padding, mode='constant', value=0)

                # Concatenate and flatten
                grad_output_flat = torch.cat(grad_output_list, dim=0).reshape(-1)
                
                # Update output shape for ranks that received zero-padded data
                if P_allgather.rank >= ctx.remainder:
                    input_tensor_shape[axes[0]] += 1
            else:
                grad_output_flat = rearrange(grad_output, operation, p=P_allgather.shape[axes]).reshape(-1)

            # Allocate output array
            cupy_dtype = torch_to_cupy_dtype_dict[output_tensor_structure.dtype]
            scattered_data = cp.zeros(input_tensor_shape, dtype=cupy_dtype)
            grad_output_flat = cp.asarray(grad_output_flat, dtype=cupy_dtype)

            # Reduce-scatter primitive
            P_allgather._comm.Reduce_scatter(grad_output_flat, scattered_data, op=MPI.SUM)

        # If we had to receive data, we need to tensorify it.
        if P_allgather.active:
            grad_input = torch.as_tensor(scattered_data, dtype=input_tensor_structure.dtype,
                                     device=device)
            grad_input.requires_grad_(input_tensor_structure.requires_grad)

            # If we're one of the workers having received zero-padded data, remove padding
            if remainder != 0 and P_allgather.rank >= remainder:
                s = [slice(None)]*(P_allgather.dim)
                s[axes[0]] = slice(0, -1)
                grad_input = grad_input[s]

        return grad_input, None, None, None, None