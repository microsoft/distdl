__all__ = ["AllGatherFunction"]

import threading
import time
import numpy as np
import torch
from mpi4py import MPI
from einops import rearrange

from distdl.utilities.dtype import torch_to_numpy_dtype_dict
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

        output = zero_volume_tensor(device=device)

        requests = []

        # There is no need to specificy a root.
        if P_allgather.active:

            # Allocate flattened output array
            numpy_dtype = torch_to_numpy_dtype_dict[input_tensor_structure.dtype]
            gathered_data = np.zeros(np.prod(output_tensor_structure.shape), dtype=numpy_dtype)

            # All-gather
            input_numpy = np.asarray(input.detach(), dtype=numpy_dtype)
            req = P_allgather._comm.Iallgather(input_numpy, gathered_data)
            requests.append(req)

        MPI.Request.Waitall(requests)

        # If we had to receive data, we need to tensorify it.
        if P_allgather.active:
            
            # Re-order flat output array from all-gather to correct cartesian shape
            gathered_data = torch.tensor(gathered_data, device=device)
            output = reorg(gathered_data, P_allgather, axes[0], input_tensor_structure.shape)
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

        grad_input = zero_volume_tensor(device=device)

        requests = []

        # All-gather operation
        if P_allgather.active:

            # Allocate output array
            numpy_dtype = torch_to_numpy_dtype_dict[output_tensor_structure.dtype]
            scattered_data = np.zeros(input_tensor_structure.shape, dtype=numpy_dtype)

            # Re-order input array
            expanded_order, new_order = get_rearrange_ordering(len(grad_output.shape), axes[0])
            operation = new_order + ' -> ' + expanded_order
            grad_output_flat = rearrange(grad_output, operation, p=P_allgather.shape[axes]).reshape(-1)
            grad_output_flat = np.asarray(grad_output_flat.detach(), dtype=numpy_dtype)

            # Reduce-scatter primitive
            req = P_allgather._comm.Ireduce_scatter(grad_output_flat, scattered_data, op=MPI.SUM)
            requests.append(req)

        MPI.Request.Waitall(requests)

        # If we had to receive data, we need to tensorify it.
        if P_allgather.active:
            grad_input = torch.as_tensor(scattered_data, dtype=input_tensor_structure.dtype,
                                     device=device)
            grad_input.requires_grad_(input_tensor_structure.requires_grad)

        return grad_input, None, None, None, None