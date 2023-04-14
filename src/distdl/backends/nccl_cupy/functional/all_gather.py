__all__ = ["AllGatherFunction"]

import threading
import time
import cupy as cp
import numpy as np
import torch
from mpi4py import MPI

from distdl.utilities.dtype import torch_to_cupy_dtype_dict
from distdl.utilities.torch import zero_volume_tensor


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
                input_tensor_structure, output_tensor_structure, slices):
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
        slices : tuple
            Tuple of slices in cartesian and flattened form for reshaping the input/output.

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
        ctx.slices = slices

        output = zero_volume_tensor(device=device)

        # There is no need to specificy a root.
        if P_allgather.active:

            # Allocate flattened output array
            cupy_dtype = torch_to_cupy_dtype_dict[input_tensor_structure.dtype]
            gathered_data = cp.zeros(np.prod(output_tensor_structure.shape), dtype=cupy_dtype)

            # All-gather (Conversion from torch cuda tensor to cupy array is via pointers. No mem copy.)
            input_cupy = cp.asarray(input.detach(), dtype=cupy_dtype)
            #assert input.__cuda_array_interface__['data'][0] == input_cupy.__cuda_array_interface__['data'][0]
            count = np.prod(input_cupy.shape).item()
            P_allgather._nccl.all_gather(input_cupy, gathered_data, count, stream=None)

        # If we had to receive data, we need to tensorify it.
        gathered_data_torch = torch.as_tensor(gathered_data, dtype=output_tensor_structure.dtype, device=device)
        #assert gathered_data_torch.__cuda_array_interface__['data'][0] == gathered_data.__cuda_array_interface__['data'][0]
        if P_allgather.active:
            
            # Re-order flat output array from all-gather to correct cartesian shape
            output = torch.zeros(torch.Size(output_tensor_structure.shape), 
                dtype=output_tensor_structure.dtype, device=device)
                
            for cart, flat in zip(*slices):
                output[cart] = gathered_data_torch[flat].reshape(input_tensor_structure.shape)
                
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
        slices = ctx.slices

        grad_input = zero_volume_tensor(device=device)

        # All-gather operation
        if P_allgather.active:

            # Allocate output array
            cupy_dtype = torch_to_cupy_dtype_dict[output_tensor_structure.dtype]
            scattered_data = cp.zeros(input_tensor_structure.shape, dtype=cupy_dtype)

            # Re-order input array
            grad_output_cupy = cp.asarray(grad_output.detach(), dtype=cupy_dtype)
            grad_output_flat = cp.zeros(np.prod(grad_output.shape), dtype=cupy_dtype)
            for cart, flat in zip(*slices):
                grad_output_flat[flat] = grad_output_cupy[cart].reshape(-1)

            # Reduce-scatter primitive
            count = np.prod(scattered_data.shape).item()
            P_allgather._nccl.reduce_scatter(grad_output_flat, scattered_data, count, op='sum', stream=None)

        # If we had to receive data, we need to tensorify it.
        if P_allgather.active:
            grad_input = torch.as_tensor(scattered_data, dtype=input_tensor_structure.dtype,
                                     device=device)
            grad_input.requires_grad_(input_tensor_structure.requires_grad)

        return grad_input, None, None, None, None