__all__ = ["ReduceScatterFunction"]

import threading
import time
import cupy as cp
import numpy as np
import torch
from mpi4py import MPI

from distdl.utilities.dtype import torch_to_cupy_dtype_dict
from distdl.utilities.torch import zero_volume_tensor

def reorder_for_scatter(input, in_shape, out_shape, cupy_dtype, P_rs, axis):
    r"""Reorder a multi-dimensional torch tensor for the reduce-scatter operation.

    The reduce-scatter primitive in MPI and NCCL operators on one-dimensional 
    arrays (and multi-dimensional arrays are implicitly vectorized). To compute
    the correct reduce-scatter operation using the cartesian partitoning scheme,
    we need to re-order the data of our input tensor.

    Parameters
    ----------
    input : `torch.tensor`
        Input torch tensor to the reduce-scatter operation.
    in_shape : `torch.Size`
        Shape of the input tensor.
    out_shape : `torch.Size`
        Shape of the output tensor.
    cupy_dtype: `cupy.dtype`
        Cupy data type of the input/output tensor.
    P_rs : `Partition`
        Partition reduce-scatter happens within.
    axis : `tuple`
        Tuple containing the axis along which to carry out the reduce-scatter.

    Returns
    -------
    output : `cp.array`
        One-dimensional cupy tensor.

    """    
    # We create a one-dimensional cupy tensor for the 
    # reduce-scatter operation.
    axis = axis[0]
    output = cp.zeros(np.prod(in_shape), dtype=cupy_dtype)

    # We slice the input data along the dimension of the reduce-scatter
    # operation and insert it into the right location in the 1D array.
    for i in range(P_rs.shape[axis]):
        
        # Slice source array
        index_slices = []
        for j in range(len(in_shape)):
            if j == axis:
                index_slices.append(slice(int(i*out_shape[axis]), 
                    int((i+1)*out_shape[axis])))
            else:
                index_slices.append(slice(0, in_shape[j]))
        output[i*np.prod(out_shape): (i+1)*np.prod(out_shape)] = cp.array(
            input[tuple(index_slices)].reshape(-1))

    return output

def reorder_from_allgather(input, in_shape, out_shape, torch_dtype, P_rs, device, axis):
    r"""Reorder a multi-dimensional torch tensor after the all-gather operation.

    The all-gather primitive in MPI and NCCL operators on one-dimensional 
    arrays (and multi-dimensional arrays are implicitly vectorized). To compute
    the correct all-gather operation using the cartesian partitoning scheme,
    we need to re-order the data after perfomring the all-gather operation.

    Parameters
    ----------
    input : `cupy.array`
        One-dimensional cupy tensor from all-gather operation.
    in_shape : `torch.Size`
        Shape of the input tensor.
    out_shape : `torch.Size`
        Shape of the output tensor.
    torch_dtype: `torch.dtype`
        PyTorch data type of the output tensor.
    P_rs : `Partition`
        Partition reduce-scatter happens within.
    axis : `tuple`
        Tuple containing the axis along which to carry out the all-gather.

    Returns
    -------
    output : `torch.tensor`
        Torch tensor with correct output shape.

    """ 
    # Reshape target array to correct form
    axis = axis[0]
    output = torch.zeros(torch.Size(out_shape), dtype=torch_dtype, device=device)

    # We undo the re-order and vecorization from the reorder_for_scatter function:
    # We locate the correct data within the 1D input array and insert it into the 
    # correct location in the multi-dimension output array.
    for i in range(P_rs.shape[axis]):
        
        # Slice source array
        index_slices = []
        for j in range(len(out_shape)):
            if j == axis:
                index_slices.append(slice(int(i*in_shape[axis]), int((i+1)*in_shape[axis])))
            else:
                index_slices.append(slice(0, out_shape[j]))     
        output[tuple(index_slices)] = torch.tensor(input[i*np.prod(in_shape): 
            (i+1)*np.prod(in_shape)].reshape(in_shape))

    return output


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
            Tuple containing the axis along which to carry out the all-reduce.

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

        output = zero_volume_tensor(device=device)

        # There is no need to specificy a root.
        if P_reducescatter.active:
            cupy_dtype = torch_to_cupy_dtype_dict[input_tensor_structure.dtype]
            scattered_data = cp.zeros(output_tensor_structure.shape, dtype=cupy_dtype)
            #input_cupy = cp.asarray(input.detach())
            input_cupy = reorder_for_scatter(input.detach(), input_tensor_structure.shape, 
                output_tensor_structure.shape, cupy_dtype, P_reducescatter, axes)
            count = cp.prod(cp.array(scattered_data.shape)).item()
            P_reducescatter._nccl.reduce_scatter(input_cupy, scattered_data, count, op='sum', stream=None)

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
        axes = ctx.axes

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
            grad_input = reorder_from_allgather(gathered_data, output_tensor_structure.shape, 
                input_tensor_structure.shape, input_tensor_structure.dtype, P_reducescatter, device, axes)
            grad_input.requires_grad_(input_tensor_structure.requires_grad)

        return grad_input, None, None, None, None
