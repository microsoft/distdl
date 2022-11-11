__all__ = ["ReduceScatterFunction"]

import numpy as np
import torch
from mpi4py import MPI

from distdl.utilities.dtype import torch_to_numpy_dtype_dict
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
                input_tensor_structure, output_tensor_structure):
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

        output = zero_volume_tensor(device=device)

        requests = []

        # There is no need to specificy a root.
        if P_reducescatter.active:
            numpy_dtype = torch_to_numpy_dtype_dict[input_tensor_structure.dtype]

            scattered_data = np.zeros(output_tensor_structure.shape, dtype=numpy_dtype)
            input_numpy = input.detach().cpu().numpy()
            req = P_reducescatter._comm.Ireduce_scatter(input_numpy, scattered_data, op=MPI.SUM)
            requests.append(req)

        MPI.Request.Waitall(requests)

        # If we had to receive data, we need to tensorify it.
        if P_reducescatter.active:
            output = torch.tensor(scattered_data,
                                  requires_grad=output_tensor_structure.requires_grad,
                                  device=device)

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
        device = ctx.device

        grad_input = zero_volume_tensor(device=device)

        requests = []

        # All-gather operation
        if P_reducescatter.active:
            numpy_dtype = torch_to_numpy_dtype_dict[input_tensor_structure.dtype]

            gathered_data = np.zeros(input_tensor_structure.shape, dtype=numpy_dtype)
            grad_output_numpy = grad_output.detach().cpu().numpy()
            req = P_reducescatter._comm.Iallgather(grad_output_numpy, gathered_data)
            requests.append(req)

        MPI.Request.Waitall(requests)

        # If we had to receive data, we need to tensorify it.
        if P_reducescatter.active:
            grad_input = torch.tensor(gathered_data,
                                      requires_grad=input_tensor_structure.requires_grad,
                                      device=device)

        return grad_input, None, None, None
