import numbers

import numpy as np
import torch

from distdl.nn.all_sum_reduce import AllSumReduce
from distdl.nn.broadcast import Broadcast
from distdl.nn.module import Module
from distdl.nn.repartition import Repartition
from distdl.utilities.slicing import compute_subshape
from distdl.utilities.slicing import worker_layout
from distdl.utilities.torch import zero_volume_tensor


class DistributedLayerNorm(Module):
    r"""A distributed layer norm layer.

    Applies Layer Normalization. This layer is a distributed and generalized
    version of the PyTorch LayerNorm layer.

    Parameters
    ----------
    P_x :
        Partition of the input tensor.  Outputs are of the same shape,
        and therefore re-use the input partition.
    normalized_shape :
        Input shape from an expected input of size. If a single integer is used,
        it is treated as a singleton list, and this module will normalize over the
        last dimension which is expected to be of that specific size.
    eps : optional
        A value added to the denominator for numerical stability.
        Default is 1e-5.
    elementwise_affine : optional
        A boolean value that when set to True, this module has learnable per-element affine
        parameters of size normalized_shape initialized to ones (for weights) and zeros (for biases).
        Default is True.
    collect_state : optional
        If True, weights and biases are gathered to the root worker and serialized to disk when the
        state_dict() function is called. Instead of the weights and biases themselves, the state
        dictionary will contain paths to the serialized files. Default is False.
    device: optional
        Computational device. Default is P_x.device.
    dtype: optional
        Data type of learnable parameters. Default is torch.float.
    scale_backward : Union[int, slice], optional
        Scale backward pass for AllGather operation by no. of workers along the given
        dimension. Default is None.
    """

    def __init__(self, P_x, normalized_shape, elementwise_affine=True, eps=1e-5,
                 collect_state=False, device=None, dtype=None, scale_backward=None
                 ):
        super(DistributedLayerNorm, self).__init__()

        self.P_x = P_x
        if not self.P_x.active:
            return

        if device is None:
            device = P_x.device
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.collect_state = collect_state
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.dtype = dtype

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = tuple(normalized_shape)
        self.normalized_shape = normalized_shape

        # Number of dimensions across which mean/var is computed
        num_dim = len(normalized_shape)

        # Dimensions across which to reduce
        self.dim_reduce = tuple(torch.arange(0, P_x.dim)[-num_dim:])
        self.allreduce = AllSumReduce(P_x, axes_reduce=self.dim_reduce)     # for computing mean/variance

        # Number of workers across we reduce
        dim_reduce_slice = slice(P_x.dim - num_dim, P_x.dim)   # dimensions across which we compute mean/var over
        dim_bcast_slice = slice(0, P_x.dim - num_dim)  # dimensions across which we broadcast weights/biases over
        self.num_reduce = np.prod(P_x.shape[dim_reduce_slice])
        self.dim_bcast_slice = dim_bcast_slice
        self.dim_reduce_slice = dim_reduce_slice

        if self.elementwise_affine:

            # Shape of partition storing weights/biases
            weight_partition_shape = np.copy(P_x.shape)
            weight_partition_shape[dim_bcast_slice] = 1

            # Ranks of workers storing weights
            index = [0] * P_x.dim
            for i in range(dim_reduce_slice.start, dim_reduce_slice.stop):
                index[i] = slice(0, P_x.shape[i])
            index = tuple(index)
            storage_workers = worker_layout(P_x.shape)[index].reshape(-1).tolist()

            # Weight partition and broadcast
            P_w_base = P_x.create_partition_inclusive(storage_workers)
            P_w = P_w_base.create_cartesian_topology_partition(weight_partition_shape)
            P_w_base.deactivate()
            self.P_w = P_w
            self.broadcast = Broadcast(P_w, P_x, scale_backward=scale_backward)

            # Determine no. of parameters on local worker
            normalized_shape_local = [1] * P_x.dim
            normalized_shape_local[dim_reduce_slice] = compute_subshape(
                P_x.shape[dim_reduce_slice],
                P_x.index[dim_reduce_slice],
                normalized_shape
            )
            normalized_shape_local = tuple(normalized_shape_local)

            if P_w.active:
                self.weight = torch.nn.Parameter(torch.empty(normalized_shape_local, **factory_kwargs))
                self.bias = torch.nn.Parameter(torch.empty(normalized_shape_local, **factory_kwargs))
            else:
                self.register_buffer('weight', zero_volume_tensor(device=device, requires_grad=True, dtype=self.dtype))
                self.register_buffer('bias', zero_volume_tensor(device=device, requires_grad=True, dtype=self.dtype))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

        # State dict hooks for gather/scattering distributed weights
        self._register_state_dict_hook(self.gather_state_dict)
        self._register_load_state_dict_pre_hook(self.scatter_state_dict)

        # Partition for collecting weights/biases for saving the state dict
        if self.collect_state:
            P_root_base = P_x.create_partition_inclusive([0])
            self.P_root = P_root_base.create_cartesian_topology_partition([1] * P_x.dim)
            self.gather = Repartition(self.P_w, self.P_root, preserve_batch=False)
            self.scatter = Repartition(self.P_root, self.P_w, preserve_batch=False)

    # Initializer for parameters
    def reset_parameters(self):
        if self.elementwise_affine and self.P_w.active:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def gather_state_dict(self, module, destination, prefix, *args):
        if self.collect_state and self.elementwise_affine and self.P_x.active:

            # Pop bias from state dict and serialize it
            bias_key = next(reversed(destination))
            bias = self.gather(destination.pop(bias_key))

            # Pop weight from state dict and serialize it
            weight_key = next(reversed(destination))
            weight = self.gather(destination.pop(weight_key))

            # Serialize weights
            if self.P_root.active:

                # Bring into same shape as the serial torch version
                weight = weight.view(weight.shape[self.dim_reduce_slice])
                bias = bias.view(bias.shape[self.dim_reduce_slice])

                # Add filenames back to state dict
                destination[weight_key] = weight
                destination[bias_key] = bias

        return destination

    def scatter_state_dict(self, destination, prefix, *args):
        if self.collect_state and self.elementwise_affine and self.P_x.active:

            # Pop entries from state dict
            weight_key = next(iter(destination))
            weight = destination.pop(weight_key)
            bias_key = next(iter(destination))
            bias = destination.pop(bias_key)

            # Load states
            if self.P_root.active:
                # Bring from PyTorch into DistDL shape (add dimensions for broadcasting)
                shape_expanded = [1] * self.P_x.dim
                shape_expanded[self.dim_reduce_slice] = weight.shape
                weight = weight.view(shape_expanded)
                bias = bias.view(shape_expanded)

            else:
                weight = zero_volume_tensor(device=self.P_x.device, requires_grad=True, dtype=self.dtype)
                bias = zero_volume_tensor(device=self.P_x.device, requires_grad=True, dtype=self.dtype)

            # Scatter states
            weight = self.scatter(weight)
            bias = self.scatter(bias)

            # Add data back to state dict
            destination[weight_key] = weight
            destination[bias_key] = bias

        return destination

    def _compute_mean(self, input):
        r"""
        Compute global feature mean (i.e., across the last d dimensions,
        where d is the dimension of self.normalized_shape).
        Ensures all ranks have the mean tensor.

        Parameters
        ----------
        input :
            PyTorch Tensor of values that should be summed.
        """
        # Local mean
        output = input.mean(dim=self.dim_reduce, keepdim=True)

        # Average across workers
        return self.allreduce(output) / self.num_reduce

    def _compute_var(self, input, mean):
        r"""
        Compute global variance across last d dimensions,
        where d is the dimension of self.normalized_shape.
        Ensures all ranks have the variance tensor.

        Parameters
        ----------
        input :
            PyTorch Tensor of values for which variance should be computed.
        """
        input = (input - mean)**2
        return self._compute_mean(input)

    def forward(self, input):
        r"""Forward function interface.

        Parameters
        ----------
        input :
            Input tensor to be normalized.

        """

        if not self.P_x.active:
            return input

        # If we compute mean/variance over more than one partition, we need to
        # use our custom mean/variance implementations with allreduce. Otherwise
        # just use the torch implementation.
        if self.num_reduce > 1:

            # Calculate mean and variance
            mean = self._compute_mean(input)
            var = self._compute_var(input, mean)

            # Re-scale
            input = (input - mean) / torch.sqrt(var + self.eps)

            if self.elementwise_affine:
                weight = self.broadcast(self.weight)
                bias = self.broadcast(self.bias)
                input = weight * input + bias

        else:
            if self.elementwise_affine:
                weight = self.broadcast(self.weight).squeeze()
                bias = self.broadcast(self.bias).squeeze()
            else:
                weight = None
                bias = None
            input = torch.nn.functional.layer_norm(input, self.normalized_shape, weight, bias, self.eps)

        return input
