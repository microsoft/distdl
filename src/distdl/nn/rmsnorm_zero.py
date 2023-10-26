import numbers

import numpy as np
import torch

from distdl.nn.all_gather import AllGather
from distdl.nn.all_sum_reduce import AllSumReduce
from distdl.nn.module import Module
from distdl.nn.repartition import Repartition
from distdl.utilities.slicing import compute_subshape
from distdl.utilities.slicing import worker_layout
from distdl.utilities.torch import zero_volume_tensor


class DistributedRMSNormZero(Module):
    r"""A distributed RMS normalization layer with FSDP and model parallelism.

    Applies RMS Normalization using fully-sharded data and tensor parallelism.

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
    bias : optional
        A boolean value that when set to True, this module has learnable bias parameters. Default is
        False. If elementwise_affine is False, this setting is ignored.
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

    def __init__(self, P_x, normalized_shape, elementwise_affine=True, bias=False, eps=1e-5,
                 collect_state=False, device=None, dtype=None, scale_backward=None):
        super(DistributedRMSNormZero, self).__init__()

        self.P_x = P_x
        if not self.P_x.active:
            return

        if device is None:
            device = P_x.device
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.use_bias = bias
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

            # Weight/bias partition
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

            # Allgather for collecting weights from data-parallel workers
            self.allgather = AllGather(P_x, axes_all_gather=(0,), scale_backward=scale_backward)

            # Split normalized work along model-parallel workers
            normalized_shape_mp = [1] * P_x.dim
            normalized_shape_mp[dim_reduce_slice] = compute_subshape(
                P_x.shape[dim_reduce_slice],
                P_x.index[dim_reduce_slice],
                normalized_shape
            )
            normalized_shape_mp = tuple(normalized_shape_mp)

            # Additionally, shard the last dimension across data-parallel workers
            normalized_shape_local = list(normalized_shape_mp)
            normalized_shape_local[-1] = compute_subshape(
                P_x.shape[0],
                P_x.index[0],
                normalized_shape_mp[-1]
            ).item()
            normalized_shape_local = tuple(normalized_shape_local)

            # Weights and bias
            self.weight = torch.nn.Parameter(torch.empty(normalized_shape_local, **factory_kwargs))
            if self.use_bias:
                self.bias = torch.nn.Parameter(torch.empty(normalized_shape_local, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
        self.reset_parameters()

        # State dict hooks for gather/scattering distributed weights
        self._register_state_dict_hook(self.gather_state_dict)
        self._register_load_state_dict_pre_hook(self.scatter_state_dict)

        # Partition for collecting weights for saving the state dict
        if self.elementwise_affine:
            P_root_base = P_x.create_partition_inclusive([0])
            self.P_root = P_root_base.create_cartesian_topology_partition([1] * P_x.dim)
            self.gather = Repartition(self.P_w, self.P_root, preserve_batch=False)
            self.scatter_mp = Repartition(self.P_root, self.P_w, preserve_batch=False)
            self.scatter_dp = Repartition(self.P_w, self.P_x, preserve_batch=False)

    # Initializer for parameters
    def reset_parameters(self):
        if self.elementwise_affine and self.P_x.active:
            torch.nn.init.ones_(self.weight)
            if self.use_bias:
                torch.nn.init.zeros_(self.bias)

    def gather_state_dict(self, module, destination, prefix, *args):
        if self.collect_state and self.elementwise_affine and self.P_x.active:

            # Pop bias from state dict and serialize it
            if self.use_bias:
                bias_key = next(reversed(destination))
                bias = self.allgather(destination.pop(bias_key).transpose(0, -1)).transpose(0, -1)
                bias = self.gather(bias)

            # Pop weight from state dict and serialize it
            weight_key = next(reversed(destination))
            weight = self.allgather(destination.pop(weight_key).transpose(0, -1)).transpose(0, -1)
            weight = self.gather(weight)

            # Serialize weights
            if self.P_root.active:

                # Bring into same shape as the serial torch version
                weight = weight.view(weight.shape[self.dim_reduce_slice])
                if self.use_bias:
                    bias = bias.view(bias.shape[self.dim_reduce_slice])

                # Add filenames back to state dict
                destination[weight_key] = weight
                if self.use_bias:
                    destination[bias_key] = bias

        return destination

    def scatter_state_dict(self, destination, prefix, *args):
        if self.collect_state and self.elementwise_affine and self.P_x.active:

            # Pop entries from state dict
            weight_key = next(iter(destination))
            weight = destination.pop(weight_key)
            if self.use_bias:
                bias_key = next(iter(destination))
                bias = destination.pop(bias_key)

            # Load states
            if self.P_root.active:
                # Bring from PyTorch into DistDL shape (add dimensions for broadcasting)
                shape_expanded = [1] * self.P_x.dim
                shape_expanded[self.dim_reduce_slice] = weight.shape
                weight = weight.view(shape_expanded)
                if self.use_bias:
                    bias = bias.view(shape_expanded)
            else:
                weight = zero_volume_tensor(device=self.P_x.device, requires_grad=True, dtype=weight.dtype)
                if self.use_bias:
                    bias = zero_volume_tensor(device=self.P_x.device, requires_grad=True, dtype=bias.dtype)

            # Scatter states
            weight = self.scatter_mp(weight)
            weight = self.scatter_dp(weight.transpose(0, -1)).transpose(0, -1)
            if self.use_bias:
                bias = self.scatter_mp(bias)
                bias = self.scatter_dp(bias.transpose(0, -1)).transpose(0, -1)

            # Add data back to state dict
            destination[weight_key] = weight
            if self.use_bias:
                destination[bias_key] = bias

        return destination

    def _rms_norm(self, input):
        r"""
        Compute global feature mean (i.e., across the last d dimensions,
        where d is the dimension of self.normalized_shape).
        Ensures all ranks have the mean tensor.

        Parameters
        ----------
        input :
            PyTorch Tensor of values that should be summed.
        """
        # Mean across all workers
        mean = input.pow(2).mean(dim=self.dim_reduce, keepdim=True)
        mean = self.allreduce(mean) / self.num_reduce

        return input * torch.rsqrt(mean + self.eps)

    def forward(self, input):
        r"""Forward function interface.

        Parameters
        ----------
        input :
            Input tensor to be normalized.

        """

        if not self.P_x.active:
            return input

        # Calculate RMS
        input = self._rms_norm(input.float())

        if self.elementwise_affine:
            weight = self.allgather(self.weight.transpose(0, -1)).transpose(0, -1)
            input = weight * input
            if self.use_bias:
                bias = self.allgather(self.bias.transpose(0, -1)).transpose(0, -1)
                input += bias

        return input.to(self.dtype)
