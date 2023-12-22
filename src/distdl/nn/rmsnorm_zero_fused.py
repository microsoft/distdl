import numbers

import numpy as np
import pytorch_pfn_extras as ppe
import torch
from flash_attn.ops.rms_norm import rms_norm

from distdl.nn.all_gather import AllGather
from distdl.nn.module import Module
from distdl.nn.repartition import Repartition
from distdl.utilities.slicing import compute_subshape
from distdl.utilities.torch import zero_volume_tensor


class DistributedFusedRMSNormZero(Module):
    r"""A distributed fused RMS normalization layer with FSDP.

    Applies RMS Normalization using fully-sharded data parallelism based on Flash Attention's
    fused RMSNorm implementation.

    Note: This version does not support normalized_shape spanning partitioned dimensions.

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

    def __init__(self, P_x, normalized_shape, eps=1e-5, collect_state=False,
                 device=None, dtype=None, scale_backward=None):
        super(DistributedFusedRMSNormZero, self).__init__()

        self.P_x = P_x
        if not self.P_x.active:
            return

        if device is None:
            device = P_x.device
        self.eps = eps
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

        # Number of workers across we reduce
        self.dim_reduce_slice = slice(P_x.dim - num_dim, P_x.dim)   # dimensions across which we compute mean/var over
        self.num_reduce = np.prod(P_x.shape[self.dim_reduce_slice])
        if self.num_reduce > 1:
            raise ValueError("FusedRMSNormZero does not support normalized_shape spanning partitioned dimensions.")

        # Allgather for collecting weights from data-parallel workers
        self.allgather = AllGather(P_x, axes_all_gather=(0,), scale_backward=scale_backward)

        # Shard weights across data-parallel workers (ZeRO-3 style)
        normalized_shape_local = list([1] * P_x.dim)
        normalized_shape_local[-1] = compute_subshape(
            P_x.shape[0],
            P_x.index[0],
            normalized_shape[-1]
        ).item()
        normalized_shape_local = tuple(normalized_shape_local)

        # Weights and bias
        self.weight = torch.nn.Parameter(torch.empty(normalized_shape_local, **factory_kwargs))

        # Buffers for weight prefetching
        self.weight_buffer = None

        # CUDA streams for weight prefetching
        if not self.P_x.device == 'cpu':
            self.stream_weight = torch.cuda.Stream(device=self.P_x.device)
        else:
            self.stream_weight = None

        self.reset_parameters()

        # State dict hooks for gather/scattering distributed weights
        self._register_state_dict_hook(self.gather_state_dict)
        self._register_load_state_dict_pre_hook(self.scatter_state_dict)

        # Partition for collecting weights for saving the state dict
        P_root_base = P_x.create_partition_inclusive([0])
        self.P_root = P_root_base.create_cartesian_topology_partition([1] * P_x.dim)
        self.scatter_dp = Repartition(self.P_root, self.P_x, preserve_batch=False)

    # Initializer for parameters
    def reset_parameters(self):
        if self.P_x.active:
            torch.nn.init.ones_(self.weight)

    def gather_state_dict(self, module, destination, prefix, *args):
        if self.collect_state and self.P_x.active:

            # Pop weight from state dict and serialize it
            weight_key = next(reversed(destination))
            weight = self.allgather(destination.pop(weight_key).transpose(0, -1)).transpose(0, -1)

            # Serialize weights
            if self.P_root.active:

                # Bring into same shape as the serial torch version
                weight = weight.view(weight.shape[self.dim_reduce_slice])

                # Add filenames back to state dict
                destination[weight_key] = weight

        return destination

    def scatter_state_dict(self, destination, prefix, *args):
        if self.collect_state and self.P_x.active:

            # Pop entries from state dict
            weight_key = next(iter(destination))
            weight = destination.pop(weight_key)

            # Load states
            if self.P_root.active:

                # Bring from PyTorch into DistDL shape (add dimensions for broadcasting)
                shape_expanded = [1] * self.P_x.dim
                shape_expanded[self.dim_reduce_slice] = weight.shape
                weight = weight.view(shape_expanded)
            else:
                weight = zero_volume_tensor(device=self.P_x.device, requires_grad=True, dtype=weight.dtype)

            # Scatter states
            weight = self.scatter_mp(weight)
            weight = self.scatter_dp(weight.transpose(0, -1)).transpose(0, -1)

            # Add data back to state dict
            destination[weight_key] = weight

        return destination

    def prefetch_weights(self):
        if self.P_x.size == 1:
            return

        if self.stream_weight is not None:
            with ppe.cuda.stream(self.stream_weight):
                self.weight_buffer = self.allgather(self.weight.transpose(0, -1)).transpose(0, -1)
        else:
            self.weight_buffer = self.allgather(self.weight.transpose(0, -1)).transpose(0, -1)

    def forward(self, input):
        r"""Forward function interface.

        Parameters
        ----------
        input :
            Input tensor to be normalized.

        """

        if not self.P_x.active:
            return input

        # Gather weights from data-parallel workers
        if self.weight_buffer is None:
            weight = self.allgather(self.weight.transpose(0, -1)).transpose(0, -1)
        else:
            weight = self.weight_buffer
            self.weight_buffer = None

        if self.stream_weight is not None:
            torch.cuda.current_stream().wait_stream(self.stream_weight)

        return rms_norm(input, weight, self.eps)
