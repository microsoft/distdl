import numbers

import pytorch_pfn_extras as ppe
import torch

from distdl import backends
from distdl.nn.all_gather import AllGather
from distdl.nn.module import Module
from distdl.nn.repartition import Repartition
from distdl.utilities.slicing import compute_subshape
from distdl.utilities.torch import zero_volume_tensor

try:
    from flash_attn.ops.layer_norm import layer_norm as flash_layer_norm  # noqa: F401
except ImportError:
    flash_layer_norm = None


class DistributedLayerNormZero(Module):
    r"""A distributed layer norm layer with FSDP.

    Applies Layer Normalization. This layer is a distributed and generalized
    version of the PyTorch LayerNorm layer using fully-sharded data parallelism.

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
                 collect_state=False, device=None, dtype=None, scale_backward=None):
        super(DistributedLayerNormZero, self).__init__()

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

        self.use_flash = (flash_layer_norm is not None and
                          backends.backend == backends.nccl_cupy and
                          elementwise_affine)

        # Number of dimensions across which mean/var is computed
        num_dim = len(normalized_shape)

        # Dimensions across which to reduce
        self.dim_reduce = tuple(torch.arange(0, P_x.dim)[-num_dim:])

        # Number of workers across we reduce
        dim_reduce_slice = slice(P_x.dim - num_dim, P_x.dim)   # dimensions across which we compute mean/var over
        dim_bcast_slice = slice(0, P_x.dim - num_dim)  # dimensions across which we broadcast weights/biases over
        self.dim_bcast_slice = dim_bcast_slice
        self.dim_reduce_slice = dim_reduce_slice

        if self.elementwise_affine:

            # Weight/bias partition
            weight_partition_shape = [1] * P_x.dim
            weight_partition_shape[0] = P_x.size

            # Weight partition and broadcast
            P_w_base = P_x.create_partition_inclusive(range(P_x.size))
            P_w = P_w_base.create_cartesian_topology_partition(weight_partition_shape)
            P_w_base.deactivate()
            self.P_w = P_w

            # Allgather for collecting weights from data-parallel workers
            self.allgather = AllGather(P_w, axes_all_gather=(0,), scale_backward=scale_backward)

            # Split normalized work along all workers
            normalized_shape_local = [1] * P_x.dim
            normalized_shape_local[-1] = compute_subshape(
                P_w.shape[0],
                P_w.index[0],
                normalized_shape[-1],
            ).item()
            normalized_shape_local = tuple(normalized_shape_local)

            # Weights and biases
            self.weight = torch.nn.Parameter(torch.empty(normalized_shape_local, **factory_kwargs))
            self.bias = torch.nn.Parameter(torch.empty(normalized_shape_local, **factory_kwargs))

            # Buffers for weight prefetching
            self.weight_buffer = None
            self.bias_buffer = None

            # CUDA streams for weight prefetching
            if not self.P_x.device == 'cpu':
                self.stream_weight = torch.cuda.Stream(device=self.P_x.device)
                self.stream_bias = torch.cuda.Stream(device=self.P_x.device)
            else:
                self.stream_weight = None
                self.stream_bias = None
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

        # State dict hooks for gather/scattering distributed weights
        self._register_state_dict_hook(self.gather_state_dict)
        self._register_load_state_dict_pre_hook(self.scatter_state_dict)

        # Partition for collecting weights/biases for saving the state dict
        if self.elementwise_affine:
            P_root_base = P_x.create_partition_inclusive([0])
            self.P_root = P_root_base.create_cartesian_topology_partition([1] * P_x.dim)
            self.gather_affine = Repartition(self.P_w, self.P_root, preserve_batch=False)
            self.scatter_affine = Repartition(self.P_root, self.P_w, preserve_batch=False)

    # Initializer for parameters
    def reset_parameters(self):
        if self.elementwise_affine and self.P_x.active:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def gather_state_dict(self, module, destination, prefix, *args):
        if self.collect_state and self.elementwise_affine and self.P_x.active:

            # Pop bias from state dict and serialize it
            bias_key = next(reversed(destination))
            bias = self.gather_affine(destination.pop(bias_key).transpose(0, -1)).transpose(0, -1)

            # Pop weight from state dict and serialize it
            weight_key = next(reversed(destination))
            weight = self.gather_affine(destination.pop(weight_key).transpose(0, -1)).transpose(0, -1)

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
                weight = zero_volume_tensor(device=self.P_x.device, requires_grad=True, dtype=weight.dtype)
                bias = zero_volume_tensor(device=self.P_x.device, requires_grad=True, dtype=bias.dtype)

            # Scatter states
            weight = self.scatter_affine(weight.transpose(0, -1)).transpose(0, -1)
            bias = self.scatter_affine(bias.transpose(0, -1)).transpose(0, -1)

            # Add data back to state dict
            destination[weight_key] = weight
            destination[bias_key] = bias

        return destination

    def prefetch_weights(self):
        if self.P_x.size == 1:
            return
        if self.stream_weight is not None:
            with ppe.cuda.stream(self.stream_weight):
                self.weight_buffer = self.allgather(self.weight.transpose(0, -1)).transpose(0, -1)
        else:
            self.weight_buffer = self.allgather(self.weight.transpose(0, -1)).transpose(0, -1)

        if self.stream_bias is not None:
            with ppe.cuda.stream(self.stream_bias):
                self.bias_buffer = self.allgather(self.bias.transpose(0, -1)).transpose(0, -1)
        else:
            self.bias_buffer = self.allgather(self.bias.transpose(0, -1)).transpose(0, -1)

    def forward(self, input):
        r"""Forward function interface.

        Parameters
        ----------
        input :
            Input tensor to be normalized.

        """

        if not self.P_x.active:
            return input

        # Collect weights and biases
        if self.elementwise_affine:
            if self.weight_buffer is None:
                weight = self.allgather(self.weight.transpose(0, -1)).transpose(0, -1)
                bias = self.allgather(self.bias.transpose(0, -1)).transpose(0, -1)
            else:
                weight = self.weight_buffer
                bias = self.bias_buffer
                self.weight_buffer = None
                self.bias_buffer = None

        if self.elementwise_affine:
            weight = weight.squeeze()
            bias = bias.squeeze()
            if self.stream_weight is not None and self.stream_bias is not None:
                torch.cuda.current_stream().wait_stream(self.stream_weight)
                torch.cuda.current_stream().wait_stream(self.stream_bias)
        else:
            weight = None
            bias = None

        if self.use_flash:
            input = flash_layer_norm(input, weight, bias, self.eps)
        else:
            input = torch.nn.functional.layer_norm(input, self.normalized_shape, weight, bias, self.eps)

        return input
