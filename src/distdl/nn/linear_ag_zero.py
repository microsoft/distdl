import math
from contextlib import nullcontext

import numpy as np
import pytorch_pfn_extras as ppe
import torch
from einops import rearrange

import distdl.nn.init as init
from distdl.backends.common.tensor_comm import assemble_global_tensor_structure
from distdl.nn.all_gather import AllGather
from distdl.nn.module import Module
from distdl.nn.reduce_scatter import ReduceScatter
from distdl.nn.repartition import Repartition
from distdl.utilities.misc import stream_barrier
from distdl.utilities.slicing import compute_subshape
from distdl.utilities.slicing import worker_layout
from distdl.utilities.torch import zero_volume_tensor


# Custom forward/backward functions
class LinearAllGatherZeROFunc(torch.autograd.Function):

    @staticmethod
    def forward(input, weight, bias, ag_input, rs_input, ag_weight, rs_weight, ag_bias, rs_bias, scale_backward):

        # Gather inputs
        input = ag_input(input)
        weight = ag_weight(weight).squeeze(2)

        # Broadcast bias
        if bias is not None:
            bias = ag_bias(bias.transpose(0, -2)).transpose(0, -2).view(1, 1, -1)

        # Affine layer
        return torch.einsum('bij,jk->bik', input, weight) + bias

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, weight, bias, ag_input, rs_input, ag_weight, rs_weight, ag_bias, rs_bias, scale_backward = inputs
        ctx.save_for_backward(input, weight, bias)
        ctx.constant = (ag_input, rs_input, ag_weight, rs_weight, rs_bias, scale_backward)

    @staticmethod
    def backward(ctx, grad_output):

        # Load saved tensors and operators
        input, weight, bias = ctx.saved_tensors
        ag_input, rs_input, ag_weight, rs_weight, rs_bias, scale_backward = ctx.constant

        # Input gradient
        if ctx.needs_input_grad[0]:
            weight = ag_weight(weight).squeeze(2)
            grad_input = torch.einsum('bij,kj->bik', grad_output, weight)
            grad_input = rs_input(grad_input)
        else:
            grad_input = None

        # Weight gradient
        if ctx.needs_input_grad[1]:
            input = ag_input(input)
            grad_weight = torch.einsum('bij,bik->jk', input, grad_output).unsqueeze(2)
            if scale_backward is not None:
                grad_weight.div_(np.prod(rs_weight.P_reducescatter.shape[scale_backward]))
            grad_weight = rs_weight(grad_weight)
        else:
            grad_weight = None

        # Bias gradient
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0, 1)).view(1, -1, 1)
            if scale_backward is not None:
                grad_bias.div_(np.prod(rs_bias.P_reducescatter.shape[scale_backward]))
            grad_bias = rs_bias(grad_bias.transpose(0, -2)).transpose(0, -2)
        else:
            grad_bias = None

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None


class DistributedLinearAllGatherZero(Module):
    r"""A distributed linear or affine layer with 2D parallelism for weights
    and input/outputs (also called ZeRO-3 or FSDP).

    This class provides the user interface to a distributed linear layer
    with 2D partitioning of input/output data and 2D partitioning of weights
    and biases. Outputs can be partitioned along the batch dimension (dimension 0)
    and/or the last dimension, as specified by the output partition P_y.

    Inputs can be partitioned along the batch dimension (dimension 0) plus either
    the last dimension or second last dimension. If inputs are partitioned along
    the second last dimension, an additional input partition P_x must be specified.
    If P_x is not supplied, the input partitoning is assumed to be the same as the
    output partitioning.

    Weights and biases are partitoned along both the input and output feature dimension.
    Input features are partitioned along the first dimension (usually the data-parallel
    dimension), and output features are partitioned along the last dimension of P_y.
    In the forward pass, a weight-allgather is performed along the first dimension,
    and data-allgather along the last dimension (usually the model-parallel dimension).
    This version is preferred when the input feature dimension is smaller than the output
    feature dimension. For the reverse case, see DistributedLinearReduceScatterZero.

    This class supports computing QKV tensors for multi-head attention. If
    collect_state is set to true, the number of attention heads must be specified,
    as well as the number of output variables (3 for QKV, 2 for QK, 1 for Q only).
    Supplying the no. of heads and output variables is required to rearrange
    the weights and biases such that they yield the same result if the layer is
    called for a different number of (model-parallel) workers (e.g., if we load
    trained weights on a single GPU for inference).

    Parameters
    ----------
    P_y :
        Partition of input/output tensor with shape [ D, ..., 1, M ], where D is
        the no. of data parallel workers and M is the no. of model parallel workers.
    in_features :
        Number of features in the *global* input tensor.
    out_features :
        Number of features in the *global* output tensor.
    bias : bool
        Indicates if a bias term should be used.
    P_x : optional
        Partition of the input tensor if input is partitioned along the second
        last dimension. Must have shape of form [ D, ..., M, 1 ].
    P_weight : optional
        Partition for storing weights and biases with shape [ D, ..., M, 1 ].
    P_store_bias : optional
        Partition for storing biases with shape [ 1, ..., M, 1 ].
    collect_state: bool, optional
        If true, collects the weights and biases to the root worker and
        serializes them to disk when the state_dict() function is called.
        Instead of the weights and biases, the state dictionary contains
        paths to those files. Default is false.
    num_heads: int, optional
        Total number of attention heads across all workers for multi-head attention.
        Only required if collect_state=True. Default is 0.
    num_heads_kv: int, optional
        Number of attention heads for key and value tensors. Only required if
        collect_state=True. Default is 0.
    num_vars: int, optional
        Number of output variables if used as a linear layer for QKV computations.
        Set to 3 for QKV, 2 for KV, and 1 for Q. Only required if collect_state=True.
        Default is 3 (QKV).
    geglu: bool, optional
        Set to true if a gated linear unit is used directly after the linear layer and
        collect_state=True. Default is False.
    checkpoint : bool, optional
        If true, use custom backward implementation that recomputes the
        all-gathers in the backward pass.
    scale_backward : Union[int, slice], optional
        Scale backward pass for AllGather operation by no. of workers along the given
        dimension. Default is None.
    auto_clear_buffer: bool, optional
        If true, clears the weight buffers after each forward pass. Default is True.
        For ZeRO stage 1 and to take advantage of gradient accumulation, set this
        to False and call clear_weight_buffer() manually after the optimizer step.
    """

    def __init__(self, P_y, in_features, out_features, bias=True, device=None, dtype=None,
                 P_x=None, P_store_bias=None, P_weight=None, collect_state=False, num_heads=None,
                 num_heads_kv=None, num_vars=3, geglu=False, checkpoint=False, scale_backward=None,
                 auto_clear_buffer=True):

        super(DistributedLinearAllGatherZero, self).__init__()

        # P_y is assumed to have shape [ *, 1, p]
        # Data is assumed to have shape [ *, n, channel_in/p ]
        self.P_y = P_y
        if not self.P_y.active:
            return
        else:
            assert P_y.shape[-2] == 1

        # Input partition can be different than output partition
        # (i.e. if input is partitioned along tokens)
        if P_x is None:
            self.P_x = P_y
        else:
            assert P_x.dim == P_y.dim
            assert P_x.shape[-2] == P_y.shape[-1]
            assert P_x.shape[-1] == 1
            self.P_x = P_x

        if device is None:
            device = P_y.device
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.in_features = in_features
        self.out_features = out_features
        self.collect_state = collect_state
        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv
        self.num_vars = num_vars
        self.geglu = geglu
        self.use_bias = bias
        self.dtype = dtype
        self.checkpoint = checkpoint
        self.scale_backward = scale_backward
        self.auto_clear_buffer = auto_clear_buffer

        # Partition for storing weights & biases
        if P_store_bias is not None:
            assert P_y.dim == P_store_bias.dim
            assert P_store_bias.shape[-2] == P_y.shape[-1]
            assert P_store_bias.shape[-1] == 1
            if P_y.dim > 2:
                assert np.prod(P_store_bias.shape[:P_y.dim - 2]) == 1
        else:
            store_weight_partition_shape = [1] * P_y.dim
            store_weight_partition_shape[-2] = P_y.shape[-1]

            index_store_weight = [slice(0, 1)] * P_y.dim
            index_store_weight[-1] = slice(0, P_y.shape[-1])
            store_weight_workers = worker_layout(P_y.shape)[tuple(index_store_weight)].\
                reshape(-1).tolist()

            P_store_bias_base = P_y.create_partition_inclusive(store_weight_workers)
            P_store_bias = P_store_bias_base.create_cartesian_topology_partition(
                store_weight_partition_shape)
            P_store_bias_base.deactivate()

        # Partition for applying weights & biases (same size as P_y, but with last
        # two dims swapped).
        if P_weight is not None:
            assert P_y.dim == P_weight.dim
            assert P_weight.shape[-1] == 1
            assert P_weight.shape[-2] == P_y.shape[-1]
            for i in range(P_y.dim - 2):
                assert P_weight.shape[i] == P_y.shape[i]
        else:
            apply_weight_partition_shape = P_y.shape.copy()
            apply_weight_partition_shape[-1] = 1
            apply_weight_partition_shape[-2] = P_y.shape[-1]

            P_weight_base = P_y.create_partition_inclusive(range(P_y.size))
            P_weight = P_weight_base.create_cartesian_topology_partition(
                apply_weight_partition_shape)
            P_weight_base.deactivate()

        # Store partitions for later  access
        self.P_store_bias = P_store_bias
        self.P_weight = P_weight

        # Function to gather weights and biases
        self.allgather_weight = AllGather(P_weight, axes_all_gather=(0,), scale_backward=scale_backward)
        self.reduce_scatter_weight = ReduceScatter(P_weight, axes_reduce_scatter=(0,))
        if bias:
            self.allgather_bias = AllGather(P_weight, axes_all_gather=(0,), scale_backward=scale_backward)
            self.reducescatter_bias = ReduceScatter(P_weight, axes_reduce_scatter=(0,))

        # Create weights
        if P_weight.active:

            # Local shape of weights, which must have the same no. of dimensions as P_y
            weight_shape = [1] * P_y.dim
            in_features_local = compute_subshape(P_weight.shape[0],
                                                 P_weight.index[0],
                                                 [in_features])[0]
            out_features_local = compute_subshape(P_weight.shape[-2],
                                                  P_weight.index[-2],
                                                  [out_features])[0]
            weight_shape[0] = in_features_local
            weight_shape[-2] = out_features_local

            # Create weights. Every worker either has weights or receives weights.
            self.weight = torch.nn.Parameter(torch.empty(tuple(weight_shape), **factory_kwargs))
        else:
            self.register_buffer('weight', zero_volume_tensor(device=device, requires_grad=True, dtype=self.dtype))

        # Create bias
        if self.use_bias and P_weight.active:
            bias_shape = [1] * P_y.dim
            out_features_bias_local = compute_subshape(P_weight.shape[0],
                                                       P_weight.index[0],
                                                       [out_features_local])[0]
            bias_shape[-2] = out_features_bias_local
            self.bias = torch.nn.Parameter(torch.empty(tuple(bias_shape), **factory_kwargs))
        elif self.use_bias:
            self.register_buffer('bias', zero_volume_tensor(device=device, requires_grad=True, dtype=self.dtype))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters
        self.reset_parameters()

        # All-gather operation
        gather_dim = torch.argmax(torch.tensor(self.P_x.shape[-2:])) + self.P_x.dim - 2
        self.all_gather = AllGather(self.P_x, axes_all_gather=(gather_dim,))
        self.reduce_scatter = ReduceScatter(self.P_x, axes_reduce_scatter=(gather_dim,))

        # CUDA streams for weight prefetching. Only used if cuda is enabled.
        if not self.P_y.device == 'cpu':
            self.stream_context = nullcontext  # ppe.cuda.stream TODO Fix
            self.stream_weight = torch.cuda.Stream(device=self.P_y.device)
            if self.use_bias:
                self.stream_bias = torch.cuda.Stream(device=self.P_y.device)
        else:
            self.stream_context = nullcontext
            self.stream_weight = None
            if self.use_bias:
                self.stream_bias = None

        # Buffers for weight prefetching
        self.weight_buffer = None
        self.bias_buffer = None

        # State dict hooks for gather/scattering distributed weights
        self._register_state_dict_hook(self.gather_state_dict)
        self._register_load_state_dict_pre_hook(self.scatter_state_dict)

        # Partition for collecting weights/biases for saving the state dict
        P_root_base = P_y.create_partition_inclusive([0])
        self.P_root = P_root_base.create_cartesian_topology_partition([1] * P_y.dim)
        self.gather_weight = Repartition(P_weight, self.P_root, preserve_batch=False)
        self.scatter_weight = Repartition(self.P_root, P_weight, preserve_batch=False)
        if self.use_bias:
            self.gather_bias = Repartition(P_store_bias, self.P_root, preserve_batch=False)
            self.scatter_bias_mp = Repartition(self.P_root, P_store_bias, preserve_batch=False)
            self.scatter_bias_dp = Repartition(self.P_store_bias, P_weight, preserve_batch=False)

    def reset_parameters(self) -> None:

        if self.P_weight.active:
            init.kaiming_uniform_(self.P_weight, self.weight, a=math.sqrt(5))
            weight_global_shape = assemble_global_tensor_structure(
                self.weight, self.P_weight).shape

        if self.P_store_bias.active and self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(weight_global_shape)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

    def _unsqueeze_weight(self, weight):
        shape = [1] * self.P_y.dim
        shape[0] = weight.shape[0]
        shape[1] = weight.shape[1]
        return weight.view(shape)

    def _squeeze_weight(self, weight):
        c_in = weight.shape[0]
        c_out = weight.shape[-2]
        return weight.view(c_in, c_out)

    def _unsqueeze_bias(self, bias):
        shape = [1] * self.P_y.dim
        shape[-2] = bias.shape[0]
        return bias.view(shape)

    def _squeeze_bias(self, bias):
        c_out = bias.shape[-2]
        return bias.view(c_out)

    # If we collect the weights on the root worker, we need to rearrange the weights,
    # such that the split into QKV occurs in the 2nd slowest dimension (dim 1). This
    # enables us to load weights for a different partitioning scheme than they were
    # saved in.
    def qkv_weight_to_serial(self, weight):
        if self.num_heads_kv is None:
            head_size = weight.shape[-2] // self.num_vars // self.num_heads
            num_gpu = self.P_weight.shape[-2]
            weight = rearrange(self._squeeze_weight(weight), "n (p v h) -> n (v p h)",
                               p=num_gpu, v=self.num_vars, h=self.num_heads // num_gpu * head_size)
            return self._unsqueeze_weight(weight)
        else:
            head_size = weight.shape[-2] // (self.num_heads_kv * 2 + self.num_heads)
            num_heads_local = compute_subshape(self.P_weight.shape[-2], self.P_weight.index[-2], [self.num_heads])[0]
            num_heads_kv_local = compute_subshape(self.P_weight.shape[-2], self.P_weight.index[-2],
                                                  [self.num_heads_kv])[0]
            q_size_local = head_size * num_heads_local
            kv_size_local = head_size * num_heads_kv_local * 2
            num_gpu = self.P_weight.shape[-2]

            # Split into Q and KV components
            weight = rearrange(self._squeeze_weight(weight), "n (p m) -> n p m",
                               p=num_gpu, m=q_size_local + kv_size_local)
            q_weight = weight[:, :, :q_size_local]
            kv_weight = weight[:, :, q_size_local:]

            # Rearrange
            q_weight = rearrange(q_weight, "n p (v h) -> n (v p h)", v=1, h=num_heads_local * head_size)
            kv_weight = rearrange(kv_weight, "n p (v h) -> n (v p h)", v=2, h=num_heads_kv_local * head_size)
            weight = torch.cat([q_weight, kv_weight], dim=1)

            return self._unsqueeze_weight(weight)

    # Similarly, if we want to load weights from a serial partitioning scheme and
    # use them in a parallel scheme, we need to rearrange the weights to move the
    # QKV/QK split into the 3rd slowest dimension (dim 2).
    def qkv_weight_to_parallel(self, weight):
        if self.num_heads_kv is None:
            head_size = weight.shape[-2] // self.num_vars // self.num_heads
            num_gpu = self.P_weight.shape[-2]
            weight = rearrange(self._squeeze_weight(weight), "n (v p h) -> n (p v h)",
                               p=num_gpu, v=self.num_vars, h=self.num_heads // num_gpu * head_size)
            return self._unsqueeze_weight(weight)
        else:
            head_size = weight.shape[-2] // (self.num_heads_kv * 2 + self.num_heads)
            num_heads_local = compute_subshape(self.P_weight.shape[-2], self.P_weight.index[-2], [self.num_heads])[0]
            num_heads_kv_local = compute_subshape(self.P_weight.shape[-2], self.P_weight.index[-2],
                                                  [self.num_heads_kv])[0]
            q_size = head_size * self.num_heads
            num_gpu = self.P_weight.shape[-2]

            # Split into Q and KV components
            q_weight = self._squeeze_weight(weight)[:, :q_size]
            kv_weight = self._squeeze_weight(weight)[:, q_size:]

            # Rearrange
            q_weight = rearrange(q_weight, "n (v p h) -> n p (v h)", v=1, h=num_heads_local * head_size)
            kv_weight = rearrange(kv_weight, "n (v p h) -> n p (v h)", v=2, h=num_heads_kv_local * head_size)
            weight = torch.cat([q_weight, kv_weight], dim=2)
            weight = rearrange(weight, "n p m -> n (p m)")

            return self._unsqueeze_weight(weight)

    # If we collect the weights on the root worker and want to use a gated linear
    # unit right after the linear layer, we need to rearrange the weights, such that
    # the behavior on a single GPU is the same as on multiple GPUs.
    def geglu_weight_to_serial(self, weight):
        num_gpu = self.P_weight.shape[-2]
        weight_size = weight.shape[-2] // 2 // num_gpu
        weight = rearrange(self._squeeze_weight(weight), "n (p v h) -> n (v p h)",
                           p=num_gpu, v=2, h=weight_size)
        return self._unsqueeze_weight(weight)

    # Rearrangment function for loading weights from a serial partitioning scheme
    # if a gated linear unit is used right after the linear layer.
    def geglu_weight_to_parallel(self, weight):
        num_gpu = self.P_weight.shape[-2]
        weight_size = weight.shape[-2] // 2 // num_gpu
        weight = rearrange(self._squeeze_weight(weight), "n (v p h) -> n (p v h)",
                           p=num_gpu, v=2, h=weight_size)
        return self._unsqueeze_weight(weight)

    def gather_state_dict(self, module, destination, prefix, *args):

        if self.collect_state and self.P_y.active:
            if self.use_bias:

                # Collect bias and serialize (last entry added to dict).
                # All workers should pop their bias from the state dict.
                bias_key = next(reversed(destination))
                bias = self.allgather_bias(destination.pop(bias_key).transpose(0, -2)).transpose(0, -2)
                bias = self.gather_bias(bias)

                if self.P_root.active:
                    if self.num_heads is not None:
                        bias = self.qkv_weight_to_serial(bias)
                    if self.geglu:
                        bias = self.geglu_weight_to_serial(bias)

            # Collect weights and serialize (second last entry added to dict)
            weight_key = next(reversed(destination))
            weight = self.gather_weight(destination.pop(weight_key))

            if self.P_root.active:
                if self.num_heads is not None:
                    weight = self.qkv_weight_to_serial(weight)   # [ c_in, c_out, 1]
                if self.geglu:
                    weight = self.geglu_weight_to_serial(weight)

                # Save filenames in state dict rather than the full weights. Only the root
                # should have the keys in the end.
                weight = self._squeeze_weight(weight).permute(1, 0)
                destination[weight_key] = weight

                if self.use_bias:
                    destination[bias_key] = self._squeeze_bias(bias)

        return destination

    def scatter_state_dict(self, destination, prefix, *args):
        if self.collect_state and self.P_y.active:

            # Scatter weights
            weight_key = next(iter(destination))
            weight = destination.pop(weight_key)
            if self.P_root.active:
                weight = self._unsqueeze_weight(weight.permute(1, 0))
                if self.num_heads is not None:
                    weight = self.qkv_weight_to_parallel(weight)
                if self.geglu:
                    weight = self.geglu_weight_to_parallel(weight)
            else:
                weight = zero_volume_tensor(device=self.P_y.device, requires_grad=True, dtype=weight.dtype)
            if self.P_weight.active:
                weight = self.scatter_weight(weight)

            # Scatter bias
            if self.use_bias:
                bias_key = next(iter(destination))
                bias = destination.pop(bias_key)
                if self.P_root.active:
                    bias = self._unsqueeze_bias(bias)
                    if self.num_heads is not None:
                        bias = self.qkv_weight_to_parallel(bias)
                    if self.geglu:
                        bias = self.geglu_weight_to_parallel(bias)
                elif self.P_weight.active:
                    bias = zero_volume_tensor(device=self.P_y.device, requires_grad=True, dtype=bias.dtype)
                if self.P_weight.active:
                    bias = self.scatter_bias_mp(bias)
                    if self.P_store_bias.active:
                        bias = bias.transpose(0, -2)
                    bias = self.scatter_bias_dp(bias).transpose(0, -2)
                if self.P_weight.active:
                    destination[bias_key] = bias

            # Add scattered weight to state dict
            if self.P_y.active:
                destination[weight_key] = weight

        return destination

    def collect_weights(self):

        # If weight buffer is not already filled, start an allgather call. If cuda is used,
        # this call will be asynchronously executed in a separate stream.
        if self.weight_buffer is None:
            with self.stream_context(self.stream_weight):
                self.weight_buffer = self.allgather_weight(self.weight).transpose(-1, 0).view(-1, self.in_features)

        # Same for this bias buffer if bias is used.
        if self.bias is not None and self.bias_buffer is None:
            with self.stream_context(self.stream_bias):
                self.bias_buffer = self.allgather_bias(self.bias.transpose(0, -2)).transpose(0, -2).view(-1)

    def prefetch_weights(self):     # for backward compatibility
        self.collect_weights()

    def clear_weight_buffer(self):
        self.weight_buffer = None
        self.bias_buffer = None

    def wait_for_streams(self):
        stream_barrier(self.stream_weight)
        if self.use_bias:
            stream_barrier(self.stream_bias)

    def forward(self, input):
        r"""Forward function interface.

        Parameters
        ----------
        input :
            Input tensor to the convolution.

        """

        if not self.P_y.active:
            return input

        if self.checkpoint:
            return LinearAllGatherZeROFunc.apply(
                input,
                self.weight,
                self.bias,
                self.all_gather,
                self.reduce_scatter,
                self.allgather_weight,
                self.reduce_scatter_weight,
                self.allgather_bias,
                self.reducescatter_bias,
                self.scale_backward
            )
        else:

            # All-gather weights & bias. If prefetch_weights() has been called before,
            # this call doesn't do anything.
            self.collect_weights()

            # All-gather input (tensor parallelism)
            input = self.all_gather(input)

            # Wait for all-gathers to finish
            self.wait_for_streams()

            # Affine/linear transform
            input = torch.nn.functional.linear(input, self.weight_buffer, self.bias_buffer)

            # Clear weight buffers
            if self.auto_clear_buffer:
                self.clear_weight_buffer()

            return input
