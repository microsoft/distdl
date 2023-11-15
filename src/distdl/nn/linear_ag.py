import numpy as np
import torch, math

from distdl.backends.common.tensor_comm import assemble_global_tensor_structure
from distdl.nn.module import Module
from distdl.nn.all_gather import AllGather
from distdl.utilities.slicing import compute_subshape
from distdl.utilities.torch import TensorStructure
from distdl.utilities.slicing import worker_layout
from distdl.nn.repartition import Repartition
from distdl.nn.broadcast import Broadcast
from distdl.utilities.torch import zero_volume_tensor
import distdl.nn.init as init
from einops import rearrange

class DistributedLinearAllGather(Module):
    r"""A distributed linear or affine layer with weight column parallelism.

    This class provides the user interface to a distributed linear layer
    with 2D partitioning of input/output data and 1D partitioning of weights
    and biases. Outputs can be partitioned along the batch dimension (dimension 0)
    and/or the last dimension, as specified by the output partition P_y.

    Inputs can be partitioned along the batch dimension (dimension 0) plus either
    the last dimension or second last dimension. If inputs are partitioned along
    the second last dimension, an additional input partition P_x must be specified.
    If P_x is not supplied, the input partitoning is assumed to be the same as the
    output partitioning.

    Weights and biases are partitoned along the output feature dimension. Therefore,
    an allgather is performed on the input prior to the matrix multiplication. For
    this reason, this layer is preferrable when the input feature dimension is
    smaller than the output feature dimension. For the reverse case, see
    DistributedLinearReduceScatter. Weights and biases are stored on the 1st data-
    parallel worker only.

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
        Partition of the input tensor if input is partitioned
        along the second last dimension. Shape of [ D, ..., M, 1 ].
    P_store_weight : optional
        Partition for storing weights and biases with shape [ 1, ..., M, 1 ].
    P_apply_weight: optional
        Partition for applying weights and biases with shape [ D, ..., M, 1].
    collect_state: bool, optional
        If true, collects the weights and biases to the root worker and
        serializes them to disk when the state_dict() function is called.
        Instead of the weights and biases, the state dictionary contains
        paths to those files. Default is false.
    num_heads: int, optional
        Total number of attention heads across all workers for multi-head attention.
        Only required if collect_state=True. Default is None.
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
    scale_backward : Union[int, slice], optional
        Scale backward pass for AllGather operation by no. of workers along the given
        dimension. Default is None.
    """

    def __init__(self, P_y, in_features, out_features, bias=True, device=None, dtype=None,
        P_x=None, P_store_weight=None, P_apply_weight=None, collect_state=False, num_heads=None,
        num_heads_kv=None, num_vars=3, geglu=False, scale_backward=None):

        super(DistributedLinearAllGather, self).__init__()

        # P_y is assumed to have shape [ *, 1, p]
        # Data is assumed to have shape [ *, n, channel_in/p ]
        self.P_y = P_y
        if not self.P_y.active:
            return
        else:
            assert P_y.shape[-2] == 1 or P_y.shape[-1] == 1

        # Input partition can be different than output partition
        # (i.e. if input is partitioned along tokens)
        if P_x is None:
            self.P_x = P_y
        else:
            assert P_x.dim == P_y.dim
            assert P_x.shape[-2] == P_y.shape[-1]
            assert P_x.shape[-1] == 1
            self.P_x = P_x

        if device is None: device = P_y.device
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.in_features = in_features
        self.out_features = out_features
        self.collect_state = collect_state
        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv
        self.num_vars = num_vars    #  3, 2, 1 (QKV, KV, Q)
        self.geglu = geglu
        self.use_bias = bias

        # Partition for storing weights & biases
        if P_store_weight is not None:
            assert P_y.dim == P_store_weight.dim
            assert P_store_weight.shape[-2] == P_y.shape[-1]
            assert P_store_weight.shape[-1] == 1
            if P_y.dim > 2:
                assert np.prod(P_store_weight.shape[:P_y.dim-2]) == 1
        else:
            store_weight_partition_shape = [1] * P_y.dim
            store_weight_partition_shape[-2] = P_y.shape[-1]

            index_store_weight = [slice(0, 1)] * P_y.dim
            index_store_weight[-1] = slice(0, P_y.shape[-1])
            store_weight_workers = worker_layout(P_y.shape)[tuple(index_store_weight)].\
                reshape(-1).tolist()

            P_store_weight_base = P_y.create_partition_inclusive(store_weight_workers)
            P_store_weight = P_store_weight_base.create_cartesian_topology_partition(
                store_weight_partition_shape)
            P_store_weight_base.deactivate()

        # Partition for applying weights & biases (same size as P_y, but with last
        # two dims swapped).
        if P_apply_weight is not None:
            assert P_y.dim == P_apply_weight.dim
            assert P_apply_weight.shape[-1] == 1
            assert P_apply_weight.shape[-2] == P_y.shape[-1]
            for i in range(P_y.dim-2):
                assert P_apply_weight.shape[i] == P_y.shape[i]
        else:
            apply_weight_partition_shape = P_y.shape.copy()
            apply_weight_partition_shape[-1] = 1
            apply_weight_partition_shape[-2] = P_y.shape[-1]

            P_apply_weight_base = P_y.create_partition_inclusive(range(P_y.size))
            P_apply_weight = P_apply_weight_base.create_cartesian_topology_partition(
                apply_weight_partition_shape)
            P_apply_weight_base.deactivate()

        # Store partitions for later  access
        self.P_store_weight = P_store_weight
        self.P_apply_weight = P_apply_weight

        # Function to broadcast weights and biases
        self.broadcast_weight = Broadcast(P_store_weight, P_apply_weight, scale_backward=scale_backward)
        if bias:
            self.broadcast_bias = Broadcast(P_store_weight, P_apply_weight, scale_backward=scale_backward)

        # Create weights
        if P_store_weight.active:

            # Local shape of weights, which must have the same no. of dimensions as P_y
            weight_shape = [1] * P_y.dim
            out_features_local = compute_subshape(P_store_weight.shape[-2],
                                                  P_store_weight.index[-2],
                                                 [out_features])[0]
            weight_shape[-1] = in_features
            weight_shape[-2] = out_features_local

            # Create weights. Every worker either has weights or receives weights.
            self.weight = torch.nn.Parameter(torch.empty(tuple(weight_shape), **factory_kwargs))
        else:
            self.register_buffer('weight', zero_volume_tensor(device=device, requires_grad=True))

        # Create bias
        if self.use_bias and P_store_weight.active:
            bias_shape = [1] * P_y.dim
            bias_shape[-2] = out_features_local
            self.bias = torch.nn.Parameter(torch.empty(tuple(bias_shape), **factory_kwargs))
        elif self.use_bias:
            self.register_buffer('bias', zero_volume_tensor(device=device, requires_grad=True))
        else:
           self.register_parameter('bias', None)

        # Initialize parameters
        self.reset_parameters()

        # All-gather operation
        gather_dim = torch.argmax(torch.tensor(self.P_x.shape[-2:])) + self.P_x.dim - 2
        self.all_gather = AllGather(self.P_x, axes_all_gather=(gather_dim,), scale_backward=scale_backward)

        # State dict hooks for gather/scattering distributed weights
        self._register_state_dict_hook(self.gather_state_dict)
        self._register_load_state_dict_pre_hook(self.scatter_state_dict)

        # Partition for collecting weights/biases for saving the state dict
        if self.collect_state:
            P_root_base = P_y.create_partition_inclusive([0])
            self.P_root = P_root_base.create_cartesian_topology_partition([1]*P_y.dim)
            self.gather_weight = Repartition(P_store_weight, self.P_root, preserve_batch=False)
            self.scatter_weight = Repartition(self.P_root, P_store_weight, preserve_batch=False)
            if self.use_bias:
                self.gather_bias = Repartition(P_store_weight, self.P_root, preserve_batch=False)
                self.scatter_bias = Repartition(self.P_root, P_store_weight, preserve_batch=False)

    def reset_parameters(self) -> None:

        if self.P_store_weight.active:
            init.kaiming_uniform_(self.P_store_weight, self.weight, a=math.sqrt(5))
            weight_global_shape = assemble_global_tensor_structure(
                self.weight, self.P_store_weight).shape

            if self.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(weight_global_shape)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(self.bias, -bound, bound)

    def _unsqueeze_weight(self, weight):
        shape = [1]*self.P_y.dim
        shape[-2] = weight.shape[-2]
        shape[-1] = weight.shape[-1]
        return weight.view(shape)

    def _squeeze_weight(self, weight):
        c_out, c_in = weight.shape[-2:]
        return weight.view(c_out, c_in)

    def _unsqueeze_bias(self, bias):
        shape = [1]*self.P_y.dim
        shape[-2] = bias.shape[0]
        return bias.view(shape)

    def _squeeze_bias(self, bias):
        c_out = bias.shape[-2]
        return bias.view(c_out)

    # If we collect the weights on the root worker, we need to rearrange the weights,
    # such that the split into QKV occurs in the slowest (dim 0) dimension. This enables
    # us to load weights for a different partitioning scheme than they were saved in.
    def qkv_weight_to_serial(self, weight):
        if self.num_heads_kv is None:
            head_size = weight.shape[-2] // self.num_vars // self.num_heads
            num_gpu = self.P_store_weight.shape[-2]
            weight = rearrange(self._squeeze_weight(weight), "(p v h) n -> (v p h) n",
                p=num_gpu, v=self.num_vars, h=self.num_heads//num_gpu*head_size)
            return self._unsqueeze_weight(weight)
        else:
            head_size = weight.shape[-2] // (self.num_heads_kv * 2 + self.num_heads)
            num_heads_local = compute_subshape(self.P_store_weight.shape[-2], self.P_store_weight.index[-2], [self.num_heads])[0]
            num_heads_kv_local = compute_subshape(self.P_store_weight.shape[-2], self.P_store_weight.index[-2], [self.num_heads_kv])[0]
            q_size_local = head_size * num_heads_local
            kv_size_local = head_size * num_heads_kv_local * 2
            num_gpu = self.P_store_weight.shape[-2]

            # Split into Q and KV components
            weight = rearrange(self._squeeze_weight(weight), "(p m) n -> p m n",
                p=num_gpu, m=q_size_local + kv_size_local)
            q_weight = weight[:, :q_size_local, :]
            kv_weight = weight[:, q_size_local:, :]

            # Rearrange
            q_weight = rearrange(q_weight, "p (v h) n -> (v p h) n", v=1, h=num_heads_local*head_size)
            kv_weight = rearrange(kv_weight, "p (v h) n -> (v p h) n", v=2, h=num_heads_kv_local*head_size)
            weight = torch.cat([q_weight, kv_weight], dim=0)

            return self._unsqueeze_weight(weight)

    # Similarly, if we want to load weights from a serial partitioning scheme and
    # use them in a parallel scheme, we need to rearrange the weights to move the
    # QKV/QK split into the 2nd slowest dimension (dim 1).
    def qkv_weight_to_parallel(self, weight):
        if self.num_heads_kv is None:
            head_size = weight.shape[-2] // self.num_vars // self.num_heads
            num_gpu = self.P_store_weight.shape[-2]
            weight = rearrange(self._squeeze_weight(weight), "(v p h) n -> (p v h) n",
                p=num_gpu, v=self.num_vars, h=self.num_heads//num_gpu*head_size)
            return self._unsqueeze_weight(weight)
        else:
            head_size = weight.shape[-2] // (self.num_heads_kv * 2 + self.num_heads)
            num_heads_local = compute_subshape(self.P_store_weight.shape[-2], self.P_store_weight.index[-2], [self.num_heads])[0]
            num_heads_kv_local = compute_subshape(self.P_store_weight.shape[-2], self.P_store_weight.index[-2], [self.num_heads_kv])[0]
            q_size = head_size * self.num_heads
            kv_size = head_size * self.num_heads_kv * 2
            num_gpu = self.P_store_weight.shape[-2]

            # Split into Q and KV components
            q_weight = self._squeeze_weight(weight)[:q_size, :]
            kv_weight = self._squeeze_weight(weight)[q_size:, :]

            # Rearrange
            q_weight = rearrange(q_weight, "(v p h) n -> p (v h) n", v=1, h=num_heads_local*head_size)
            kv_weight = rearrange(kv_weight, "(v p h) n -> p (v h) n", v=2, h=num_heads_kv_local*head_size)
            weight = torch.cat([q_weight, kv_weight], dim=1)
            weight = rearrange(weight, "p m n -> (p m) n")

            return self._unsqueeze_weight(weight)

    # If we collect the weights on the root worker and want to use a gated linear
    # unit right after the linear layer, we need to rearrange the weights, such that
    # the behavior on a single GPU is the same as on multiple GPUs.
    def geglu_weight_to_serial(self, weight):
        num_gpu = self.P_store_weight.shape[-2]
        weight_size = weight.shape[-2] // 2 // num_gpu
        weight = rearrange(self._squeeze_weight(weight), "(p v h) n -> (v p h) n",
            p=num_gpu, v=2, h=weight_size)
        return self._unsqueeze_weight(weight)

    # Rearrangment function for loading weights from a serial partitioning scheme
    # if a gated linear unit is used right after the linear layer.
    def geglu_weight_to_parallel(self, weight):
        num_gpu = self.P_store_weight.shape[-2]
        weight_size = weight.shape[-2] // 2 // num_gpu
        weight = rearrange(self._squeeze_weight(weight), "(v p h) n -> (p v h) n",
            p=num_gpu, v=2, h=weight_size)
        return self._unsqueeze_weight(weight)

    def gather_state_dict(self, module, destination, prefix, *args):

        if self.collect_state and self.P_y.active:
            if self.use_bias:

                # Collect bias and serialize (last entry added to dict).
                # All workers should pop their bias from the state dict.
                bias_key = next(reversed(destination))
                bias = self.gather_bias(destination.pop(bias_key))

                if self.P_root.active:
                    if self.num_heads is not None: bias = self.qkv_weight_to_serial(bias)
                    if self.geglu: bias = self.geglu_weight_to_serial(bias)

            # Collect weights and serialize (second last entry added to dict)
            weight_key = next(reversed(destination))
            weight = self.gather_weight(destination.pop(weight_key))

            if self.P_root.active:
                if self.num_heads is not None: weight = self.qkv_weight_to_serial(weight)
                if self.geglu: weight = self.geglu_weight_to_serial(weight)

                # Save filenames in state dict rather than the full weights. Only the root
                # should have the keys in the end.
                destination[weight_key] = self._squeeze_weight(weight)

                if self.use_bias:
                    destination[bias_key] = self._squeeze_bias(bias)

        return destination

    def scatter_state_dict(self, destination, prefix, *args):
        if self.collect_state and self.P_y.active:

            # Scatter weights
            weight_key = next(iter(destination))
            weight = destination.pop(weight_key)
            if self.P_root.active:
                weight = self._unsqueeze_weight(weight)
                if self.num_heads is not None: weight = self.qkv_weight_to_parallel(weight)
                if self.geglu: weight = self.geglu_weight_to_parallel(weight)
            else:
                weight = zero_volume_tensor(device=self.P_y.device, requires_grad=True)
            if self.P_store_weight.active:
                weight = self.scatter_weight(weight)

            # Scatter bias
            if self.use_bias:
                bias_key = next(iter(destination))
                bias = destination.pop(bias_key)
                if self.P_root.active:
                    bias = self._unsqueeze_bias(bias)
                    if self.num_heads is not None: bias = self.qkv_weight_to_parallel(bias)
                    if self.geglu: bias = self.geglu_weight_to_parallel(bias)
                elif self.P_apply_weight.active:
                    bias = zero_volume_tensor(device=self.P_y.device, requires_grad=True)
                if self.P_store_weight.active:
                    bias = self.scatter_bias(bias)
                if self.P_apply_weight.active:
                    destination[bias_key] = bias

            # Add scattered weight to state dict
            if self.P_y.active:
                destination[weight_key] = weight

        return destination

    def forward(self, input):
        r"""Forward function interface.

        Parameters
        ----------
        input :
            Input tensor to the convolution.

        """

        if not self.P_y.active:
            return input#.clone()

        # All-gather input
        input = self.all_gather(input)

        # Broadcast weights to everyone
        weight = self.broadcast_weight(self.weight).view(-1, self.in_features)

        # Broadcast bias
        if self.bias is not None:
            bias = self.broadcast_bias(self.bias).view(weight.shape[-2])
        else:
            bias = self.bias

        # Affine/linear transform
        return torch.nn.functional.linear(input, weight, bias)