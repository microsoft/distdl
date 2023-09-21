import numpy as np
import torch, math

from distdl.backends.common.tensor_comm import assemble_global_tensor_structure
from distdl.nn.module import Module
from distdl.nn.all_gather import AllGather
from distdl.nn.reduce_scatter import ReduceScatter
from distdl.utilities.slicing import compute_subshape
from distdl.utilities.torch import TensorStructure
from distdl.utilities.slicing import worker_layout
from distdl.nn.repartition import Repartition   # TODO may have to remove
from distdl.utilities.torch import zero_volume_tensor
import distdl.nn.init as init
from einops import rearrange


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
    """


    def __init__(self, P_y, in_features, out_features, bias=True, device=None, dtype=None, 
        P_x=None, P_store_bias=None, P_weight=None, collect_state=False, num_heads=None, 
        num_vars=3, geglu=False, checkpoint=False, num_cluster=1):

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

        if device is None: device = P_y.device
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.in_features = in_features
        self.out_features = out_features
        self.collect_state = collect_state
        self.num_heads = num_heads
        self.num_vars = num_vars    #  3, 2, 1 (QKV, KV, Q)
        self.geglu = geglu
        self.use_bias = bias
        self.dtype = dtype
        self.checkpoint = checkpoint
        self.num_cluster = num_cluster

        # Partition for storing weights & biases
        if P_store_bias is not None:
            assert P_y.dim == P_store_bias.dim
            assert P_store_bias.shape[-2] == P_y.shape[-1]
            assert P_store_bias.shape[-1] == 1
            if P_y.dim > 2:
                assert np.prod(P_store_bias.shape[:P_y.dim-2]) == 1
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
            for i in range(P_y.dim-2):
                assert P_weight.shape[i] == P_y.shape[i]
        else:
            apply_weight_partition_shape = P_y.shape.copy()
            apply_weight_partition_shape[-1] = 1
            apply_weight_partition_shape[-2] = P_y.shape[-1]

            P_weight_base = P_y.create_partition_inclusive(range(P_y.size))
            P_weight = P_weight_base.create_cartesian_topology_partition(
                apply_weight_partition_shape)

            # Multi-cluster all-gather partition
            if num_cluster > 1:
                inter_weight_partition_shape = list(P_y.shape.copy())
                inter_weight_partition_shape[-1] = 1
                inter_weight_partition_shape[-2] = P_y.shape[-1]
                inter_weight_partition_shape.insert(0, num_cluster)
                inter_weight_partition_shape[1] = inter_weight_partition_shape[1] // num_cluster

                P_weight_4d = P_weight_base.create_cartesian_topology_partition(
                    inter_weight_partition_shape)
            else:
                P_weight_4d = None
            P_weight_base.deactivate()

        # Store partitions for later  access
        self.P_store_bias = P_store_bias
        self.P_weight = P_weight
        self.P_weight_4d = P_weight_4d

        # Function to broadcast weights and biases
        if num_cluster > 1:
            self.allgather_weight_inter = AllGather(P_weight_4d, axes_all_gather=(0,), use_frontend=True)
            self.allgather_weight_intra = AllGather(P_weight_4d, axes_all_gather=(1,), use_frontend=False)
        else:
            self.allgather_weight = AllGather(P_weight, axes_all_gather=(0,), use_frontend=False)
        
        if bias:
            if num_cluster > 1:
                self.allgather_bias_inter = AllGather(P_weight_4d, axes_all_gather=(0,), use_frontend=True)
                self.allgather_bias_intra = AllGather(P_weight_4d, axes_all_gather=(1,), use_frontend=False)
            else:
                self.allgather_bias = AllGather(P_weight, axes_all_gather=(0,), use_frontend=False)

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

    def reset_parameters(self) -> None:

        if self.P_weight.active:
            init.kaiming_uniform_(self.P_weight, self.weight, a=math.sqrt(5))
            weight_global_shape = assemble_global_tensor_structure(
                self.weight, self.P_weight).shape

        if self.P_store_bias.active and self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(weight_global_shape)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        r"""Forward function interface.

        Parameters
        ----------
        input :
            Input tensor to the convolution.

        """

        if not self.P_y.active:
            return input

        # All-gather input
        input = self.all_gather(input)

        # Broadcast weights to everyone
        if self.P_weight_4d is None:
            weight = self.allgather_weight(self.weight)
        else:
            weight = self.weight.reshape(self.num_cluster, -1, self.weight.shape[-2], self.weight.shape[-1])
            weight = self.allgather_weight_inter(weight)
            weight = weight.view(1, -1, weight.shape[-2], weight.shape[-1])
            weight = self.allgather_weight_intra(weight)
            weight = weight.view(-1, weight.shape[-2], weight.shape[-1])
        weight = weight.transpose(-1, 0).view(-1, self.in_features)

        # Broadcast bias
        if self.bias is not None:
            if self.P_weight_4d is None:
                bias = self.allgather_bias(self.bias.transpose(0,-2)).transpose(0,-2)
            else:
                bias = self.bias.transpose(0,-2)
                bias = bias.reshape(self.num_cluster, -1, bias.shape[-2], bias.shape[-1])
                bias = self.allgather_bias_inter(bias)
                bias = bias.view(1, -1, bias.shape[-2], bias.shape[-1])
                bias = self.allgather_bias_intra(bias)
                bias = bias.view(-1, bias.shape[-2], bias.shape[-1])
            bias = bias.view(weight.shape[-2])
        else:
            bias = self.bias

        # Affine/linear transform
        return torch.nn.functional.linear(input, weight, bias)