import math

import einops
import torch

import distdl.nn.init as init
from distdl.backends.common.tensor_comm import assemble_global_tensor_structure
from distdl.nn.all_gather import AllGather
from distdl.nn.broadcast import Broadcast
from distdl.nn.module import Module
from distdl.nn.repartition import Repartition
from distdl.utilities.slicing import compute_subshape
from distdl.utilities.slicing import worker_layout
from distdl.utilities.torch import zero_volume_tensor


class DistributedExpertAllGather(Module):
    r"""A distributed linear for mixture of experts (MoE) with column parallelism.

    This class provides the user interface to a distributed linear layer for
    Mixture of Experts. In contrast to the standard (distributed) linear layer,
    weights and biases of this layer contain an additional expert dimension.
    Supported partitionings for the input/output are along the expert dimension
    (dimension 0) and/or the feature/embedding dimension (dimension 2).

    Weights are partitoned along the expert and output feature dimension. Therefore,
    the forward pass calls an AllGather on the input prior to the matrix multiplication.
    For this reason, the column-parallel version is preferrable when the input feature
    dimension is smaller than the output feature dimension. For the reverse case, see
    DistributedExpertReduceScatter.

    This class assumes that the input/output tensors are three-dimensional with the
    following dimension ordering: [expert, capacity, feature_in/out].

    Parameters
    ----------
    P_expert_emb :
        4D Partition of input/output tensor with shape [ E, N, 1, M ], where E is the no.
        of expert-parallel workers, N is the no. of capacity-parallel workers and M is the
        no. of model parallel workers.
    num_experts :
        Number of experts in the *global* input tensor.
    in_features :
        Number of features in the *global* input tensor.
    out_features :
        Number of features in the *global* output tensor.
    bias : bool
        Indicates if a bias term should be used. Default is true.
    P_weight: optionsl
        Partition for the weights of shape: [ E, N, M ].
    P_bias : optional
        Partition for the bias of shape: [ E, 1, M ].
    collect_state: bool, optional
        If true, collects the weights and biases to the root worker and
        serializes them to disk when the state_dict() function is called.
        Instead of the weights and biases, the state dictionary contains
        paths to those files. Default is false.
    scale_backward : Union[int, slice], optional
        Scale backward pass for AllGather operation by no. of workers along the given
        dimension. Default is None.
    P_expert_seq : optional
        4D Partition of the input tensor if it is partitioned with sequence parallelism.
        This partition has size [ E, N, S, 1 ], where S is the no. of sequence-parallel
        workers. All other dimensions are the same as P_expert_emb. Default is None.
    """

    def __init__(self, P_expert_emb, num_experts, in_features, out_features, bias=True, device=None, dtype=None,
                 P_weight=None, P_bias=None, collect_state=False, scale_backward=None, P_expert_seq=None):

        super(DistributedExpertAllGather, self).__init__()

        self.P_expert_emb = P_expert_emb
        self.P_expert_seq = P_expert_seq
        if not self.P_expert_emb.active:
            return

        if device is None:
            device = P_expert_emb.device
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.in_features = in_features
        self.out_features = out_features
        self.collect_state = collect_state
        self.use_bias = bias

        # Partition for weights (3D partition of size [ E, N, M ]).
        if P_weight is not None:
            assert P_weight.dim == P_expert_emb.dim - 1
            assert P_weight.shape[0] == P_expert_emb.shape[0]
            assert P_weight.shape[1] == P_expert_emb.shape[1]
            assert P_weight.shape[2] == P_expert_emb.shape[3]
        else:
            weight_partition_shape = [1] * (P_expert_emb.dim - 1)
            weight_partition_shape[0] = P_expert_emb.shape[0]
            weight_partition_shape[1] = P_expert_emb.shape[1]
            weight_partition_shape[2] = P_expert_emb.shape[3]

            P_weight_base = P_expert_emb.create_partition_inclusive(range(P_expert_emb.size))
            P_weight = P_weight_base.create_cartesian_topology_partition(weight_partition_shape)
            P_weight_base.deactivate()

        # Partition for biases (same size as P_weight, but with 2nd dim equal to 1).
        if P_bias is not None:
            assert P_bias.dim == P_weight.dim
            assert P_bias.shape[1] == 1
            assert P_bias.shape[0] == P_weight.shape[0]
            assert P_bias.shape[2] == P_weight.shape[2]
        elif self.use_bias:
            store_bias_partition_shape = P_weight.shape.copy()
            store_bias_partition_shape[1] = 1

            index_store_bias = [slice(0, 1)] * P_weight.dim
            index_store_bias[0] = slice(0, P_weight.shape[0])
            index_store_bias[2] = slice(0, P_weight.shape[2])
            store_bias_workers = worker_layout(P_weight.shape)[tuple(index_store_bias)].\
                reshape(-1).tolist()

            P_bias_base = P_weight.create_partition_inclusive(store_bias_workers)
            P_bias = P_bias_base.create_cartesian_topology_partition(
                store_bias_partition_shape)
            P_bias_base.deactivate()

        # Store parameter partitions for later  access
        self.P_weight = P_weight
        if self.use_bias:
            self.P_bias = P_bias

        # Create weights
        if P_weight.active:

            # Local shape of weights, which must have the same no. of dimensions as P_weight
            weight_shape = [1] * P_weight.dim

            # Partition weights along expert dimension
            num_experts_local = compute_subshape(P_weight.shape[0],
                                                 P_weight.index[0],
                                                 [num_experts])[0]

            # Shard weights along data-parallel dimension (ZeRO-3 style)
            in_features_local = compute_subshape(P_weight.shape[1],
                                                 P_weight.index[1],
                                                 [in_features])[0]

            # Also partition weights along the model-parallel (embedding) dimension
            out_features_local = compute_subshape(P_weight.shape[2],
                                                  P_weight.index[2],
                                                  [out_features])[0]
            weight_shape[0] = num_experts_local
            weight_shape[1] = in_features_local
            weight_shape[2] = out_features_local

            # Create weights. Every worker either has weights or receives weights.
            self.weight = torch.nn.Parameter(torch.empty(tuple(weight_shape), **factory_kwargs))
        else:
            self.register_buffer('weight', zero_volume_tensor(device=device, dtype=dtype, requires_grad=True))

        # Create bias
        if self.use_bias and P_bias.active:
            bias_shape = [1] * P_weight.dim
            bias_shape[0] = num_experts_local
            bias_shape[2] = out_features_local
            self.bias = torch.nn.Parameter(torch.empty(tuple(bias_shape), **factory_kwargs))
        elif self.use_bias and P_weight.active:
            self.register_buffer('bias', zero_volume_tensor(device=device, dtype=dtype, requires_grad=True))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters
        self.reset_parameters()

        # All-gather operation. For sequence parallelism, gather hidden states along the sequence dimension.
        # Otherwise, gather along the embedding dimension. Weights are all-gathered along the capacity dimension.
        if P_expert_seq is not None:
            self.all_gather_hidden = AllGather(self.P_expert_seq, axes_all_gather=(2,))
        else:
            self.all_gather_hidden = AllGather(self.P_expert_emb, axes_all_gather=(3,))
        self.all_gather_weight = AllGather(self.P_weight, axes_all_gather=(1,), scale_backward=scale_backward)
        if self.use_bias:
            self.broadcast_bias = Broadcast(self.P_bias, self.P_weight, scale_backward=scale_backward)

        # State dict hooks for gather/scattering distributed weights
        self._register_state_dict_hook(self.gather_state_dict)
        self._register_load_state_dict_pre_hook(self.scatter_state_dict)

        # Partition for collecting weights/biases for saving the state dict
        P_root_base = P_weight.create_partition_inclusive([0])
        self.P_root = P_root_base.create_cartesian_topology_partition([1] * P_weight.dim)
        self.gather_weight = Repartition(P_weight, self.P_root, preserve_batch=False)
        self.scatter_weight = Repartition(self.P_root, P_weight, preserve_batch=False)
        if self.use_bias:
            self.gather_bias = Repartition(P_bias, self.P_root, preserve_batch=False)
            self.scatter_bias = Repartition(self.P_root, P_bias, preserve_batch=False)

    def reset_parameters(self) -> None:

        if self.P_weight.active:
            init.kaiming_uniform_(self.P_weight, self.weight, a=math.sqrt(5))
            weight_global_shape = assemble_global_tensor_structure(
                self.weight, self.P_weight).shape

            if self.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(weight_global_shape)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(self.bias, -bound, bound)

    def gather_state_dict(self, module, destination, prefix, *args):

        if self.collect_state:
            if self.use_bias and self.P_weight.active:

                # Collect bias and serialize (last entry added to dict).
                # All workers should pop their bias from the state dict.
                bias_key = next(reversed(destination))
                bias = self.gather_bias(destination.pop(bias_key))

            if self.P_weight.active:
                # Collect weights and serialize (second last entry added to dict)
                weight_key = next(reversed(destination))
                weight = self.gather_weight(destination.pop(weight_key))

            if self.P_root.active:

                # Save filenames in state dict rather than the full weights. Only the root
                # should have the keys in the end.
                destination[weight_key] = weight

                if self.use_bias:
                    destination[bias_key] = bias

        return destination

    def scatter_state_dict(self, destination, prefix, *args):
        if self.collect_state:
            if self.P_weight.active:

                # Scatter weights
                weight_key = next(iter(destination))
                weight = destination.pop(weight_key)
                if not self.P_root.active:
                    weight = zero_volume_tensor(device=self.P_weight.device, dtype=weight.dtype, requires_grad=True)
                weight = self.scatter_weight(weight)

            # Scatter bias
            if self.use_bias and self.P_weight.active:
                bias_key = next(iter(destination))
                bias = destination.pop(bias_key)
                if not self.P_root.active:
                    bias = zero_volume_tensor(device=self.P_weight.device, dtype=bias.dtype, requires_grad=True)
                bias = self.scatter_bias(bias)
                destination[bias_key] = bias

            # Add scattered weight to state dict
            if self.P_weight.active:
                destination[weight_key] = weight

        return destination

    def forward(self, input):
        r"""Forward function interface.

        Parameters
        ----------
        input :
            Input tensor to the convolution.

        """

        if not self.P_expert_emb.active:
            return input

        # All-gather input along model-parallel dimension
        input = self.all_gather_hidden(input)

        # Merge capacity and sequence dimensions
        local_capacity = input.shape[1]
        input = einops.rearrange(input, 'e c s m -> e (c s) m')

        # All-gather weights along capacity dimension
        weight = self.all_gather_weight(self.weight).transpose(1, 2)

        # Affine/linear transform
        if self.bias is not None:

            # Broadcast bias
            bias = self.broadcast_bias(self.bias)
            y = torch.einsum('ecm,enm->ecn', input, weight) + bias
        else:
            y = torch.einsum('ecm,enm->ecn', input, weight)

        # Split capacity and sequence dimensions again
        y = einops.rearrange(y, 'e (c s) n -> e c s n', c=local_capacity)

        return y
