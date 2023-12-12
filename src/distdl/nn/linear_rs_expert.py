import math

import numpy as np
import torch

import distdl.nn.init as init
from distdl.backends.common.tensor_comm import assemble_global_tensor_structure
from distdl.nn.all_gather import AllGather
from distdl.nn.broadcast import Broadcast
from distdl.nn.module import Module
from distdl.nn.reduce_scatter import ReduceScatter
from distdl.nn.repartition import Repartition
from distdl.utilities.slicing import compute_subshape
from distdl.utilities.slicing import worker_layout
from distdl.utilities.torch import zero_volume_tensor


class DistributedExpertReduceScatter(Module):
    r"""A distributed linear for mixture of experts (MoE) with row parallelism.

    This class provides the user interface to a distributed linear layer for
    Mixture of Experts. In contrast to the standard (distributed) linear layer,
    weights and biases of this layer contain an additional expert dimension.
    Supported partitionings for the input/output are along the expert dimension
    (dimension 0) and/or the feature/embedding dimension (dimension 2).

    Weights are partitoned along the expert and input feature dimension. Therefore,
    the forward pass calls a ReduceScatter on the output of the matrix multiplication.
    For this reason, the row-parallel version is preferrable when the output feature
    dimension is smaller than the intput feature dimension. For the reverse case, see
    DistributedExpertAllGather.

    This class assumes that the input/output tensors are three-dimensional with the
    following dimension ordering: [expert, capacity, feature_in/out].

    Parameters
    ----------
    P_e :
        3D Partition of input/output tensor with shape [ E, C, M ], where E is the no.
        of expert-parallel workers, M is the no. of model parallel workers and C is the
        no. of capacity-parallel workers.
    num_experts :
        Number of experts in the *global* input tensor.
    in_features :
        Number of features in the *global* input tensor.
    out_features :
        Number of features in the *global* output tensor.
    bias : bool
        Indicates if a bias term should be used. Default is true.
    P_store_bias : optional
        Partition for the storing bias of shape: [ E, 1, 1 ].
    P_apply_bias : options
        Partition for applying bias of shape: [ E, C, 1 ].
    collect_state: bool, optional
        If true, collects the weights and biases to the root worker and
        serializes them to disk when the state_dict() function is called.
        Instead of the weights and biases, the state dictionary contains
        paths to those files. Default is false.
    scale_backward : Union[int, slice], optional
        Scale backward pass for AllGather operation by no. of workers along the given
        dimension. Default is None.
    """

    def __init__(self, P_e, num_experts, in_features, out_features, bias=True, device=None, dtype=None,
                 P_store_bias=None, P_apply_bias=None, collect_state=False, scale_backward=None):

        super(DistributedExpertReduceScatter, self).__init__()

        # P_e is assumed to have shape [ experts, capacity, embedding ]
        self.P_e = P_e
        if not self.P_e.active:
            return

        if device is None:
            device = P_e.device
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.collect_state = collect_state
        self.use_bias = bias

        # Partition for storing bias
        if P_store_bias is not None:
            assert P_store_bias.dim == P_e.dim
            assert P_store_bias.shape[0] == P_e.shape[0]
            assert P_store_bias.shape[1:] == 1
        elif self.use_bias:
            store_bias_partition_shape = P_e.shape.copy()
            store_bias_partition_shape[1:] = 1

            index_store_bias = [slice(0, 1)] * P_e.dim
            index_store_bias[0] = slice(0, P_e.shape[0])
            store_bias_workers = worker_layout(P_e.shape)[tuple(index_store_bias)].reshape(-1).tolist()

            P_store_bias_base = P_e.create_partition_inclusive(store_bias_workers)
            P_store_bias = P_store_bias_base.create_cartesian_topology_partition(store_bias_partition_shape)
            P_store_bias_base.deactivate()

        # Partition for applying bias (same as P_e with last dimension set to 1)
        if P_apply_bias is not None:
            assert P_apply_bias.dim == P_e.dim
            assert P_apply_bias.shape[2] == 1
            for i in range(P_apply_bias.dim - 1):
                assert P_apply_bias.shape[i] == P_e.shape[i]
        elif self.use_bias:
            apply_bias_partition_shape = P_e.shape.copy()
            apply_bias_partition_shape[2] = 1

            index_apply_bias = [slice(0, 1)] * P_e.dim
            for i in range(P_e.dim - 1):
                index_apply_bias[i] = slice(0, P_e.shape[i])
            apply_bias_workers = worker_layout(P_e.shape)[tuple(index_apply_bias)].reshape(-1).tolist()

            P_apply_bias_base = P_e.create_partition_inclusive(apply_bias_workers)
            P_apply_bias = P_apply_bias_base.create_cartesian_topology_partition(apply_bias_partition_shape)
            P_apply_bias_base.deactivate()

        # Store partitions for later  access
        if self.use_bias:
            self.P_store_bias = P_store_bias
            self.P_apply_bias = P_apply_bias

        # Create weights
        if P_e.active:

            # Local shape of weights, which must have the same no. of dimensions as P_e
            weight_shape = [1] * P_e.dim

            num_experts_local = compute_subshape(P_e.shape[0],
                                                 P_e.index[0],
                                                 [num_experts])[0]

            out_features_local = compute_subshape(P_e.shape[1],
                                                  P_e.index[1],
                                                  [out_features])[0]

            in_features_local = compute_subshape(P_e.shape[2],
                                                 P_e.index[2],
                                                 [in_features])[0]

            weight_shape[0] = num_experts_local
            weight_shape[1] = out_features_local
            weight_shape[2] = in_features_local

            # Create weights. Every worker either has weights or receives weights.
            self.weight = torch.nn.Parameter(torch.empty(tuple(weight_shape), **factory_kwargs))
        else:
            self.register_buffer('weight', zero_volume_tensor(device=device, requires_grad=True))

        # Create bias. Only 1 worker stores the bias and a subset of workers receive it.
        if self.use_bias and self.P_store_bias.active:
            bias_shape = [1] * P_e.dim
            bias_shape[0] = num_experts_local
            bias_shape[1] = out_features
            self.bias = torch.nn.Parameter(torch.empty(tuple(bias_shape), **factory_kwargs))    # store bias
        elif self.use_bias and self.P_apply_bias.active:
            self.register_buffer('bias', zero_volume_tensor(device=device, dtype=dtype, requires_grad=True))
        else:
            self.register_parameter('bias', None)    # don't receive bias

        # Initialize parameters
        self.reset_parameters()

        # Reduce-scatter operation
        self.reduce_scatter = ReduceScatter(self.P_e, axes_reduce_scatter=(2,))
        self.all_gather_weight = AllGather(self.P_e, axes_all_gather=(1,), scale_backward=scale_backward)
        if self.use_bias:
            self.broadcast_bias = Broadcast(self.P_store_bias, self.P_apply_bias, scale_backward=scale_backward)

        # State dict hooks for gather/scattering distributed weights
        self._register_state_dict_hook(self.gather_state_dict)
        self._register_load_state_dict_pre_hook(self.scatter_state_dict)

        # Partition for collecting weights/biases for saving the state dict
        P_root_base = P_e.create_partition_inclusive([0])
        self.P_root = P_root_base.create_cartesian_topology_partition([1] * P_e.dim)
        self.gather_weight = Repartition(P_e, self.P_root, preserve_batch=False)
        self.scatter_weight = Repartition(self.P_root, P_e, preserve_batch=False)
        if self.use_bias:
            self.gather_bias = Repartition(P_store_bias, self.P_root, preserve_batch=False)
            self.scatter_bias = Repartition(self.P_root, P_store_bias, preserve_batch=False)

    def reset_parameters(self) -> None:

        if self.P_e.active:
            init.kaiming_uniform_(self.P_e, self.weight, a=math.sqrt(5))
            weight_global_shape = assemble_global_tensor_structure(self.weight, self.P_e).shape

        if self.use_bias and self.P_store_bias.active:
            fan_in, _ = init._calculate_fan_in_and_fan_out(weight_global_shape)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def gather_state_dict(self, module, destination, prefix, *args):
        if self.collect_state and self.P_e.active:
            if self.use_bias and self.bias is not None:

                # Pop bias from state dict and serialize it
                bias_key = next(reversed(destination))
                bias = self.gather_bias(destination.pop(bias_key))

            # Collect weights (second last entry added to dict)
            weight_key = next(reversed(destination))
            weight = self.gather_weight(destination.pop(weight_key))

            # Serialize weights
            if self.P_root.active:

                # Add filenames back to state dict
                destination[weight_key] = weight

                if self.use_bias:
                    destination[bias_key] = bias

        return destination

    def scatter_state_dict(self, destination, prefix, *args):
        if self.collect_state and self.P_e.active:

            # Scatter weights
            weight_key = next(iter(destination))
            weight = destination.pop(weight_key)
            if not self.P_root.active:
                weight = zero_volume_tensor(device=self.P_e.device, dtype=weight.dtype, requires_grad=True)
            weight = self.scatter_weight(weight)

            # Load bias
            if self.use_bias:
                bias_key = next(iter(destination))
                bias = destination.pop(bias_key)
                if not self.P_root.active and self.P_store_bias.active:
                    bias = zero_volume_tensor(device=self.P_e.device, dtype=bias.dtype, requires_grad=True)

                if self.P_store_bias.active:
                    bias = self.scatter_bias(bias)
                    destination[bias_key] = bias

            if self.P_e.active:
                destination[weight_key] = weight

        return destination

    def forward(self, input):
        r"""Forward function interface.

        Parameters
        ----------
        input :
            Input tensor to the convolution.

        """

        if not self.P_e.active:
            return input

        # Gather weights
        weight = self.all_gather_weight(self.weight)

        # Broadcast bias to capacity-parallel workers
        if self.use_bias:
            bias = self.broadcast_bias(self.bias)
            if self.P_apply_bias.active:
                bias = bias.view(bias.shape[0], 1, -1)

        if self.use_bias and self.P_apply_bias.active:
            y = torch.einsum('ecm,enm->ecn', input, weight) + bias
        else:
            y = torch.einsum('ecm,enm->ecn', input, weight)

        # Reduce-scatter
        y = self.reduce_scatter(y)
        return y
