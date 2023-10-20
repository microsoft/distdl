import numpy as np
import torch
import math

from distdl.backends.common.tensor_comm import assemble_global_tensor_structure
from distdl.nn.module import Module
from distdl.nn.reduce_scatter import ReduceScatter
from distdl.utilities.slicing import compute_subshape
from distdl.utilities.slicing import worker_layout
from distdl.nn.repartition import Repartition
from distdl.utilities.torch import zero_volume_tensor
import distdl.nn.init as init


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
    P_x :
        3D Partition of input/output tensor with shape [ E, 1, M ], where E is the no.
        of expert-parallel workers and M is the no. of model parallel workers.
    num_experts :
        Number of experts in the *global* input tensor.
    in_features :
        Number of features in the *global* input tensor.
    out_features :
        Number of features in the *global* output tensor.
    bias : bool
        Indicates if a bias term should be used. Default is true.
    P_bias : optional
        Partition for the bias of shape: [ E, 1, 1 ].
    collect_state: bool, optional
        If true, collects the weights and biases to the root worker and
        serializes them to disk when the state_dict() function is called.
        Instead of the weights and biases, the state dictionary contains
        paths to those files. Default is false.
    scale_backward : Union[int, slice], optional
        Scale backward pass for AllGather operation by no. of workers along the given
        dimension. Default is None.
    """

    def __init__(self, P_x, num_experts, in_features, out_features, bias=True, device=None, dtype=None,
                 P_bias=None, collect_state=False, scale_backward=None):

        super(DistributedExpertReduceScatter, self).__init__()

        # P_x is assumed to have shape [ experts, capacity, embedding ]
        self.P_x = P_x
        if not self.P_x.active:
            return
        else:
            assert P_x.shape[-2] == 1 or P_x.shape[-1] == 1

        if device is None:
            device = P_x.device
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.collect_state = collect_state
        self.use_bias = bias

        # Partition for applying bias
        if P_bias is not None:
            assert P_x.dim == P_bias.dim
            assert np.prod(P_bias.shape[-2:]) == 1
            for i in range(P_x.dim-2):
                assert P_bias.shape[i] == P_x.shape[i]
        elif bias:
            apply_bias_partition_shape = P_x.shape.copy()
            apply_bias_partition_shape[-2:] = 1

            index_bias = [slice(0, 1)] * P_x.dim
            for i in range(P_x.dim-2):
                index_bias[i] = slice(0, P_x.shape[i])
            apply_bias_workers = worker_layout(P_x.shape)[tuple(index_bias)].reshape(-1).tolist()

            P_bias_base = P_x.create_partition_inclusive(apply_bias_workers)
            P_bias = P_bias_base.create_cartesian_topology_partition(apply_bias_partition_shape)
            P_bias_base.deactivate()

        # Store partitions for later  access
        if bias:
            self.P_bias = P_bias

        # Create weights
        if P_x.active:

            # Local shape of weights, which must have the same no. of dimensions as P_x
            weight_shape = [1] * P_x.dim

            num_experts_local = compute_subshape(P_x.shape[0],
                                                 P_x.index[0],
                                                 [num_experts])[0]

            in_features_local = compute_subshape(P_x.shape[-1],
                                                 P_x.index[-1],
                                                 [in_features])[0]

            weight_shape[0] = num_experts_local
            weight_shape[-2] = out_features
            weight_shape[-1] = in_features_local

            # Create weights. Every worker either has weights or receives weights.
            self.weight = torch.nn.Parameter(torch.empty(tuple(weight_shape), **factory_kwargs))
        else:
            self.register_buffer('weight', zero_volume_tensor(device=device, requires_grad=True))

        # Create bias. Only 1 worker stores the bias and a subset of workers receive it.
        if self.use_bias and self.P_bias.active:
            bias_shape = [1] * P_x.dim
            bias_shape[0] = num_experts_local
            bias_shape[-2] = out_features
            self.bias = torch.nn.Parameter(torch.empty(tuple(bias_shape), **factory_kwargs))    # store bias
        else:
            self.register_parameter('bias', None)    # don't receive bias

        # Initialize parameters
        self.reset_parameters()

        # Reduce-scatter operation
        self.reduce_scatter = ReduceScatter(self.P_x, axes_reduce_scatter=(2,), scale_backward=scale_backward)

        # State dict hooks for gather/scattering distributed weights
        self._register_state_dict_hook(self.gather_state_dict)
        self._register_load_state_dict_pre_hook(self.scatter_state_dict)

        # Partition for collecting weights/biases for saving the state dict
        if self.collect_state:
            P_root_base = P_x.create_partition_inclusive([0])
            self.P_root = P_root_base.create_cartesian_topology_partition([1]*P_x.dim)
            self.gather_weight = Repartition(P_x, self.P_root, preserve_batch=False)
            self.scatter_weight = Repartition(self.P_root, P_x, preserve_batch=False)
            self.gather_bias = Repartition(P_bias, self.P_root, preserve_batch=False)
            self.scatter_bias = Repartition(self.P_root, P_bias, preserve_batch=False)

    def reset_parameters(self) -> None:

        if self.P_x.active:
            init.kaiming_uniform_(self.P_x, self.weight, a=math.sqrt(5))
            weight_global_shape = assemble_global_tensor_structure(self.weight, self.P_x).shape

        if self.use_bias and self.P_bias.active:
            fan_in, _ = init._calculate_fan_in_and_fan_out(weight_global_shape)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def gather_state_dict(self, module, destination, prefix, *args):
        if self.collect_state and self.P_x.active:
            if self.use_bias and self.bias is not None:

                # Pop bias from state dict and serialize it
                bias_key = next(reversed(destination))
                bias = self.gather_bias(destination.pop(bias_key))

                if self.P_root.active:
                    torch.save(bias, bias_key)

            # Collect weights (second last entry added to dict)
            weight_key = next(reversed(destination))
            weight = self.gather_weight(destination.pop(weight_key))

            # Serialize weights
            if self.P_root.active:
                torch.save(weight, weight_key)

                # Add filenames back to state dict
                destination[weight_key] = weight_key

                if self.use_bias:
                    destination[bias_key] = bias_key

        return destination

    def scatter_state_dict(self, destination, prefix, *args):
        if self.collect_state and self.P_x.active:

            # Scatter weights
            weight_key = next(iter(destination))
            destination.pop(weight_key)
            if self.P_root.active:
                weight = torch.load(weight_key)
            elif self.P_x.active:
                weight = zero_volume_tensor(device=self.P_x.device, requires_grad=True)
            if self.P_x.active:
                weight = self.scatter_weight(weight)

            # Load bias
            if self.use_bias:
                bias_key = next(iter(destination))
                destination.pop(bias_key)

                if self.P_root.active:
                    bias = torch.load(bias_key)
                elif self.P_bias.active:
                    bias = zero_volume_tensor(device=self.P_bias.device, requires_grad=True)

                if self.P_bias.active:
                    bias = self.scatter_bias(bias)
                    destination[bias_key] = bias

            if self.P_x.active:
                destination[weight_key] = weight

        return destination

    def forward(self, input):
        r"""Forward function interface.

        Parameters
        ----------
        input :
            Input tensor to the convolution.

        """

        if not self.P_x.active:
            return input

        # Affine/linear transform
        if self.bias is not None:
            bias = self.bias.view(self.bias.shape[0], 1, self.bias.shape[-2])
            y = torch.einsum('ecm,enm->ecn', input, self.weight) + bias
        else:
            y = torch.einsum('ecm,enm->ecn', input, self.weight)

        # Reduce-scatter
        y = self.reduce_scatter(y)
        return y
