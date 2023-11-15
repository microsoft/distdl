import math

import torch

import distdl.nn.init as init
from distdl.backends.common.tensor_comm import assemble_global_tensor_structure
from distdl.nn.all_gather import AllGather
from distdl.nn.module import Module
from distdl.nn.repartition import Repartition
from distdl.utilities.slicing import compute_subshape
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
    P_weight : optional
        Partition for storing weights and biases. Must be of shape [ E, M, 1 ].
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
                 P_weight=None, collect_state=False, scale_backward=None):

        super(DistributedExpertAllGather, self).__init__()

        self.P_x = P_x
        if not self.P_x.active:
            return
        else:
            assert P_x.shape[-2] == 1 or P_x.shape[-1] == 1

        if device is None:
            device = P_x.device
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.in_features = in_features
        self.out_features = out_features
        self.collect_state = collect_state
        self.use_bias = bias

        # Partition for weights & biases (same size as P_x, but with last two dims swapped).
        if P_weight is not None:
            assert P_x.dim == P_weight.dim
            assert P_weight.shape[-1] == 1
            assert P_weight.shape[-2] == P_x.shape[-1]
            for i in range(P_x.dim - 2):
                assert P_weight.shape[i] == P_x.shape[i]
        else:
            apply_weight_partition_shape = P_x.shape.copy()
            apply_weight_partition_shape[-1] = 1
            apply_weight_partition_shape[-2] = P_x.shape[-1]

            P_weight_base = P_x.create_partition_inclusive(range(P_x.size))
            P_weight = P_weight_base.create_cartesian_topology_partition(
                apply_weight_partition_shape)
            P_weight_base.deactivate()

        # Store partitions for later  access
        self.P_weight = P_weight

        # Create weights
        if P_weight.active:

            # Local shape of weights, which must have the same no. of dimensions as P_x
            weight_shape = [1] * P_x.dim
            num_experts_local = compute_subshape(P_weight.shape[0],
                                                 P_weight.index[0],
                                                 [num_experts])[0]
            out_features_local = compute_subshape(P_weight.shape[-2],
                                                  P_weight.index[-2],
                                                  [out_features])[0]
            weight_shape[0] = num_experts_local
            weight_shape[-2] = out_features_local
            weight_shape[-1] = in_features

            # Create weights. Every worker either has weights or receives weights.
            self.weight = torch.nn.Parameter(torch.empty(tuple(weight_shape), **factory_kwargs))
        else:
            self.register_buffer('weight', zero_volume_tensor(device=device, requires_grad=True))

        # Create bias
        if self.use_bias and P_weight.active:
            bias_shape = [1] * P_x.dim
            bias_shape[0] = num_experts_local
            bias_shape[-2] = out_features_local
            self.bias = torch.nn.Parameter(torch.empty(tuple(bias_shape), **factory_kwargs))
        elif self.use_bias:
            self.register_buffer('bias', zero_volume_tensor(device=device, requires_grad=True))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters
        self.reset_parameters()

        # All-gather operation
        self.all_gather = AllGather(self.P_x, axes_all_gather=(2,), scale_backward=scale_backward)

        # State dict hooks for gather/scattering distributed weights
        self._register_state_dict_hook(self.gather_state_dict)
        self._register_load_state_dict_pre_hook(self.scatter_state_dict)

        # Partition for collecting weights/biases for saving the state dict
        if self.collect_state:
            P_root_base = P_x.create_partition_inclusive([0])
            self.P_root = P_root_base.create_cartesian_topology_partition([1] * P_x.dim)
            self.gather_weight = Repartition(P_weight, self.P_root, preserve_batch=False)
            self.scatter_weight = Repartition(self.P_root, P_weight, preserve_batch=False)
            if self.use_bias:
                self.gather_bias = Repartition(P_weight, self.P_root, preserve_batch=False)
                self.scatter_bias = Repartition(self.P_root, P_weight, preserve_batch=False)

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

        if self.collect_state and self.P_x.active:
            if self.use_bias:

                # Collect bias and serialize (last entry added to dict).
                # All workers should pop their bias from the state dict.
                bias_key = next(reversed(destination))
                bias = self.gather_bias(destination.pop(bias_key))

                if self.P_root.active:
                    torch.save(bias, bias_key)

            # Collect weights and serialize (second last entry added to dict)
            weight_key = next(reversed(destination))
            weight = self.gather_weight(destination.pop(weight_key))

            if self.P_root.active:
                torch.save(weight, weight_key)

                # Save filenames in state dict rather than the full weights. Only the root
                # should have the keys in the end.
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
            else:
                weight = zero_volume_tensor(device=self.P_x.device, requires_grad=True)
            if self.P_weight.active:
                weight = self.scatter_weight(weight)

            # Scatter bias
            if self.use_bias:
                bias_key = next(iter(destination))
                destination.pop(bias_key)
                if self.P_root.active:
                    bias = torch.load(bias_key)
                elif self.P_weight.active:
                    bias = zero_volume_tensor(device=self.P_x.device, requires_grad=True)
                if self.P_weight.active:
                    bias = self.scatter_bias(bias)
                    destination[bias_key] = bias

            # Add scattered weight to state dict
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

        # All-gather input
        input = self.all_gather(input)

        # Affine/linear transform
        if self.bias is not None:
            bias = self.bias.view(self.bias.shape[0], 1, self.bias.shape[-2])
            y = torch.einsum('ecm,enm->ecn', input, self.weight) + bias
        else:
            y = torch.einsum('ecm,enm->ecn', input, self.weight)

        return y
