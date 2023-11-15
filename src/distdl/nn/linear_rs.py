import math

import numpy as np
import torch

import distdl.nn.init as init
from distdl.backends.common.tensor_comm import assemble_global_tensor_structure
from distdl.nn.broadcast import Broadcast
from distdl.nn.module import Module
from distdl.nn.reduce_scatter import ReduceScatter
from distdl.nn.repartition import Repartition
from distdl.utilities.slicing import compute_subshape
from distdl.utilities.slicing import worker_layout
from distdl.utilities.torch import zero_volume_tensor


class DistributedLinearReduceScatter(Module):
    r"""A distributed linear or affine layer with weight row parallelism.

    This class provides the user interface to a distributed linear layer
    with 2D partitioning of input/output data and 1D partitioning of weights
    and biases. Inputs can be partitioned along the batch dimension (dimension 0)
    and/or the last dimension, as specified by the input partition P_x.

    Outputs can be partitioned along the batch dimension (dimension 0) plus either
    the last dimension or second last dimension. If inputs are partitioned along
    the second last dimension, an additional output partition P_y must be specified.
    If P_y is not supplied, the output partitoning is assumed to be the same as the
    intput partitioning.

    Weights and biases are partitoned along the input feature dimension. Therefore,
    a reduce-scatter is performed on the output after the matrix multiplication. For
    this reason, this layer is preferrable when the output feature dimension is
    smaller than the intput feature dimension. For the reverse case, see
    DistributedLinearAllGather. Weights and biases are stored on the 1st data-
    parallel worker only.

    Parameters
    ----------
    P_x :
        Partition of input/output tensor with shape of form: [ D, ..., 1, M ],
        where D is the number of data-parallel workers, and M is the number of
        model-parallel workers.
    in_features :
        Number of features in the *global* input tensor.
    out_features :
        Number of features in the *global* output tensor.
    bias : bool
        Indicates if a bias term should be used.
    P_y : optional
        Partition of the output tensor if output is partitioned along the second last
        dimension. Shape must be of form: [ D, ..., M, 1 ].
    P_weight : optional
        Partition for weights of shape: [ 1, ..., 1, M ].
    P_store_bias: optional
        Partition for storing the bias of shape: [ 1, ..., 1, 1 ].
    P_apply_bias: optional
        Partition for applying the bias of shape: [ D, ..., 1, 1 ].
    collect_state: bool, optional
        If true, collects the weights and biases to the root worker and
        serializes them to disk when the state_dict() function is called.
        Instead of the weights and biases, the state dictionary contains
        paths to those files. Default is false.
    scale_backward : Union[int, slice], optional
        Scale backward pass for AllGather operation by no. of workers along the given
        dimension. Default is None.
    """

    def __init__(self, P_x, in_features, out_features, bias=True, device=None, dtype=None,
                 P_y=None, P_weight=None, P_store_bias=None, P_apply_bias=None,
                 collect_state=False, scale_backward=None):

        super(DistributedLinearReduceScatter, self).__init__()

        # P_x is assumed to have shape [ *, 1, p]
        # Data is assumed to have shape [ *, n, channel_in/p ]
        self.P_x = P_x
        if not self.P_x.active:
            return
        else:
            assert P_x.shape[-2] == 1 or P_x.shape[-1] == 1

        # Input partition can be different than output partition
        # (i.e. if input is partitioned along tokens)
        if P_y is None:
            self.P_y = P_x
        else:
            assert P_y.dim == P_x.dim
            assert P_y.shape[-2] == P_x.shape[-1]
            assert P_y.shape[-1] == 1
            self.P_y = P_y

        if device is None:
            device = P_x.device
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.in_features = in_features
        self.out_features = out_features
        self.collect_state = collect_state
        self.use_bias = bias

        # Partition for storing weights (the partition for applying weights is P_x)
        if P_weight is not None:
            assert P_x.dim == P_weight.dim
            assert P_weight.shape[-1] == P_x.shape[-1]
            assert np.prod(P_weight.shape[:P_x.dim - 1]) == 1
        else:
            weight_partition_shape = [1] * P_x.dim
            weight_partition_shape[-1] = P_x.shape[-1]

            index_weight = [slice(0, 1)] * P_x.dim
            index_weight[-1] = slice(0, P_x.shape[-1])
            weight_workers = worker_layout(P_x.shape)[tuple(index_weight)].reshape(-1).tolist()

            P_weight_base = P_x.create_partition_inclusive(weight_workers)
            P_weight = P_weight_base.create_cartesian_topology_partition(weight_partition_shape)
            P_weight_base.deactivate()

        # Partition for storing bias
        if P_store_bias is not None:
            assert P_x.dim == P_store_bias.dim
            assert np.prod(P_store_bias.shape) == 1
        elif bias:
            P_store_bias_base = P_x.create_partition_inclusive([0])
            P_store_bias = P_store_bias_base.create_cartesian_topology_partition([1] * P_x.dim)
            P_store_bias_base.deactivate()

        # Partition for applying bias
        if P_apply_bias is not None:
            assert P_x.dim == P_apply_bias.dim
            assert np.prod(P_apply_bias.shape[-2:]) == 1
            for i in range(P_x.dim - 2):
                assert P_apply_bias.shape[i] == P_x.shape[i]
        elif bias:
            apply_bias_partition_shape = P_x.shape.copy()
            apply_bias_partition_shape[-2:] = 1

            index_bias = [slice(0, 1)] * P_x.dim
            for i in range(P_x.dim - 2):
                index_bias[i] = slice(0, P_x.shape[i])
            apply_bias_workers = worker_layout(P_x.shape)[tuple(index_bias)].reshape(-1).tolist()

            P_apply_bias_base = P_x.create_partition_inclusive(apply_bias_workers)
            P_apply_bias = P_apply_bias_base.create_cartesian_topology_partition(apply_bias_partition_shape)
            P_apply_bias_base.deactivate()

        # Store partitions for later  access
        self.P_weight = P_weight
        if bias:
            self.P_store_bias = P_store_bias
            self.P_apply_bias = P_apply_bias

        # Function to broadcast weights and biases
        self.broadcast_weight = Broadcast(P_weight, P_x, scale_backward=scale_backward)
        if bias and self.P_apply_bias.active:
            self.broadcast_bias = Broadcast(P_store_bias, P_apply_bias, scale_backward=scale_backward)

        # Create weights
        if P_weight.active:

            # Local shape of weights, which must have the same no. of dimensions as P_x
            weight_shape = [1] * P_x.dim
            in_features_local = compute_subshape(P_weight.shape[-1],
                                                 P_weight.index[-1],
                                                 [in_features])[0]
            weight_shape[-1] = in_features_local
            weight_shape[-2] = out_features

            # Create weights. Every worker either has weights or receives weights.
            self.weight = torch.nn.Parameter(torch.empty(tuple(weight_shape), **factory_kwargs))
        else:
            self.register_buffer('weight', zero_volume_tensor(device=device, requires_grad=True))

        # Create bias. Only 1 worker stores the bias and a subset of workers receive it.
        if self.use_bias and self.P_store_bias.active:
            bias_shape = [1] * P_x.dim
            bias_shape[-2] = out_features
            self.bias = torch.nn.Parameter(torch.empty(tuple(bias_shape), **factory_kwargs))    # store bias

        elif self.use_bias and self.P_apply_bias.active:
            self.register_buffer('bias', zero_volume_tensor(device=device, requires_grad=True))  # receive bias
        else:
            self.register_parameter('bias', None)

        # Initialize parameters
        self.reset_parameters()

        # Reduce-scatter operation
        scatter_dim = torch.argmax(torch.tensor(self.P_y.shape[-2:])) + self.P_y.dim - 2
        self.reduce_scatter = ReduceScatter(self.P_y, axes_reduce_scatter=(scatter_dim,), scale_backward=scale_backward)

        # State dict hooks for gather/scattering distributed weights
        self._register_state_dict_hook(self.gather_state_dict)
        self._register_load_state_dict_pre_hook(self.scatter_state_dict)

        # Partition for collecting weights/biases for saving the state dict
        if self.collect_state:
            P_root_base = P_x.create_partition_inclusive([0])
            self.P_root = P_root_base.create_cartesian_topology_partition([1] * P_x.dim)
            self.gather_weight = Repartition(P_weight, self.P_root, preserve_batch=False)
            self.scatter_weight = Repartition(self.P_root, P_weight, preserve_batch=False)

    def reset_parameters(self) -> None:

        if self.P_weight.active:
            init.kaiming_uniform_(self.P_weight, self.weight, a=math.sqrt(5))
            weight_global_shape = assemble_global_tensor_structure(self.weight, self.P_weight).shape

        if self.use_bias and self.P_store_bias.active:
            fan_in, _ = init._calculate_fan_in_and_fan_out(weight_global_shape)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def _unsqueeze_weight(self, weight):
        shape = [1] * self.P_y.dim
        shape[-2] = weight.shape[-2]
        shape[-1] = weight.shape[-1]
        return weight.view(shape)

    def _squeeze_weight(self, weight):
        c_out, c_in = weight.shape[-2:]
        return weight.view(c_out, c_in)

    def _unsqueeze_bias(self, bias):
        shape = [1] * self.P_y.dim
        shape[-2] = bias.shape[0]
        return bias.view(shape)

    def _squeeze_bias(self, bias):
        c_out = bias.shape[-2]
        return bias.view(c_out)

    def gather_state_dict(self, module, destination, prefix, *args):

        if self.collect_state and self.P_x.active:
            if self.use_bias and self.bias is not None:

                # Pop bias from state dict and serialize it
                bias_key = next(reversed(destination))
                bias = destination.pop(bias_key)

            # Collect weights (second last entry added to dict)
            weight_key = next(reversed(destination))
            weight = self.gather_weight(destination.pop(weight_key))

            # Serialize weights
            if self.P_root.active:

                # Add filenames back to state dict
                destination[weight_key] = self._squeeze_weight(weight)

                if self.use_bias:
                    destination[bias_key] = self._squeeze_bias(bias)

        return destination

    def scatter_state_dict(self, destination, prefix, *args):
        if self.collect_state and self.P_x.active:

            # Scatter weights
            weight_key = next(iter(destination))
            weight = destination.pop(weight_key)
            if self.P_root.active:
                weight = self._unsqueeze_weight(weight)
            else:
                weight = zero_volume_tensor(device=self.P_x.device, requires_grad=True)
            if self.P_weight.active:
                weight = self.scatter_weight(weight)

            # Load bias
            if self.use_bias:
                bias_key = next(iter(destination))
                bias = destination.pop(bias_key)

                if self.P_store_bias.active:
                    bias = self._unsqueeze_bias(bias)
                    destination[bias_key] = bias

                elif self.P_apply_bias.active:
                    bias = zero_volume_tensor(device=self.P_x.device, requires_grad=True)
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

        # Broadcast weights to everyone
        weight = self.broadcast_weight(self.weight).view(self.out_features, -1)

        # Broadcast bias
        if self.bias is not None:
            bias = self.broadcast_bias(self.bias).view(self.out_features)
        else:
            bias = self.bias

        # Affine/linear transform
        y = torch.nn.functional.linear(input, weight, bias)

        # Reduce-scatter
        return self.reduce_scatter(y)
