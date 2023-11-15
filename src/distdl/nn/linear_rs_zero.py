import math

import numpy as np
import torch

import distdl.nn.init as init
from distdl.backends.common.tensor_comm import assemble_global_tensor_structure
from distdl.nn.all_gather import AllGather
from distdl.nn.module import Module
from distdl.nn.reduce_scatter import ReduceScatter
from distdl.nn.repartition import Repartition
from distdl.utilities.slicing import compute_subshape
from distdl.utilities.slicing import worker_layout
from distdl.utilities.torch import zero_volume_tensor


# Custom forward/backward functions
class LinearReduceScatterZeROFunc(torch.autograd.Function):

    @staticmethod
    def forward(input, weight, bias, ag_input, rs_input, ag_weight, rs_weight,
                ag_bias, rs_bias, bias_active, scale_backward):

        # Gather weights    [ c_cout, 1, c_in]
        weight = ag_weight(weight).squeeze(1)   # -> [c_out, c_in]

        # Broadcast bias [c_out, 1, 1]
        if bias is not None and bias_active:
            bias = ag_bias(bias).view(1, 1, -1)  # -> [1, 1, c_out]

        # Affine layer
        output = torch.einsum('bij,kj->bik', input, weight)
        if bias is not None and bias_active:
            output += bias
        return rs_input(output)

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, weight, bias, ag_input, rs_input, ag_weight, rs_weight, \
            ag_bias, rs_bias, bias_active, scale_backward = inputs
        ctx.save_for_backward(input, weight, bias)
        ctx.constant = (ag_input, rs_input, ag_weight, rs_weight, ag_bias, rs_bias, bias_active, scale_backward)

    @staticmethod
    def backward(ctx, grad_output):

        # Load saved tensors and operators
        input, weight, bias = ctx.saved_tensors
        ag_input, rs_input, ag_weight, rs_weight, ag_bias, rs_bias, bias_active, scale_backward = ctx.constant

        # Gather input
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            if scale_backward is not None:
                grad_output.div_(np.prod(ag_input.P_allgather.shape[scale_backward]))
            grad_output = ag_input(grad_output.contiguous())

        # Input gradient
        if ctx.needs_input_grad[0]:
            weight = ag_weight(weight).squeeze(1)
            grad_input = torch.einsum('bij,jk->bik', grad_output, weight)
        else:
            grad_input = None

        # Weight gradient
        if ctx.needs_input_grad[1]:
            grad_weight = torch.einsum('bij,bik->kj', input, grad_output).unsqueeze(1)
            if scale_backward is not None:
                grad_weight.div_(np.prod(rs_weight.P_reducescatter.shape[scale_backward]))
            grad_weight = rs_weight(grad_weight)
        else:
            grad_weight = None

        # Bias gradient
        if bias_active and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0, 1)).view(-1, 1, 1)
            if scale_backward is not None:
                grad_bias.div_(np.prod(rs_bias.P_reducescatter.shape[scale_backward]))
            grad_bias = rs_bias(grad_bias)
        else:
            grad_bias = None

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None


class DistributedLinearReduceScatterZero(Module):
    r"""A distributed linear or affine layer with 2D parallelism for weights
    and input/outputs (also called ZeRO-3 or FSDP).

    This class provides the user interface to a distributed linear layer
    with 2D partitioning of input/output data and 2D partitioning of weights
    and biases. Inputs can be partitioned along the batch dimension (dimension 0)
    and/or the last dimension, as specified by the input partition P_x.

    Outputs can be partitioned along the batch dimension (dimension 0) plus either
    the last dimension or second last dimension. If inputs are partitioned along
    the second last dimension, an additional output partition P_y must be specified.
    If P_y is not supplied, the output partitoning is assumed to be the same as the
    intput partitioning.

    Weights and biases are partitoned along both the input and output feature dimension.
    Input features are partitioned along the first dimension (usually the data-parallel
    dimension), and output features are partitioned along the last dimension of P_y.
    In the forward pass, a weight-allgather is performed along the first dimension,
    and data reduce-scatter along the last dimension (usually the model-parallel dimension).
    This version is preferred when the output feature dimension is smaller than the intput
    feature dimension. For the reverse case, see DistributedLinearAllGatherZero.
    Parameters
    ----------
    P_x :
        Partition of input/output tensor with shape of form: [ D, ..., 1, M ], where D
        is the number of data-parallel workers, and M is the number of model-parallel workers.
        Weights are distributed on the P_x partition as well.
    in_features :
        Number of features in the *global* input tensor.
    out_features :
        Number of features in the *global* output tensor.
    bias : bool
        Indicates if a bias term should be used.
    P_y : optional
        Partition of the output tensor if output is partitioned along the second last
        dimension. Shape must be of form: [ D, ..., M, 1 ].
    P_bias : optional
        Partition for biases of shape: [ D, ..., 1, 1 ].
    collect_state: bool, optional
        If true, collects the weights and biases to the root worker and
        serializes them to disk when the state_dict() function is called.
        Instead of the weights and biases, the state dictionary contains
        paths to those files. Default is false.
    checkpoint : bool, optional
        If true, use custom backward implementation that recomputes the
        all-gathers in the backward pass.
    scale_backward : Union[int, slice], optional
        Scale backward pass for AllGather operation by no. of workers along the given
        dimension. Default is None.
    """

    def __init__(self, P_x, in_features, out_features, bias=True, device=None, dtype=None,
                 P_y=None, P_bias=None, collect_state=False, checkpoint=False, scale_backward=None):

        super(DistributedLinearReduceScatterZero, self).__init__()

        # P_x is assumed to have shape [ *, 1, p]
        # Data is assumed to have shape [ *, n, channel_in/p ]
        self.P_x = P_x
        if not self.P_x.active:
            return
        else:
            assert P_x.shape[-2] == 1

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
        self.dtype = dtype
        self.checkpoint = checkpoint
        self.scale_backward = scale_backward

        # Partition for applying bias
        if P_bias is not None:
            assert P_x.dim == P_bias.dim
            assert np.prod(P_bias.shape[-2:]) == 1
            for i in range(P_x.dim - 2):
                assert P_bias.shape[i] == P_x.shape[i]
        elif bias:
            apply_bias_partition_shape = P_x.shape.copy()
            apply_bias_partition_shape[-2:] = 1

            index_bias = [slice(0, 1)] * P_x.dim
            for i in range(P_x.dim - 2):
                index_bias[i] = slice(0, P_x.shape[i])
            apply_bias_workers = worker_layout(P_x.shape)[tuple(index_bias)].reshape(-1).tolist()

            P_bias_base = P_x.create_partition_inclusive(apply_bias_workers)
            P_bias = P_bias_base.create_cartesian_topology_partition(apply_bias_partition_shape)
            P_bias_base.deactivate()

        # Store partitions for later  access
        if bias:
            self.P_bias = P_bias

        # Function to broadcast weights and biases
        self.all_gather_weight = AllGather(P_x, axes_all_gather=(0,), scale_backward=scale_backward)
        self.reduce_scatter_weight = ReduceScatter(P_x, axes_reduce_scatter=(0,))
        if bias and self.P_bias.active:
            self.all_gather_bias = AllGather(P_bias, axes_all_gather=(0,), scale_backward=scale_backward)
            self.reduce_scatter_bias = ReduceScatter(P_bias, axes_reduce_scatter=(0,))
        else:
            self.all_gather_bias = None
            self.reduce_scatter_bias = None

        # Create weights
        if P_x.active:

            # Local shape of weights, which must have the same no. of dimensions as P_x
            weight_shape = [1] * P_x.dim
            in_features_local = compute_subshape(P_x.shape[-1],
                                                 P_x.index[-1],
                                                 [in_features])[0]
            out_features_local = compute_subshape(P_x.shape[0],
                                                  P_x.index[0],
                                                  [out_features])[0]

            # Fold channel out dimension into batch dimension
            weight_shape[-1] = in_features_local
            weight_shape[0] = out_features_local

            # Create weights. Every worker either has weights or receives weights.
            self.weight = torch.nn.Parameter(torch.empty(tuple(weight_shape), **factory_kwargs))
        else:
            self.register_buffer('weight', zero_volume_tensor(device=device, requires_grad=True, dtype=self.dtype))

        # Create bias. Only 1 worker stores the bias and a subset of workers receive it.
        if self.use_bias and self.P_bias.active:
            bias_shape = [1] * P_x.dim
            bias_shape[0] = out_features_local
            self.bias = torch.nn.Parameter(torch.empty(tuple(bias_shape), **factory_kwargs))    # store bias
        else:
            self.register_parameter('bias', None)    # don't receive bias

        # Initialize parameters
        self.reset_parameters()

        # Reduce-scatter operation
        scatter_dim = torch.argmax(torch.tensor(self.P_y.shape[-2:])) + self.P_y.dim - 2
        self.reduce_scatter = ReduceScatter(self.P_y, axes_reduce_scatter=(scatter_dim,), scale_backward=scale_backward)
        self.all_gather = AllGather(self.P_y, axes_all_gather=(scatter_dim,), scale_backward=scale_backward)

        # State dict hooks for gather/scattering distributed weights
        self._register_state_dict_hook(self.gather_state_dict)
        self._register_load_state_dict_pre_hook(self.scatter_state_dict)

        # Partition for collecting weights/biases for saving the state dict
        P_root_base = P_x.create_partition_inclusive([0])
        self.P_root = P_root_base.create_cartesian_topology_partition([1] * P_x.dim)
        self.gather_weight = Repartition(P_x, self.P_root, preserve_batch=False)
        self.scatter_weight = Repartition(self.P_root, P_x, preserve_batch=False)
        if self.use_bias:
            self.gather_bias = Repartition(self.P_bias, self.P_root, preserve_batch=False)
            self.scatter_bias = Repartition(self.P_root, self.P_bias, preserve_batch=False)

    def reset_parameters(self) -> None:

        if self.P_x.active:
            init.kaiming_uniform_(self.P_x, self.weight, a=math.sqrt(5))
            weight_global_shape = assemble_global_tensor_structure(self.weight, self.P_x).shape

        if self.use_bias and self.P_bias.active:
            fan_in, _ = init._calculate_fan_in_and_fan_out(weight_global_shape)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def _unsqueeze_weight(self, weight):
        shape = [1] * self.P_y.dim
        shape[0] = weight.shape[0]
        shape[-1] = weight.shape[1]
        return weight.view(shape)

    def _squeeze_weight(self, weight):
        c_out = weight.shape[0]
        c_in = weight.shape[-1]
        return weight.view(c_out, c_in)

    def _unsqueeze_bias(self, bias):
        shape = [1] * self.P_y.dim
        shape[0] = bias.shape[0]
        return bias.view(shape)

    def _squeeze_bias(self, bias):
        c_out = bias.shape[0]
        return bias.view(c_out)

    def gather_state_dict(self, module, destination, prefix, *args):

        if self.collect_state and self.P_x.active:
            if self.use_bias and self.P_bias.active:

                # Pop bias from state dict and serialize it
                bias_key = next(reversed(destination))
                bias = self.gather_bias(destination.pop(bias_key))

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
                weight = zero_volume_tensor(device=self.P_x.device, requires_grad=True, dtype=weight.dtype)
            if self.P_x.active:
                weight = self.scatter_weight(weight)

            # Load bias
            if self.use_bias:
                bias_key = next(iter(destination))
                bias = destination.pop(bias_key)

                if self.P_root.active:
                    bias = self._unsqueeze_bias(bias)
                else:
                    bias = zero_volume_tensor(device=self.P_x.device, requires_grad=True, dtype=bias.dtype)
                if self.P_bias.active:
                    destination[bias_key] = self.scatter_bias(bias)

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

        if self.checkpoint:
            return LinearReduceScatterZeROFunc.apply(
                input,
                self.weight,
                self.bias,
                self.all_gather,
                self.reduce_scatter,
                self.all_gather_weight,
                self.reduce_scatter_weight,
                self.all_gather_bias,
                self.reduce_scatter_bias,
                self.P_bias.active,
                self.scale_backward
            )

        else:

            # Gather weights
            weight = self.all_gather_weight(self.weight)
            weight = weight.view(self.out_features, -1)

            # Broadcast bias
            if self.bias is not None and self.P_bias.active:
                bias = self.all_gather_bias(self.bias).view(self.out_features)
            else:
                bias = self.bias

            # Affine/linear transform
            y = torch.nn.functional.linear(input, weight, bias)

            # Reduce-scatter
            return self.reduce_scatter(y)
