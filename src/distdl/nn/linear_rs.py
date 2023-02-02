import numpy as np
import torch, math

from distdl.nn.module import Module
from distdl.nn.reduce_scatter import ReduceScatter
from distdl.utilities.slicing import compute_subshape
from distdl.utilities.torch import TensorStructure
from distdl.utilities.slicing import worker_layout
from distdl.nn.broadcast import Broadcast
from distdl.utilities.torch import zero_volume_tensor

class DistributedLinearReduceScatter(Module):
    r"""A distributed linear or affine layer.

    This class provides the user interface to a distributed linear layer.
    It utlizes back-end specific parallel data movement primitives but
    does not require its own back-end implementation.

    The base unit of work is given by the partition over the weight tensor.
    This class requires the following of the tensor partitions:

    1. :math:`P_x` over input/output tensor :math:`x` has shape :math:`1 \times
       P_{\text{f_in}}`.

    The bias term does not have its own partition.  The first dimension of the
    input and output partitions is the batch dimension and the second is the
    feature dimension.

    .. warning::
       This departs from PyTorch Linear layers, which allow intermediate
       dimensions in the tensors.

    Parameters
    ----------
    P_x :
        Partition of input/output tensor.
    in_features :
        Number of features in the *global* input tensor.
    out_features :
        Number of features in the *global* output tensor.
    bias : bool
        Indicates if a bias term should be used.

    """

    def __init__(self, P_x, in_features, out_features, bias=True, device=None, dtype=None,
        P_weight=None, P_store_bias=None, P_apply_bias=None):

        super(DistributedLinearReduceScatter, self).__init__()

        # P_x is assumed to have shape [ *, 1, p]
        # Data is assumed to have shape [ *, n, channel_in/p ]
        if not P_x.active:
            return
        else:        
            assert P_x.shape[-2] == 1

        if device is None: device = P_x.device
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.in_features = in_features
        self.out_features = out_features

        # Partition for storing weights (the partition for applying weights is P_x)
        if P_weight is not None:
            assert P_x.dim == P_weight.dim
            assert P_weight.shape[-1] == P_x.shape[-1]
            assert np.prod(P_weight.shape[:P_x.dim-1]) == 1
        else:
            weight_partition_shape = [1] * P_x.dim
            weight_partition_shape[-1] = P_x.shape[-1]

            index_weight = [0] * P_x.dim
            index_weight[-1] = slice(0, P_x.shape[-1])
            index_weight = tuple(index_weight)
            weight_workers = worker_layout(P_x.shape)[index_weight].tolist()

            P_weight_base = P_x.create_partition_inclusive(weight_workers)
            P_weight = P_weight_base.create_cartesian_topology_partition(weight_partition_shape)
            P_weight_base.deactivate()

        # Partition for storing bias
        if P_store_bias is not None:
            assert P_x.dim == P_store_bias.dim
            assert np.prod(P_store_bias.shape) == 1
        else:
            P_store_bias_base = P_x.create_partition_inclusive([0])
            P_store_bias = P_store_bias_base.create_cartesian_topology_partition([1] * P_x.dim)
            P_store_bias_base.deactivate()

        # Partition for applying bias
        if P_apply_bias is not None:
            assert P_x.dim == P_apply_bias.dim
            assert np.prod(P_apply_bias.shape[-2:]) == 1
            for i in range(P_x.dim-2):
                assert P_apply_bias.shape[i] == P_x.shape[i]
        else:
            index_bias = [0] * P_x.dim
            for i in range(P_x.dim-2):
                index_bias[i] = slice(0, P_x.shape[i])
            apply_bias_workers = worker_layout(P_x.shape)[tuple(index_bias)].tolist()

            apply_bias_partition_shape = P_x.shape.copy()
            apply_bias_partition_shape[-2:] = 1
            
            P_apply_bias_base = P_x.create_partition_inclusive(apply_bias_workers)
            P_apply_bias = P_apply_bias_base.create_cartesian_topology_partition(apply_bias_partition_shape)
            P_apply_bias_base.deactivate()

        # Store partitions for later  access
        self.P_x = P_x
        self.P_weight = P_weight
        self.P_store_bias = P_store_bias
        self.P_apply_bias = P_apply_bias

        # Function to broadcast weights from worker partition to P_x
        self.broadcast_weight = Broadcast(P_weight, P_x)
        if self.P_apply_bias.active:
            self.broadcast_bias = Broadcast(P_store_bias, P_apply_bias)

        # Create weights
        if P_weight.active:

            # Local shape of weights, which must have the same no. of dimensions as P_x
            weight_shape = [1] * P_x.dim
            in_features_local = compute_subshape(P_weight.shape[-1],
                                                 P_weight.index[-1],
                                                [in_features])[0]
            weight_shape[-1] = in_features_local
            weight_shape[-2] = out_features

            # Create weights
            self.weight = torch.nn.Parameter(torch.empty(tuple(weight_shape), **factory_kwargs))
        else:
            self.register_buffer('weight', zero_volume_tensor(device=device, requires_grad=True))

        # Create bias
        if self.P_store_bias.active:
            bias_shape = [1] * P_x.dim
            bias_shape[-2] = out_features
            self.bias = torch.nn.Parameter(torch.empty(tuple(bias_shape), **factory_kwargs))

        elif self.P_apply_bias.active:
            self.register_buffer('bias', zero_volume_tensor(device=device, requires_grad=True))

        else:
           self.register_parameter('bias', None)

        # Initialize parameters
        self.reset_parameters()

        # Reduce-scatter operation
        self.reduce_scatter = ReduceScatter(self.P_x, axes_reduce_scatter=(P_x.dim-1,))

    # TODO account for partition sizes
    def reset_parameters(self) -> None:

        if self.P_weight.active:
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.P_store_bias.active:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            print("fan: ", fan_in)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)


    def forward(self, input):
        r"""Forward function interface.

        Parameters
        ----------
        input :
            Input tensor to the convolution.

        """

        if not self.P_x.active:
            return input.clone()

        # Broadcast weights to everyone
        weight = self.broadcast_weight(self.weight).view(self.out_features, -1)

        # Broadcast bias
        if self.P_apply_bias.active:
            bias = self.broadcast_bias(self.bias).view(self.out_features)
        else:
            bias = self.bias

        # Affine/linear transform
        y = torch.nn.functional.linear(input, weight, bias)

        # Reduce-scatter
        return self.reduce_scatter(y)