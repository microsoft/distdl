import numpy as np
import torch, math

from distdl.nn.module import Module
from distdl.nn.reduce_scatter import ReduceScatter
from distdl.utilities.slicing import compute_subshape
from distdl.utilities.torch import TensorStructure
from torch.utils.checkpoint import checkpoint
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

    # Convolution class for base unit of work.
    TorchConvType = None

    # Number of dimensions of a feature
    num_dimensions = None

    def __init__(self, P_x, in_features, out_features, bias=True, checkpointing=False, device=None, dtype=None):

        super(DistributedLinearReduceScatter, self).__init__()

        # P_x is assumed to have shape [ *, 1, p]
        # Data is assumed to have shape [ *, in_features, out_features ]
        assert P_x.shape[-2] == 1
        if device is None: device = P_x.device
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.P_x = P_x
        if not self.P_x.active:
            return

        # If bias is used, only initialize it on rank 0 TODO check this w/ data parallelism
        if bias is True and P_x.rank == 0:
            stores_bias = True
        else:
            stores_bias = False

        self.in_features = in_features
        self.out_features = out_features
        self.stores_bias = stores_bias
        self.checkpointing = checkpointing

        # Shape of weight partition
        weight_partition_shape = [1] * P_x.dim
        weight_partition_shape[-1] = P_x.shape[-1]

        # Ranks of workers storing weights
        index = [0] * P_x.dim
        index[-1] = slice(0, P_x.shape[-1])
        index = tuple(index)
        weight_workers = worker_layout(P_x.shape)[index].tolist()

        # Create weight partitions
        P_w_base = P_x.create_partition_inclusive(weight_workers)
        P_w = P_w_base.create_cartesian_topology_partition(weight_partition_shape)
        P_w_base.deactivate()
        self.P_w = P_w

        # Function to broadcast weights from worker partition to P_x
        self.broadcast_weight = Broadcast(P_w, P_x)
        self.broadcast_bias = Broadcast(P_w, P_x)

        if P_w.active:
            # Local shape of weights, which must have the same no. of dimensions as P_x.
            weight_shape = [1] * P_x.dim
            bias_shape = [1] * P_x.dim
            in_features_local = compute_subshape(P_w.shape[-1],
                                                 P_w.index[-1],
                                                [in_features])[0]
            weight_shape[-2] = out_features
            weight_shape[-1] = in_features_local
            bias_shape[-2] = out_features

            # Create weights
            self.weight = torch.nn.Parameter(torch.empty(tuple(weight_shape), **factory_kwargs))
            if bias:
                self.bias = torch.nn.Parameter(torch.empty(tuple(bias_shape), **factory_kwargs))
            else:
                self.register_parameter('bias', None)
        else:
            self.register_buffer('weight', zero_volume_tensor(device=device, requires_grad=True))
            if bias:
                self.register_buffer('bias', zero_volume_tensor(device=device, requires_grad=True))

        self.reset_parameters()

        # Reduce-scatter operation
        self.reduce_scatter = ReduceScatter(self.P_x, axes_reduce_scatter=(P_x.dim-1,))

    # TODO account for partition sizes
    def reset_parameters(self) -> None:
        if self.P_w.active:
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
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
        bias = self.broadcast_bias(self.bias).view(self.out_features)

        # Affine/linear transform
        y = torch.nn.functional.linear(input, weight, bias)

        # Reduce-scatter
        return self.reduce_scatter(y)