import numpy as np
import torch, math

from distdl.backends.common.tensor_comm import assemble_global_tensor_structure
from distdl.nn.module import Module
from distdl.nn.reduce_scatter import ReduceScatter
from distdl.utilities.slicing import compute_subshape
from distdl.utilities.torch import TensorStructure
from distdl.utilities.slicing import worker_layout
from distdl.nn.repartition import Repartition
from distdl.nn.broadcast import Broadcast
from distdl.utilities.torch import zero_volume_tensor
import distdl.nn.init as init
from einops import rearrange

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
        Partition of input/output tensor. Must be of size 1
        in the second last dimension (i.e. the channel out
        dimension.)
    in_features :
        Number of features in the *global* input tensor.
    out_features :
        Number of features in the *global* output tensor.
    bias : bool
        Indicates if a bias term should be used.
    P_y: optional
        Partition for the output if output is partitioned
        along the second last dimension instead of the last.
        Must have the same size and shape as P_x with the 
        second two last dimensions swapped.
    P_weight : optional
        Partition for the weights. Must be of size 1 in
        every dimension but the last.
    P_store_bias: optional
        Partition for storing the bias. Must have the same
        dimension as P_x with size 1 in every dimension.
    P_apply_bias: optional
        Partition on which bias is applied. Must have the 
        same size as P_x in every but the last 2 dimensions,
        which must be 1.
    collect_state: bool, optional
        If true, creates partitions to gather/scatter weights & 
        biases for saving/loading state dictionaries.
    """

    def __init__(self, P_x, in_features, out_features, bias=True, device=None, dtype=None, 
        P_y=None, P_weight=None, P_store_bias=None, P_apply_bias=None, 
        collect_state=False, geglu=False):

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

        if device is None: device = P_x.device
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.in_features = in_features
        self.out_features = out_features
        self.collect_state = collect_state
        self.use_bias = bias
        self.geglu = geglu

        # Partition for storing weights (the partition for applying weights is P_x)
        if P_weight is not None:
            assert P_x.dim == P_weight.dim
            assert P_weight.shape[-1] == P_x.shape[-1]
            assert np.prod(P_weight.shape[:P_x.dim-1]) == 1
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
            for i in range(P_x.dim-2):
                assert P_apply_bias.shape[i] == P_x.shape[i]
        elif bias:
            apply_bias_partition_shape = P_x.shape.copy()
            apply_bias_partition_shape[-2:] = 1

            index_bias = [slice(0, 1)] * P_x.dim
            for i in range(P_x.dim-2):
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
        self.broadcast_weight = Broadcast(P_weight, P_x)
        if bias and self.P_apply_bias.active:
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
            self.register_buffer('bias', zero_volume_tensor(device=device, requires_grad=True)) # receive bias
        else:
           self.register_parameter('bias', None)    # don't receive bias

        # Initialize parameters
        self.reset_parameters()

        # Reduce-scatter operation
        scatter_dim = torch.argmax(torch.tensor(self.P_y.shape[-2:])) + self.P_y.dim - 2
        self.reduce_scatter = ReduceScatter(self.P_y, axes_reduce_scatter=(scatter_dim,))

        # State dict hooks for gather/scattering distributed weights
        self._register_state_dict_hook(self.gather_state_dict)
        self._register_load_state_dict_pre_hook(self.scatter_state_dict)

        # Partition for collecting weights/biases for saving the state dict
        if self.collect_state:
            P_root_base = P_x.create_partition_inclusive([0])
            self.P_root = P_root_base.create_cartesian_topology_partition([1]*P_x.dim)
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

    def gather_state_dict(self, module, destination, prefix, *args):

        if self.collect_state and self.P_x.active:
            if self.use_bias and self.bias is not None:
                
                # Pop bias from state dict and serialize it
                bias_key = next(reversed(destination))
                bias = destination.pop(bias_key)

                if self.P_store_bias.active:
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
                pass
                weight = torch.load(weight_key)
            else:
                weight = zero_volume_tensor(device=self.P_x.device, requires_grad=True)
            if self.P_weight.active:
                weight = self.scatter_weight(weight)

            # Load bias
            if self.use_bias:
                bias_key = next(iter(destination))
                destination.pop(bias_key)

                if self.P_store_bias.active:
                    bias = torch.load(bias_key)
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
            return input.clone()

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