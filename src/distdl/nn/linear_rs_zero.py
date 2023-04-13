import numpy as np
import torch, math

from distdl.backends.common.tensor_comm import assemble_global_tensor_structure
from distdl.nn.module import Module
from distdl.nn.reduce_scatter import ReduceScatter
from distdl.nn.all_gather import AllGather
from distdl.utilities.slicing import compute_subshape
from distdl.utilities.torch import TensorStructure
from distdl.utilities.slicing import worker_layout
from distdl.nn.repartition import Repartition
from distdl.nn.broadcast import Broadcast
from distdl.utilities.torch import zero_volume_tensor
import distdl.nn.init as init
from einops import rearrange

class DistributedLinearReduceScatterZero(Module):
   r"""A distributed linear or affine layer with 2D parallelism for weights 
    and input/outputs.

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
        Partition of input/output tensor. Must be of size 1
        in the second last dimension.
    in_features :
        Number of features in the *global* input tensor.
    out_features :
        Number of features in the *global* output tensor.
    bias : bool
        Indicates if a bias term should be used.
    P_y : optional
        Partition of the output tensor if output is partitioned
        along the second last dimension. Must be of size 1
        along the last dimension and the 2nd last dimension
        must be the same size as P_x's last dimension.
    P_weight : optional
        Partition for weights. Must have the same size as P_x in
        the last dimensions with size 1 in all other dimensions.
    P_store_bias: optional
        Partition for storing the bias. Must be of size 1 in all
        dimensions.
    P_apply_bias: optional
        Partition for applying the bias. Must be of size 1 in all 
        dimensions but the first, which must be the same size as
        P_x's first dimension.
    collect_state: bool, optional
        If true, collects the weights and biases to the root worker and
        serializes them to disk when the state_dict() function is called.
        Instead of the weights and biases, the state dictionary contains 
        paths to those files. Default is false.
    """

    def __init__(self, P_x, in_features, out_features, bias=True, device=None, dtype=None,
        P_y=None, P_bias=None, collect_state=False, geglu=False):

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

        if device is None: device = P_x.device
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.in_features = in_features
        self.out_features = out_features
        self.collect_state = collect_state
        self.use_bias = bias
        self.geglu = geglu

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

        # Function to broadcast weights and biases
        self.all_gather_weight = AllGather(P_x, axes_all_gather=(0,))
        if bias and self.P_bias.active:
            self.all_gather_bias = AllGather(P_bias, axes_all_gather=(0,))

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
            self.register_buffer('weight', zero_volume_tensor(device=device, requires_grad=True))

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
        self.reduce_scatter = ReduceScatter(self.P_y, axes_reduce_scatter=(scatter_dim,))


        # State dict hooks for gather/scattering distributed weights
        self._register_state_dict_hook(self.gather_state_dict)
        self._register_load_state_dict_pre_hook(self.scatter_state_dict)

        # Partition for collecting weights/biases for saving the state dict
        if self.collect_state:
            P_root_base = P_x.create_partition_inclusive([0])
            self.P_root = P_root_base.create_cartesian_topology_partition([1]*P_x.dim)
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

    def gather_state_dict(self, module, destination, prefix, *args):

        if self.collect_state and self.P_x.active:
            if self.use_bias and self.P_bias.active:
                
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
            else:
                weight = zero_volume_tensor(device=self.P_x.device, requires_grad=True)
            if self.P_x.active:
                weight = self.scatter_weight(weight)

            # Load bias
            if self.use_bias:
                bias_key = next(iter(destination))
                destination.pop(bias_key)

                if self.P_root.active:
                    bias = torch.load(bias_key)
                else:
                    bias = zero_volume_tensor(device=self.P_x.device, requires_grad=True)
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
            return input.clone()

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