import torch, numbers

from distdl.backends.common.tensor_comm import assemble_global_tensor_structure
from distdl.utilities.slicing import compute_subshape
from distdl.nn.module import Module
from distdl.nn.all_sum_reduce import AllSumReduce
from distdl.nn.broadcast import Broadcast

import numpy as np

class DistributedLayerNorm(Module):
    r"""A distributed layer norm layer.

    Applies Layer Normalization. This layer is a distributed and generalized 
    version of the PyTorch LayerNorm layer.

    Parameters
    ----------
    P_x :
        Partition of the input tensor.  Outputs are of the same shape,
        and therefore re-use the input partition.
    normalized_shape :
        Input shape from an expected input of size. If a single integer is used, 
        it is treated as a singleton list, and this module will normalize over the 
        last dimension which is expected to be of that specific size.
    eps : optional
        A value added to the denominator for numerical stability.
        Default is 1e-5.
    elementwise_affine : optional
        A boolean value that when set to True, this module has learnable per-element affine 
        parameters initialized to ones (for weights) and zeros (for biases). Default is True.
    device: optional
        Computational device. Default is P_x.device.
    dtype:
        Data type of learnable parameters. Default is torch.float.
    """

    def __init__(self, P_x, normalized_shape, elementwise_affine=True, eps=1e-5, device=None, dtype=None):
        super(DistributedLayerNorm, self).__init__()
        
        if device is None: device = P_x.device
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        factory_kwargs = {'device': device, 'dtype': dtype}

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = tuple(normalized_shape)

        # Number of dimensions across which mean/var is computed
        self.num_dim = len(normalized_shape)    
        
        # Dimensions across which to reduce
        self.dim_reduce = tuple(torch.arange(0, P_x.dim)[-self.num_dim:])    
        self.allreduce = AllSumReduce(P_x, axes_reduce=self.dim_reduce)     # for computing mean/variance
        
        # Number of workers across we reduce
        dim_reduce_slice = slice(P_x.dim - self.num_dim, P_x.dim)
        self.num_reduce = np.prod(P_x.shape[dim_reduce_slice])   

        if self.elementwise_affine:
            # TODO
            #root_shape = [1] * self.P_x.dim
            #P_root_base = P_world.create_partition_inclusive(in_workers)
            #P_x = P_x_base.create_cartesian_topology_partition(in_shape)
            #P_root = 
            #P_weight = 
            #self.broadcast = Broadcast(P_root, P_weight)

            # Determine no. of parameters on local worker
            normalized_shape_local = compute_subshape(
                P_x.shape[dim_reduce_slice], 
                P_x.index[dim_reduce_slice], 
                normalized_shape
                )
            normalized_shape_local = tuple(normalized_shape_local)

            self.bias = torch.nn.Parameter(torch.empty(normalized_shape_local, **factory_kwargs))
            self.weight = torch.nn.Parameter(torch.empty(normalized_shape_local, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def _compute_mean(self, input):
        r"""
        Compute global feature mean (i.e., across the last d dimensions,
        where d is the dimension of self.normalized_shape).
        Ensures all ranks have the mean tensor.

        Parameters
        ----------
        input :
            PyTorch Tensor of values that should be summed.
        """
        # Local mean
        output = input.mean(dim=self.dim_reduce, keepdim=True)

        # Average across workers
        return self.allreduce(output) / self.num_reduce

    def _compute_var(self, input, mean):
        r"""
        Compute global variance across last d dimensions,
        where d is the dimension of self.normalized_shape.
        Ensures all ranks have the variance tensor.

        Parameters
        ----------
        input :
            PyTorch Tensor of values for which variance should be computed.
        """
        input = (input - mean)**2
        return self._compute_mean(input)
    

    def forward(self, input):
        r"""Forward function interface.

        Parameters
        ----------
        input :
            Input tensor to be normalized.

        """

        # Calculate mean and variance
        mean = self._compute_mean(input)
        var = self._compute_var(input, mean)

        # Re-scale
        input = (input - mean) / torch.sqrt(var + self.eps)

        if self.elementwise_affine:
            input = self.weight*input + self.bias

        return input