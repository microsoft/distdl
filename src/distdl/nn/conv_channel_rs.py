import numpy as np
import torch, math, einops
from typing import Optional, List, Tuple, Union

from torch import Tensor
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple
from torch._torch_docs import reproducibility_notes
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t

import distdl.nn.init as init
from distdl.nn.module import Module
from distdl.nn.reduce_scatter import ReduceScatter
from distdl.nn.broadcast import Broadcast
from distdl.nn.repartition import Repartition
from distdl.utilities.slicing import compute_subshape
from distdl.utilities.torch import TensorStructure
from distdl.utilities.slicing import worker_layout
from distdl.backends.common.partition import MPIPartition
from distdl.backends.common.tensor_comm import assemble_global_tensor_structure
from distdl.utilities.torch import zero_volume_tensor

# -----------------------Extended From Pytorch -------------------------------
# https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/conv.py

# The MIT License (MIT)

# Copyright (c) 2016 Outbrain Inc.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

__all__ = ['Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d']

class _DistributedChannelReduceScatterConvNd(Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']
    __annotations__ = {'bias': Optional[torch.Tensor]}

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        ...

    P_x: MPIPartition
    in_channels: int
    _reversed_padding_repeated_twice: List[int]
    out_channels: int
    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Union[str, Tuple[int, ...]]
    dilation: Tuple[int, ...]
    transposed: bool
    output_padding: Tuple[int, ...]
    groups: int
    padding_mode: str
    collect_state: bool
    weight: Tensor
    bias: Optional[Tensor]

    def __init__(self,
                 P_x: MPIPartition,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, ...],
                 stride: Tuple[int, ...],
                 padding: Tuple[int, ...],
                 dilation: Tuple[int, ...],
                 transposed: bool,
                 output_padding: Tuple[int, ...],
                 groups: int,
                 bias: bool,
                 padding_mode: str,
                 collect_state: bool,
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': P_x.device, 'dtype': dtype}
        super().__init__()
        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_strings = {'same', 'valid'}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(
                    "Invalid padding string {!r}, should be one of {}".format(
                        padding, valid_padding_strings))
            if padding == 'same' and any(s != 1 for s in stride):
                raise ValueError("padding='same' is not supported for strided convolutions")

        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))

        self.P_x = P_x
        if not self.P_x.active:
            return
        
        # Weight partition [ 1 M 1 ...]
        weight_partition_shape = [1] * P_x.dim
        weight_partition_shape[1] = P_x.shape[1]

        index_weight = [slice(0, 1)] * P_x.dim
        index_weight[1] = slice(0, P_x.shape[1])
        weight_workers = worker_layout(P_x.shape)[tuple(index_weight)].reshape(-1).tolist()

        P_weight_base = P_x.create_partition_inclusive(weight_workers)
        P_weight = P_weight_base.create_cartesian_topology_partition(weight_partition_shape)
        P_weight_base.deactivate()

        # Bias partitions
        if bias:
            P_store_bias_base = P_x.create_partition_inclusive([0])
            P_store_bias = P_store_bias_base.create_cartesian_topology_partition([1] * P_x.dim)
            P_store_bias_base.deactivate()

            apply_bias_partition_shape = [1] * P_x.dim
            apply_bias_partition_shape[0] = P_x.shape[0]

            index_bias = [slice(0, 1)] * P_x.dim
            index_bias[0] = slice(0, P_x.shape[0])
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

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        self.collect_state = collect_state
        self.use_bias = bias
        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
            if padding == 'same':
                for d, k, i in zip(dilation, kernel_size,
                                   range(len(kernel_size) - 1, -1, -1)):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                        total_padding - left_pad)
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)

        # Initialize weights and biases
        if transposed:
            if self.P_weight.active:
                in_channels_local = compute_subshape(P_weight.shape[1],
                                                     P_weight.index[1],
                                                    [in_channels])[0]
                self.weight = torch.nn.Parameter(torch.empty(
                    (in_channels_local, out_channels // groups, *kernel_size), **factory_kwargs))
            else:
                self.register_buffer('weight', zero_volume_tensor(device=P_x.device, requires_grad=True))

        else:
            if self.P_weight.active:
                in_channels_local = compute_subshape(P_weight.shape[1],
                                                     P_weight.index[1],
                                                    [in_channels])[0]
                self.weight = torch.nn.Parameter(torch.empty(
                    (out_channels, in_channels_local // groups, *kernel_size), **factory_kwargs))
            else:
                self.register_buffer('weight', zero_volume_tensor(device=P_x.device, requires_grad=True))

        if self.use_bias and self.P_store_bias.active:
            self.bias = torch.nn.Parameter(torch.empty(out_channels, **factory_kwargs))
        elif self.use_bias and self.P_apply_bias.active:
            self.register_buffer('bias', zero_volume_tensor(device=P_x.device, requires_grad=True)) # receive bias
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        # Reduce scatter operator
        self.reduce_scatter = ReduceScatter(self.P_x, axes_reduce_scatter=(1,))

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
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        if self.P_weight.active:
            init.kaiming_uniform_(self.P_weight, self.weight, a=math.sqrt(5))
            weight_global_shape = assemble_global_tensor_structure(self.weight, self.P_weight).shape

        if self.use_bias and self.P_store_bias.active:
            fan_in, _ = init._calculate_fan_in_and_fan_out(weight_global_shape)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'

    def gather_state_dict(self, module, destination, prefix, *args):

        if self.collect_state and self.P_x.active:
            if self.bias is not None:
                
                # Pop bias from state dict and serialize it
                bias_key = next(reversed(destination))
                bias = destination.pop(bias_key)

            # Collect weights (second last entry added to dict)
            weight_key = next(reversed(destination))
            weight = destination.pop(weight_key)
            if self.transposed and self.P_weight.active:
                weight = einops.rearrange(weight, 'a b ... -> b a ...')
            weight = self.gather_weight(weight)
            if self.transposed and self.P_root.active:
                weight = einops.rearrange(weight, 'b a ... -> a b ...')

            # Serialize weights
            if self.P_root.active:

                # Add filenames back to state dict
                destination[weight_key] = weight

                if self.use_bias:
                    destination[bias_key] = bias
                
        return destination

    def scatter_state_dict(self, destination, prefix, *args):
        if self.collect_state and self.P_x.active:

            # Scatter weights
            weight_key = next(iter(destination))
            weight = destination.pop(weight_key)
            if self.P_root.active:
                if self.transposed:
                    weight = einops.rearrange(weight, 'a b ... -> b a ...')
            else:
                weight = zero_volume_tensor(device=self.P_x.device, requires_grad=True)
            if self.P_weight.active:
                weight = self.scatter_weight(weight)
                if self.transposed:
                    weight = einops.rearrange(weight, 'b a ... -> a b ...')

            # Load bias
            if self.use_bias:
                bias_key = next(iter(destination))
                bias = destination.pop(bias_key)

                if self.P_store_bias.active:
                    destination[bias_key] = bias

                elif self.P_apply_bias.active:
                    bias = zero_volume_tensor(device=self.P_x.device, requires_grad=True)
                    destination[bias_key] = bias

            if self.P_x.active:
                destination[weight_key] = weight

class DistributedChannelReduceScatterConv1d(_DistributedChannelReduceScatterConvNd):
    r"""A channel-space partitioned 1D convolutional layer.

    This class provides the user interface to a distributed convolutional
    layer, where the input (and output) tensors are partitioned in the
    channel-dimension and optionally in the batch-dimension. 
    
    This version of the layer uses a reduce-scatter operation to average and
    partition the output across workers after  the convolution operation. This
    approach is preferable when the number of output channels is smaller than 
    the number of input channels. For the opposite case, see the equivalent
    DistributedChannelAllGatherConv1d layer.

    This class requires a single tensor partition of the following shape: 

    1. :math:`P_x` over input tensor :math:`x` has shape :math:`P_{\text{data}} 
        \times P_{\text{c_in}} \times 1`.

    The first dimension of the input and output partitions is the batch
    dimension,the second is the channel dimension, and remaining dimensions
    are feature dimensions.

    The bias term does not have its own partition.  It is stored in the first
    "column" of weight partition, which is created internally by the layer.

    Parameters
    ----------
    P_x :
        Partition of input tensor of shape [ D, M, 1 ], where D is the number of
        data-parallel workers and M is the number of model-parallel workers.
    in_channels :
        Number of channels in the *global* input tensor.
    out_channels :
        Number of channels in the *global* output tensor.
    kernel_size :
        (int or tuple)
        Size of the convolving kernel
    stride :
        (int or tuple, optional)
        Stride of the convolution. Default: 1
    padding :
        (int or tuple, optional)
        Zero-padding added to both sides of the input. Default: 0
    padding_mode :
        (string, optional)
        'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
    dilation :
        (int or tuple, optional)
        Spacing between kernel elements. Default: 1
    groups :
        (int, optional)
        Number of blocked connections from input channels to output channels. Default: 1
    bias :
        (bool, optional)
        If True, adds a learnable bias to the output. Default: True
    collect_state :
        (bool, optional)
        If True, collect weights and biases on the root partition when state_dict is called.
        For set_state_dict, scatter weights and biases from the root partition. Default: False.
    device :
        (torch.device, optional)
        Device location of the layer parameters. Default: P_x.device.

    """

    def __init__(
        self,
        P_x: MPIPartition,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: Union[str, _size_1_t] = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        collect_state: bool = False,
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': P_x.device, 'dtype': dtype}
        # we create new variables below to make mypy happy since kernel_size has
        # type Union[int, Tuple[int]] and kernel_size_ has type Tuple[int]
        kernel_size_ = _single(kernel_size)
        stride_ = _single(stride)
        padding_ = padding if isinstance(padding, str) else _single(padding)
        dilation_ = _single(dilation)
        super().__init__(
            P_x, in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _single(0), groups, bias, padding_mode, collect_state, **factory_kwargs)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return torch.nn.functional.conv1d(torch.nn.functional.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _single(0), self.dilation, self.groups)
        return torch.nn.functional.conv1d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        if not self.P_x.active:
            return input.clone()

        # Broadcast weights
        weight = self.broadcast_weight(self.weight)

        # Broadcast bias
        if self.bias is not None:
            bias = self.broadcast_bias(self.bias)
        else:
            bias = self.bias

        output = self._conv_forward(input, weight, bias)
        return self.reduce_scatter(output)


class DistributedChannelReduceScatterConv2d(_DistributedChannelReduceScatterConvNd):
    r"""A channel-space partitioned 2D convolutional layer.

    This class provides the user interface to a distributed convolutional
    layer, where the input (and output) tensors are partitioned in the
    channel-dimension and optionally in the batch-dimension. 
    
    This version of the layer uses a reduce-scatter operation to average and
    partition the output across workers after  the convolution operation. This
    approach is preferable when the number of output channels is smaller than 
    the number of input channels. For the opposite case, see the equivalent
    DistributedChannelAllGatherConv2d layer.

    This class requires a single tensor partition of the following shape: 

    1. :math:`P_x` over input tensor :math:`x` has shape :math:`P_{\text{data}} 
        \times P_{\text{c_in}} \times 1 \times 1`.

    The first dimension of the input and output partitions is the batch
    dimension,the second is the channel dimension, and remaining dimensions
    are feature dimensions.

    The bias term does not have its own partition.  It is stored in the first
    "column" of weight partition, which is created internally by the layer.

    Parameters
    ----------
    P_x :
        Partition of input tensor of shape [ D, M, 1, 1 ], where D is the number
        of data-parallel workers and M is the number of model-parallel workers.
    in_channels :
        Number of channels in the *global* input tensor.
    out_channels :
        Number of channels in the *global* output tensor.
    kernel_size :
        (int or tuple)
        Size of the convolving kernel
    stride :
        (int or tuple, optional)
        Stride of the convolution. Default: 1
    padding :
        (int or tuple, optional)
        Zero-padding added to both sides of the input. Default: 0
    padding_mode :
        (string, optional)
        'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
    dilation :
        (int or tuple, optional)
        Spacing between kernel elements. Default: 1
    groups :
        (int, optional)
        Number of blocked connections from input channels to output channels. Default: 1
    bias :
        (bool, optional)
        If True, adds a learnable bias to the output. Default: True
    collect_state :
        (bool, optional)
        If True, collect weights and biases on the root partition when state_dict is called.
        For set_state_dict, scatter weights and biases from the root partition. Default: False.
    device :
        (torch.device, optional)
        Device location of the layer parameters. Default: P_x.device.

    """

    def __init__(
        self,
        P_x: MPIPartition,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        collect_state: bool = False,
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': P_x.device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            P_x, in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, collect_state, **factory_kwargs)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return torch.nn.functional.conv2d(torch.nn.functional.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return torch.nn.functional.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        if not self.P_x.active:
            return input.clone()

        # Broadcast weights
        weight = self.broadcast_weight(self.weight)

        # Broadcast bias
        if self.bias is not None:
            bias = self.broadcast_bias(self.bias)
        else:
            bias = self.bias

        output = self._conv_forward(input, weight, bias)
        return self.reduce_scatter(output)

class DistributedChannelReduceScatterConv3d(_DistributedChannelReduceScatterConvNd):
    r"""A channel-space partitioned 1D convolutional layer.

    This class provides the user interface to a distributed convolutional
    layer, where the input (and output) tensors are partitioned in the
    channel-dimension and optionally in the batch-dimension. 
    
    This version of the layer uses a reduce-scatter operation to average and
    partition the output across workers after  the convolution operation. This
    approach is preferable when the number of output channels is smaller than 
    the number of input channels. For the opposite case, see the equivalent
    DistributedChannelAllGatherConv1d layer.

    This class requires a single tensor partition of the following shape: 

    1. :math:`P_x` over input tensor :math:`x` has shape :math:`P_{\text{data}} 
        \times P_{\text{c_in}} \times 1`.

    The first dimension of the input and output partitions is the batch
    dimension,the second is the channel dimension, and remaining dimensions
    are feature dimensions.

    The bias term does not have its own partition.  It is stored in the first
    "column" of weight partition, which is created internally by the layer.

    Parameters
    ----------
    P_x :
        Partition of input tensor of shape [ D, M, 1 ], where D is the number of
        data-parallel workers and M is the number of model-parallel workers.
    in_channels :
        Number of channels in the *global* input tensor.
    out_channels :
        Number of channels in the *global* output tensor.
    kernel_size :
        (int or tuple)
        Size of the convolving kernel
    stride :
        (int or tuple, optional)
        Stride of the convolution. Default: 1
    padding :
        (int or tuple, optional)
        Zero-padding added to both sides of the input. Default: 0
    padding_mode :
        (string, optional)
        'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
    dilation :
        (int or tuple, optional)
        Spacing between kernel elements. Default: 1
    groups :
        (int, optional)
        Number of blocked connections from input channels to output channels. Default: 1
    bias :
        (bool, optional)
        If True, adds a learnable bias to the output. Default: True
    collect_state :
        (bool, optional)
        If True, collect weights and biases on the root partition when state_dict is called.
        For set_state_dict, scatter weights and biases from the root partition. Default: False.
    device :
        (torch.device, optional)
        Device location of the layer parameters. Default: P_x.device.

    """

    def __init__(
        self,
        P_x: MPIPartition,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: Union[str, _size_1_t] = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        collect_state: bool = False,
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': P_x.device, 'dtype': dtype}
        # we create new variables below to make mypy happy since kernel_size has
        # type Union[int, Tuple[int]] and kernel_size_ has type Tuple[int]
        kernel_size_ = _single(kernel_size)
        stride_ = _single(stride)
        padding_ = padding if isinstance(padding, str) else _single(padding)
        dilation_ = _single(dilation)
        super().__init__(
            P_x, in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _single(0), groups, bias, padding_mode, collect_state, **factory_kwargs)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return torch.nn.functional.conv1d(torch.nn.functional.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _single(0), self.dilation, self.groups)
        return torch.nn.functional.conv1d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        if not self.P_x.active:
            return input.clone()

        # Broadcast weights
        weight = self.broadcast_weight(self.weight)

        # Broadcast bias
        if self.bias is not None:
            bias = self.broadcast_bias(self.bias)
        else:
            bias = self.bias

        output = self._conv_forward(input, weight, bias)
        return self.reduce_scatter(output)


class _DistributedChannelReduceScatterConvTransposeNd(_DistributedChannelReduceScatterConvNd):
    def __init__(self, P_x, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode, collect_state, device=None, dtype=None) -> None:
        if padding_mode != 'zeros':
            raise ValueError('Only "zeros" padding mode is supported for {}'.format(self.__class__.__name__))

        factory_kwargs = {'device': P_x.device, 'dtype': dtype}
        super().__init__(
            P_x, in_channels, out_channels, kernel_size, stride,
            padding, dilation, transposed, output_padding,
            groups, bias, padding_mode, collect_state, **factory_kwargs)

    # dilation being an optional parameter is for backwards
    # compatibility
    def _output_padding(self, input: Tensor, output_size: Optional[List[int]],
                        stride: List[int], padding: List[int], kernel_size: List[int],
                        num_spatial_dims: int, dilation: Optional[List[int]] = None) -> List[int]:
        if output_size is None:
            ret = _single(self.output_padding)  # converting to list if was not already
        else:
            has_batch_dim = input.dim() == num_spatial_dims + 2
            num_non_spatial_dims = 2 if has_batch_dim else 1
            if len(output_size) == num_non_spatial_dims + num_spatial_dims:
                output_size = output_size[num_non_spatial_dims:]
            if len(output_size) != num_spatial_dims:
                raise ValueError(
                    "ConvTranspose{}D: for {}D input, output_size must have {} or {} elements (got {})"
                    .format(num_spatial_dims, input.dim(), num_spatial_dims,
                            num_non_spatial_dims + num_spatial_dims, len(output_size)))

            min_sizes = torch.jit.annotate(List[int], [])
            max_sizes = torch.jit.annotate(List[int], [])
            for d in range(num_spatial_dims):
                dim_size = ((input.size(d + num_non_spatial_dims) - 1) * stride[d] -
                            2 * padding[d] +
                            (dilation[d] if dilation is not None else 1) * (kernel_size[d] - 1) + 1)
                min_sizes.append(dim_size)
                max_sizes.append(min_sizes[d] + stride[d] - 1)

            for i in range(len(output_size)):
                size = output_size[i]
                min_size = min_sizes[i]
                max_size = max_sizes[i]
                if size < min_size or size > max_size:
                    raise ValueError((
                        "requested an output size of {}, but valid sizes range "
                        "from {} to {} (for an input of {})").format(
                            output_size, min_sizes, max_sizes, input.size()[2:]))

            res = torch.jit.annotate(List[int], [])
            for d in range(num_spatial_dims):
                res.append(output_size[d] - min_sizes[d])

            ret = res
        return ret


class DistributedChannelReduceScatterConvTranspose1d(_DistributedChannelReduceScatterConvTransposeNd):
    r"""A channel-space partitioned 1D transpose convolutional layer.

    This class provides the user interface to a distributed transpose
    convolutional layer, where the input (and output) tensors are partitioned 
    in the channel-dimension and optionally in the batch-dimension. 
    
    This version of the layer uses a reduce-scatter operation to average and
    partition the output across workers after  the convolution operation. This
    approach is preferable when the number of output channels is smaller than 
    the number of input channels. For the opposite case, see the equivalent
    DistributedChannelAllGatherConvTranspose1d layer.

    This class requires a single tensor partition of the following shape: 

    1. :math:`P_x` over input tensor :math:`x` has shape :math:`P_{\text{data}} 
        \times P_{\text{c_in}} \times 1`.

    The first dimension of the input and output partitions is the batch
    dimension,the second is the channel dimension, and remaining dimensions
    are feature dimensions.

    The bias term does not have its own partition.  It is stored in the first
    "column" of weight partition, which is created internally by the layer.

    Parameters
    ----------
    P_x :
        Partition of input tensor of shape [ D, M, 1 ], where D is the number
        of data-parallel workers and M is the number of model-parallel workers.
    in_channels :
        Number of channels in the *global* input tensor.
    out_channels :
        Number of channels in the *global* output tensor.
    kernel_size :
        (int or tuple)
        Size of the convolving kernel
    stride :
        (int or tuple, optional)
        Stride of the convolution. Default: 1
    padding :
        (int or tuple, optional)
        Zero-padding added to both sides of the input. Default: 0
    padding_mode :
        (string, optional)
        'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
    dilation :
        (int or tuple, optional)
        Spacing between kernel elements. Default: 1
    groups :
        (int, optional)
        Number of blocked connections from input channels to output channels. Default: 1
    bias :
        (bool, optional)
        If True, adds a learnable bias to the output. Default: True
    collect_state :
        (bool, optional)
        If True, collect weights and biases on the root partition when state_dict is called.
        For set_state_dict, scatter weights and biases from the root partition. Default: False.
    device :
        (torch.device, optional)
        Device location of the layer parameters. Default: P_x.device.

    """

    def __init__(
        self,
        P_x: MPIPartition,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        output_padding: _size_1_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_1_t = 1,
        padding_mode: str = 'zeros',
        collect_state: bool = False,
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': P_x.device, 'dtype': dtype}
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        output_padding = _single(output_padding)
        super().__init__(
            P_x, in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias, padding_mode, collect_state, **factory_kwargs)

    def forward(self, input: Tensor, output_size: Optional[List[int]] = None) -> Tensor:
        if not self.P_x.active:
            return input.clone()
            
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose1d')

        if not self.P_x.active:
            return input.clone()

        assert isinstance(self.padding, tuple)
        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
        num_spatial_dims = 1
        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size,  # type: ignore[arg-type]
            num_spatial_dims, self.dilation)  # type: ignore[arg-type]

        # Broadcast weights
        weight = self.broadcast_weight(self.weight)

        # Broadcast bias
        if self.bias is not None:
            bias = self.broadcast_bias(self.bias)
        else:
            bias = self.bias

        # Transpose convolution
        output = torch.nn.functional.conv_transpose1d(
            input, weight, bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)

        # Reduce-scatter
        return self.reduce_scatter(output)

class DistributedChannelReduceScatterConvTranspose2d(_DistributedChannelReduceScatterConvTransposeNd):
    r"""A channel-space partitioned 2D transpose convolutional layer.

    This class provides the user interface to a distributed transpose
    convolutional layer, where the input (and output) tensors are partitioned 
    in the channel-dimension and optionally in the batch-dimension. 
    
    This version of the layer uses a reduce-scatter operation to average and
    partition the output across workers after  the convolution operation. This
    approach is preferable when the number of output channels is smaller than 
    the number of input channels. For the opposite case, see the equivalent
    DistributedChannelAllGatherConvTranspose2d layer.

    This class requires a single tensor partition of the following shape: 

    1. :math:`P_x` over input tensor :math:`x` has shape :math:`P_{\text{data}} 
        \times P_{\text{c_in}} \times 1 \times 1`.

    The first dimension of the input and output partitions is the batch
    dimension,the second is the channel dimension, and remaining dimensions
    are feature dimensions.

    The bias term does not have its own partition.  It is stored in the first
    "column" of weight partition, which is created internally by the layer.

    Parameters
    ----------
    P_x :
        Partition of input tensor of shape [ D, M, 1, 1 ], where D is the number
        of data-parallel workers and M is the number of model-parallel workers.
    in_channels :
        Number of channels in the *global* input tensor.
    out_channels :
        Number of channels in the *global* output tensor.
    kernel_size :
        (int or tuple)
        Size of the convolving kernel
    stride :
        (int or tuple, optional)
        Stride of the convolution. Default: 1
    padding :
        (int or tuple, optional)
        Zero-padding added to both sides of the input. Default: 0
    padding_mode :
        (string, optional)
        'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
    dilation :
        (int or tuple, optional)
        Spacing between kernel elements. Default: 1
    groups :
        (int, optional)
        Number of blocked connections from input channels to output channels. Default: 1
    bias :
        (bool, optional)
        If True, adds a learnable bias to the output. Default: True
    collect_state :
        (bool, optional)
        If True, collect weights and biases on the root partition when state_dict is called.
        For set_state_dict, scatter weights and biases from the root partition. Default: False.
    device :
        (torch.device, optional)
        Device location of the layer parameters. Default: P_x.device.

    """

    def __init__(
        self,
        P_x: MPIPartition,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        output_padding: _size_2_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_2_t = 1,
        padding_mode: str = 'zeros',
        collect_state: bool = False,
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': P_x.device, 'dtype': dtype}
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)
        super().__init__(
            P_x, in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias, padding_mode, collect_state, **factory_kwargs)

    def forward(self, input: Tensor, output_size: Optional[List[int]] = None) -> Tensor:
        if not self.P_x.active:
            return input.clone()

        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')

        assert isinstance(self.padding, tuple)
        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
        num_spatial_dims = 2
        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size,  # type: ignore[arg-type]
            num_spatial_dims, self.dilation)  # type: ignore[arg-type]

        # Broadcast weights
        weight = self.broadcast_weight(self.weight)

        # Broadcast bias
        if self.bias is not None:
            bias = self.broadcast_bias(self.bias)
        else:
            bias = self.bias

        # Transpose convolution
        output = torch.nn.functional.conv_transpose2d(
            input, weight, bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)

        # Reduce-scatter
        return self.reduce_scatter(output)


class DistributedChannelReduceScatterConvTranspose3d(_DistributedChannelReduceScatterConvTransposeNd):
    r"""A channel-space partitioned 3D transpose convolutional layer.

    This class provides the user interface to a distributed transpose
    convolutional layer, where the input (and output) tensors are partitioned 
    in the channel-dimension and optionally in the batch-dimension. 
    
    This version of the layer uses a reduce-scatter operation to average and
    partition the output across workers after  the convolution operation. This
    approach is preferable when the number of output channels is smaller than 
    the number of input channels. For the opposite case, see the equivalent
    DistributedChannelAllGatherConvTranspose3d layer.

    This class requires a single tensor partition of the following shape: 

    1. :math:`P_x` over input tensor :math:`x` has shape :math:`P_{\text{data}} 
        \times P_{\text{c_in}} \times 1 \times 1` \times 1.

    The first dimension of the input and output partitions is the batch
    dimension,the second is the channel dimension, and remaining dimensions
    are feature dimensions.

    The bias term does not have its own partition.  It is stored in the first
    "column" of weight partition, which is created internally by the layer.

    Parameters
    ----------
    P_x :
        Partition of input tensor of shape [ D, M, 1, 1, 1 ], where D is the number
        of data-parallel workers and M is the number of model-parallel workers.
    in_channels :
        Number of channels in the *global* input tensor.
    out_channels :
        Number of channels in the *global* output tensor.
    kernel_size :
        (int or tuple)
        Size of the convolving kernel
    stride :
        (int or tuple, optional)
        Stride of the convolution. Default: 1
    padding :
        (int or tuple, optional)
        Zero-padding added to both sides of the input. Default: 0
    padding_mode :
        (string, optional)
        'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
    dilation :
        (int or tuple, optional)
        Spacing between kernel elements. Default: 1
    groups :
        (int, optional)
        Number of blocked connections from input channels to output channels. Default: 1
    bias :
        (bool, optional)
        If True, adds a learnable bias to the output. Default: True
    collect_state :
        (bool, optional)
        If True, collect weights and biases on the root partition when state_dict is called.
        For set_state_dict, scatter weights and biases from the root partition. Default: False.
    device :
        (torch.device, optional)
        Device location of the layer parameters. Default: P_x.device.

    """
    def __init__(
        self,
        P_x: MPIPartition,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: _size_3_t = 0,
        output_padding: _size_3_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_3_t = 1,
        padding_mode: str = 'zeros',
        collect_state: bool = False,
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': P_x.device, 'dtype': dtype}
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        output_padding = _triple(output_padding)
        super().__init__(
            P_x, in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias, padding_mode, collect_state, **factory_kwargs)

    def forward(self, input: Tensor, output_size: Optional[List[int]] = None) -> Tensor:
        if not self.P_x.active:
            return input.clone()

        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose3d')

        assert isinstance(self.padding, tuple)
        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
        num_spatial_dims = 3
        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size,  # type: ignore[arg-type]
            num_spatial_dims, self.dilation)  # type: ignore[arg-type]

        # Broadcast weights
        weight = self.broadcast_weight(self.weight)

        # Broadcast bias
        if self.bias is not None:
            bias = self.broadcast_bias(self.bias)
        else:
            bias = self.bias

        # Transpose convolution
        output = torch.nn.functional.conv_transpose3d(
            input, weight, bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)

        # Reduce-scatter
        return self.reduce_scatter(output)