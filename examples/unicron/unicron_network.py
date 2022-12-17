import torch
import torch.nn

import distdl
from distdl.nn.conv_feature_rs import DistributedFeatureReduceScatterConv1d as DistConvRS1d
from distdl.nn.conv_feature_rs import DistributedFeatureReduceScatterConv2d as DistConvRS2d
from distdl.nn.conv_feature_rs import DistributedFeatureReduceScatterConv3d as DistConvRS3d
from distdl.utilities.slicing import compute_subshape
from distdl.nn.broadcast import Broadcast
from distdl.nn.sum_reduce import SumReduce

from layers import Concatenate
from unicron_base import UnicronBase
from unicron_base import UnicronLevelBase

_layer_type_map = {
    "conv": (None, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d),
    "conv_rs": (None, DistConvRS1d, DistConvRS2d, DistConvRS3d),
    "pool": (None, torch.nn.MaxPool1d, torch.nn.MaxPool2d, torch.nn.MaxPool3d),
    "batchnorm": (None, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)
}

_relu_inplace = False

class Unicron(UnicronBase):

    def __init__(self, P_root, P_x, *args, **kwargs):   # (..., levels, in_channels, base_channels, out_channels)

        self.P_root = P_root
        self.P_x = P_x
        self.feature_dimension = len(P_x.shape[2:])
        self.ConvType = _layer_type_map["conv"][self.feature_dimension]
        self.PoolType = _layer_type_map["pool"][self.feature_dimension]
        self.BatchNorm = _layer_type_map["batchnorm"][self.feature_dimension]

        super(Unicron, self).__init__(*args, **kwargs)


    def assemble_input_map(self):

        # Bcast
        broadcast = Broadcast(self.P_root, self.P_x, preserve_batch=False)

        # Local base channel number
        base_channels_local = compute_subshape(self.P_x.shape[1],
                                               self.P_x.index[1],
                                              [self.base_channels])[0]

        # Encoder convolution
        conv = self.ConvType(in_channels=self.in_channels,
                             out_channels=base_channels_local,
                             kernel_size=3, padding=1)
        norm = self.BatchNorm(base_channels_local)
        acti = torch.nn.ReLU(inplace=_relu_inplace)

        return torch.nn.Sequential(broadcast, conv, norm, acti)


    def assemble_cycle(self):
        return UnicronLevel(self.P_x, self.levels, 0, self.base_channels, **self.level_kwargs)

    def assemble_output_map(self):

        # Channels
        base_channels = self.base_channels
        base_channels_local = compute_subshape(self.P_x.shape[1],
                                               self.P_x.index[1],
                                              [base_channels])[0]

        conv = self.ConvType(in_channels=base_channels_local,
                             out_channels=self.out_channels,
                             kernel_size=1)

        # Sum-reduce
        sum_reduce = SumReduce(self.P_x, self.P_root, preserve_batch=False)

        return torch.nn.Sequential(conv, sum_reduce)
        

class UnicronLevel(UnicronLevelBase):

    def __init__(self, P_x, *args, **kwargs):   # (..., max_levels, level, base_channels

        self.P_x = P_x
        self.feature_dimension = len(P_x.shape[2:])
        self.ConvType = _layer_type_map["conv"][self.feature_dimension]
        self.ConvTypeRS = _layer_type_map["conv_rs"][self.feature_dimension]
        self.PoolType = _layer_type_map["pool"][self.feature_dimension]
        self.BatchNorm = _layer_type_map["batchnorm"][self.feature_dimension]

        super(UnicronLevel, self).__init__(*args, **kwargs)

    def _megatron_smoothing_block(self, in_channels, out_channels):

        # Local input/output channels
        in_channels_local = compute_subshape(self.P_x.shape[1],
                                             self.P_x.index[1],
                                             [in_channels])[0]

        out_channels_local = compute_subshape(self.P_x.shape[1],
                                             self.P_x.index[1],
                                             [out_channels])[0]
        
        # Convolution 1
        conv_in = self.ConvType(in_channels=in_channels_local,
                                out_channels=out_channels,
                                kernel_size=3, padding=1)
        norm_in = self.BatchNorm(num_features=out_channels)
        acti_in = torch.nn.ReLU(inplace=_relu_inplace)

        # Convolution 2
        conv_out = self.ConvType(in_channels=out_channels,
                                 out_channels=out_channels_local,
                                 kernel_size=3, padding=1)
        norm_out = self.BatchNorm(num_features=out_channels_local)
        acti_out = torch.nn.ReLU(inplace=_relu_inplace)

        return torch.nn.Sequential(conv_in, norm_in, acti_in, conv_out, norm_out, acti_out)


    def _megatron_finest_pre_smoothing_block(self):

        # Global input/output channels
        in_channels = self.channels(self.level)
        out_channels = self.channels(self.level)

        # Convolution
        conv = self.ConvTypeRS(self.P_x,
                               in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=1)
        return conv
    
    def _megatron_pre_smoothing_block(self):

        # Global input/output channels
        in_channels = self.channels(self.level-1)
        out_channels = self.channels(self.level)

        return self._megatron_smoothing_block(in_channels, out_channels)

    def _megatron_post_smoothing_block(self):

        # Global input/output channels
        in_channels = 2*self.channels(self.level)   # due to concatenation
        out_channels = self.channels(self.level)

        return self._megatron_smoothing_block(in_channels, out_channels)

    def _megatron_coarsest_smoothing_block(self):
        
        # Global input/output channels
        in_channels = self.channels(self.level-1)
        out_channels = self.channels(self.level)

        return self._megatron_smoothing_block(in_channels, out_channels)

    def assemble_megatron_finest_pre_smooth(self):
        return self._megatron_finest_pre_smoothing_block()

    def assemble_megatron_coarsest_smooth(self):
        return self._megatron_coarsest_smoothing_block()

    def assemble_megatron_pre_smooth(self):
        return self._megatron_pre_smoothing_block()

    def assemble_megatron_post_smooth(self):
        return self._megatron_post_smoothing_block()

    def assemble_restriction(self):
        return self.PoolType(kernel_size=2, stride=2)

    def assemble_prolongation(self):
        
        # Compute local input/output channels
        in_channels = self.channels(self.level+1)
        out_channels = self.channels(self.level)

        # Upsampling
        up = torch.nn.Upsample(scale_factor=2)

        # Convolution
        conv = self.ConvTypeRS(self.P_x,
                               in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=1)

        return torch.nn.Sequential(up, conv)

    def assemble_correction(self):
        return Concatenate(1)

    def instantiate_sublevel(self):
        return UnicronLevel(self.P_x, self.max_levels, self.level+1, self.base_channels)