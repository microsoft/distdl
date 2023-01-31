import numpy as np
import torch

from distdl.nn.module import Module
from distdl.nn.all_gather import AllGather
from distdl.utilities.slicing import compute_subshape
from distdl.utilities.torch import TensorStructure
from torch.utils.checkpoint import checkpoint


class DistributedLinearAllGather(Module):
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

    def __init__(self, P_x, in_features, out_features, bias=True, checkpointing=False):

        super(DistributedLinearAllGather, self).__init__()

        # P_x
        self.P_x = P_x
        if not self.P_x.active:
            return

        # If bias is used, only initialize it on rank 0
        if bias is True:
            stores_bias = True
        else:
            stores_bias = False

        self.in_features = in_features
        self.out_features = out_features
        self.stores_bias = stores_bias
        self.checkpointing = checkpointing
        self.serial = self.P_x.size == 1

        if self.P_x.active:

            # Input channel dimension is partitioned
            out_features_local = compute_subshape(P_x.shape[1],
                                                  P_x.index[1],
                                                 [out_features])[0]

            sublinear = torch.nn.Linear(in_features,
                                        out_features_local,
                                        bias=stores_bias,
                                        device=P_x.device)

        all_gather = AllGather(self.P_x, axes_all_gather=(1,))
        self.linear = torch.nn.Sequential(
            all_gather,
            sublinear
        )

   

    def forward(self, input):
        r"""Forward function interface.

        Parameters
        ----------
        input :
            Input tensor to the convolution.

        """

        if not self.P_x.active:
            return input.clone()

        if self.serial:
            return self.linear(input)

        if self.checkpointing:
            y = checkpoint(self.linear, input)
        else:
            y = self.linear(input)
        
        return y