import torch, numbers

from distdl.backends.common.tensor_comm import assemble_global_tensor_structure
from distdl.utilities.slicing import compute_subshape
from distdl.utilities.slicing import worker_layout
from distdl.utilities.torch import zero_volume_tensor
from distdl.nn.module import Module
from distdl.nn.all_sum_reduce import AllSumReduce
from distdl.nn.broadcast import Broadcast
import distdl.nn.init as init
import numpy as np

class DistributedEmbedding(Module):
    r"""A distributed embedding layer.

    Distributed version of torch.nn.Embedding for storing embeddings of
    a fixed fixed dictionary and size. Embeddings are partitioned along
    the last dimension (i.e., the embedding dimension).
    
    Parameters
    ----------
    P_x :
        Partition of the input tensor.  The last dimension of P_x is assumed to
        be the embedding dimension. The second last dimension of P_x must be 1.
    num_embeddings  : int
        Size of the dictionary of embeddings.
    embedding_dim  : int
        The size of each embedding vector.
    padding_idx : int, optional
        If specified, the entries at padding_idx do not contribute to the gradient; 
        therefore, the embedding vector at padding_idx is not updated during training, 
        i.e. it remains as a fixed “pad”. 
    max_norm : float, optional
        If given, each embedding vector with norm larger than max_norm is renormalized
        to have norm max_norm.
    norm_type : float, optional
        The p of the p-norm to compute for the max_norm option. Default 2.
    scale_grad_by_freq : bool, optional
        If given, this will scale gradients by the inverse of frequency of the words 
        in the mini-batch. Default False.
    sparse : bool, optional
        If True, gradient w.r.t. weight matrix will be a sparse tensor. 
        See Notes for more details regarding sparse gradients.
    """

    def __init__(self, P_x, num_embeddings, embedding_dim, padding_idx=None,
        max_norm=None, norm_type=2., scale_grad_by_freq=False, sparse=False,
        _weight=None, _freeze=False, collect_output=False, device=None, 
        dtype=None):

        factory_kwargs = {'device': P_x.device, 'dtype': dtype}
        super(DistributedEmbedding, self).__init__()

        # Parameters
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        self.P_x = P_x
        if not self.P_x.active:
            return

        # Partition for storing weights
        weight_partition_shape = [1] * P_x.dim
        weight_partition_shape[-1] = P_x.shape[-1]

        index_weight = [slice(0, 1)] * P_x.dim
        index_weight[-1] = slice(0, P_x.shape[-1])
        weight_workers = worker_layout(P_x.shape)[tuple(index_weight)].reshape(-1).tolist()

        P_weight_base = P_x.create_partition_inclusive(weight_workers)
        P_weight = P_weight_base.create_cartesian_topology_partition(weight_partition_shape)
        P_weight_base.deactivate()

        # Broadcast weights
        self.P_weight = P_weight
        self.broadcast = Broadcast(self.P_weight, self.P_x)

        # Local embedding size
        embedding_dim_local = compute_subshape(P_x.shape[-1],
                                               P_x.index[-1],
                                              [embedding_dim])[0]
        
        # Weights
        if _weight is not None:
            assert _weight.shape[-1] == embedding_dim_local
            assert _weight.shape[-2] == num_embeddings
            if self.P_weight.active:
                self.weight = torch.nn.Parameter(_weight, requires_grad=not _freeze)
        elif self.P_weight.active:
            self.weight = torch.nn.Parameter(torch.empty((num_embeddings, embedding_dim_local),
                **factory_kwargs), requires_grad=not _freeze)
        else:
            self.register_buffer('weight', zero_volume_tensor(device=P_x.device, 
                requires_grad=True))

    def reset_parameters(self):
        if self.P_weight.active:
            init.normal_(self.weight)
        self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self):
        if self.padding_idx is not None and self.P_weight.active:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input):
        r"""Forward function interface.

        Parameters
        ----------
        input :
            IntTensor or LongTensor of arbitrary shape containing the indices to extract.

        """
        if not self.P_x.active:
            return zero_volume_tensor(device=self.P_x.device)

        # Broadcast weights
        weight = self.broadcast(self.weight)

        return torch.nn.functional.embedding(input, weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)