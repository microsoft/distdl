from contextlib import nullcontext

import pytorch_pfn_extras as ppe
import torch

import distdl.nn.init as init
from distdl.nn.all_gather import AllGather
from distdl.nn.module import Module
from distdl.nn.reduce_scatter import ReduceScatter
from distdl.nn.repartition import Repartition
from distdl.utilities.misc import stream_barrier
from distdl.utilities.slicing import compute_subshape
from distdl.utilities.torch import zero_volume_tensor


class DistributedEmbeddingZero(Module):
    r"""A distributed embedding layer with FSDP.

    Distributed version of torch.nn.Embedding for storing embeddings of
    a fixed fixed dictionary and size. Embeddings are partitioned along the first
    and the last dimension (i.e., the embedding dimension). Sequence parallelism
    (partitioning the 2nd last dimension) is currently not supported.

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
    collect_state : bool, optional
        If True, the entire embedding matrix is gathered to the root worker and
        serialized to disk when the state_dict() function is called. Instead
        of the weight itself, the state dictionary will contain a path to the
        serialized file. Default False.
    device : torch.device, optional
        Device on which to allocate the embedding matrix. Default is the device
        as specified by the input partition P_x.
    dtype : torch.dtype, optional
        Data type of the embedding matrix. Default is torch.float32.
    scale_backward : Union[int, slice], optional
        Scale backward pass for AllGather operation by no. of workers along the given
        dimension. Default is None.
    auto_clear_buffer: bool, optional
        If true, clears the weight buffers after each forward pass. Default is True.
        For ZeRO stage 1 and to take advantage of gradient accumulation, set this
        to False and call clear_weight_buffer() manually after the optimizer step.
    """

    def __init__(self, P_x, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2., scale_grad_by_freq=False, sparse=False,
                 _weight=None, _freeze=False, collect_state=False, device=None,
                 dtype=None, scale_backward=None, auto_clear_buffer=True):

        factory_kwargs = {'device': P_x.device, 'dtype': dtype}
        super(DistributedEmbeddingZero, self).__init__()

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
        self.collect_state = collect_state
        self.dtype = dtype
        self.auto_clear_buffer = auto_clear_buffer
        self.scale_backward = scale_backward

        self.P_x = P_x
        if not self.P_x.active:
            return

        # Root partition for initializing weights
        P_root_base = P_x.create_partition_inclusive([0])
        self.P_root = P_root_base.create_cartesian_topology_partition([1] * self.P_x.dim)
        P_root_base.deactivate()

        # Allgather
        self.allgather = AllGather(self.P_x, axes_all_gather=(0,), scale_backward=scale_backward)
        self.reducescatter = ReduceScatter(self.P_x, axes_reduce_scatter=(0,))
        self.init_scatter = Repartition(self.P_root, self.P_x)

        # Local embedding size
        num_embeddings_local = compute_subshape(P_x.shape[0],
                                                P_x.index[0],
                                                [num_embeddings])[0]
        embedding_dim_local = compute_subshape(P_x.shape[-1],
                                               P_x.index[-1],
                                               [embedding_dim])[0]
        # Weights
        if _weight is not None:
            assert _weight.shape[-1] == embedding_dim_local
            assert _weight.shape[-2] == num_embeddings_local
            if self.P_x.active:
                self.weight = torch.nn.Parameter(_weight, requires_grad=not _freeze)
        elif self.P_x.active:
            self.weight = torch.nn.Parameter(torch.empty((num_embeddings_local, embedding_dim_local),
                                             **factory_kwargs), requires_grad=not _freeze
                                             )
            self.reset_parameters()
        else:
            self.register_buffer('weight', zero_volume_tensor(device=P_x.device,
                                 requires_grad=True, dtype=self.dtype))

        # Buffer and stream for weight prefetching
        self.weight_buffer = None
        if not self.P_x.device == 'cpu':
            self.stream_context = nullcontext  # ppe.cuda.stream TODO Fix
            self.stream_weight = torch.cuda.Stream(device=self.P_x.device)
        else:
            self.stream_context = nullcontext
            self.stream_weight = None

        # State dict hooks for gather/scattering distributed weights
        self._register_state_dict_hook(self.gather_state_dict)
        self._register_load_state_dict_pre_hook(self.scatter_state_dict)

        # Gather/collect weights for saving/setting state dict
        P_root_base = P_x.create_partition_inclusive([0])
        self.P_root = P_root_base.create_cartesian_topology_partition([1] * P_x.dim)
        self.gather_weight = Repartition(P_x, self.P_root, preserve_batch=False)
        self.scatter_weight = Repartition(self.P_root, P_x, preserve_batch=False)

    def reset_parameters(self, init=init.normal_, mean=0.0, std=1.0):
        if self.P_x.active:
            weight_shape = [1] * self.P_x.dim
            weight_shape[0] = self.num_embeddings
            weight_shape[-1] = self.embedding_dim
            weight = torch.empty(weight_shape, device=self.P_x.device)
            init(weight, mean=mean, std=std)
            weight = self.init_scatter(weight)
            weight = weight.view(weight.shape[0], weight.shape[-1])
            with torch.no_grad():
                self.weight[:] = weight
        self._fill_padding_idx_with_zero()

    def extra_repr(self) -> str:
        return f'num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}'

    def _fill_padding_idx_with_zero(self):
        if self.padding_idx is not None and self.P_x.active:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def _expand(self, weight):
        if self.P_x.active:
            weight_view = [1] * self.P_x.dim
            weight_view[0] = weight.shape[-2]
            weight_view[-1] = weight.shape[-1]

            weight = weight.view(weight_view)
        return weight

    def _squeeze(self, weight):
        weight = weight.view(weight.shape[0], weight.shape[-1])
        return weight

    def gather_state_dict(self, module, destination, prefix, *args):
        if self.collect_state and self.P_x.active:

            # Collect weights (second last entry added to dict)
            weight_key = next(reversed(destination))
            weight = self._squeeze(self.gather_weight(self._expand(destination.pop(weight_key))))

            # Serialize weights
            if self.P_root.active:

                # Add filenames back to state dict
                destination[weight_key] = weight

        return destination

    def scatter_state_dict(self, destination, prefix, *args):
        if self.collect_state and self.P_x.active:

            # Scatter weights
            weight_key = next(iter(destination))
            if self.P_root.active:
                weight = destination.pop(weight_key)
                weight = self._expand(weight)
            else:
                dtype = destination.pop(weight_key).dtype
                weight = zero_volume_tensor(device=self.P_x.device, requires_grad=True, dtype=dtype)
            weight = self._squeeze(self.scatter_weight(weight))

            destination[weight_key] = weight

        return destination

   # Keep for backward compatibility
    def prefetch_weights(self):
        self.collect_weights()

    def collect_weights(self):
        # For ZeRO-1 (auto_clear_buffer: False), we want to temporarily turn the weight buffer
        # into a leaf tensor in which gradients are accumulated. Therefore, run the weight collection
        # in no_grad mode and then manually set requires_grad to True. For ZeRO-3, just track the all-gather
        # operation as usual.
        if not self.auto_clear_buffer:
            with torch.no_grad():
                self._collect_weights()
                self.weight_buffer.requires_grad = True
        else:
            self._collect_weights()

    def _collect_weights(self):

        # If weight buffer is not already filled, start an allgather call. If cuda is used,
        # this call will be asynchronously executed in a separate stream.
        if self.weight_buffer is None:
            with self.stream_context(self.stream_weight):
                self.weight_buffer = self._squeeze(self.allgather(self._expand(self.weight)))

    def clear_weight_buffer(self):

        # For ZeRO-1, this function must be manually called after the gradient accumulation loop.
        # Only at this point do we call the reduce-scatter operations and populate the weight
        # gradient. For ZeRO-3, this function is called after each forward pass and  reduce-scatter
        # is called automatically, since the forward all-gather was tracked.
        if not self.auto_clear_buffer:
            with torch.no_grad():

                # Reduce-scatter gradient
                if self.scale_backward is not None:
                    self.weight_buffer.grad.div_(self.scale_backward)
                self.weight.grad = self._squeeze(self.reducescatter(self._expand(self.weight_buffer.grad)))

        self.weight_buffer = None

    def forward(self, input):
        r"""Forward function interface.

        Parameters
        ----------
        input :
            IntTensor or LongTensor of arbitrary shape containing the indices to extract.

        """
        if not self.P_x.active:
            return zero_volume_tensor(device=self.P_x.device, dtype=self.dtype)

        # All-gather weights into the weight buffer. If prefetch_weights() has been
        # called previously, this doesn't do anything.
        self.collect_weights()
        stream_barrier(self.stream_weight)

        input = torch.nn.functional.embedding(input, self.weight_buffer, self.padding_idx, self.max_norm,
                                              self.norm_type, self.scale_grad_by_freq, self.sparse
                                              )
        # Clear weight buffers
        if self.auto_clear_buffer:
            self.clear_weight_buffer()

        return input
