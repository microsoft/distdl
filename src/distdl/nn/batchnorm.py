import pytorch_pfn_extras as ppe
import torch

from distdl.backends.common.tensor_comm import assemble_global_tensor_structure
from distdl.nn.broadcast import Broadcast
from distdl.nn.module import Module
from distdl.nn.repartition import Repartition
from distdl.nn.sum_reduce import SumReduce
from distdl.utilities.slicing import compute_start_index
from distdl.utilities.slicing import compute_stop_index
from distdl.utilities.slicing import worker_layout
from distdl.utilities.torch import zero_volume_tensor


class DistributedBatchNorm(Module):
    r"""A distributed batch norm layer.

    Applies Batch Normalization using mini-batch statistics.
    This layer is a distributed and generalized version of the PyTorch BatchNormNd layers.

    Parameters
    ----------
    P_x :
        Partition of the input tensor.  Outputs are of the same shape,
        and therefore re-use the input partition.
    num_features :
        Number of features in the input.
        For example, this should equal C in an input of shape (N, C, L).
    eps : optional
        A value added to the denominator for numerical stability.
        Default is 1e-5.
    momentum : optional
        The value used for the running_mean and running_var computation.
        Can be set to None for cumulative moving average (i.e. simple average).
        Default is 0.1.
    affine : optional
        a boolean value that when set to True, this module has learnable affine parameters.
        Default is True.
    track_running_stats : optional
        a boolean value that when set to True, this module tracks the running mean and variance,
        and when set to False, this module does not track such statistics and uses batch statistics
        instead in both training and eval modes if the running mean and variance are None.
        Default is True.
    scale_backward : Union[int, slice], optional
        Scale backward pass for AllGather operation by no. of workers along the given
        dimension. Default is None.
    """

    def __init__(self, P_x,
                 num_features, eps=1e-05, momentum=0.1, affine=True,
                 track_running_stats=True, device=None, dtype=None,
                 collect_state=False, scale_backward=None):
        super(DistributedBatchNorm, self).__init__()

        self.num_dimensions = len(P_x.shape)
        if self.num_dimensions < 2:
            raise ValueError('Number of dimensions of P_x should be at least 2.')
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.inputs_seen = 0
        self.collect_state = collect_state

        if device is None:
            device = P_x.device
        factory_kwargs = {'device': device, 'dtype': dtype}

        # Determine the size of the local trainable parameters (this is a bit of a hack)
        possible_input_shape = P_x.shape.tolist()
        possible_input_shape[1] = num_features
        start_index = compute_start_index(P_x.shape, P_x.index, possible_input_shape)
        stop_index = compute_stop_index(P_x.shape, P_x.index, possible_input_shape)
        self.local_num_features = stop_index[1] - start_index[1]

        internal_data_shape = [1] * self.num_dimensions
        internal_data_shape[1] = self.local_num_features
        internal_partition_shape = [1] * self.num_dimensions
        internal_partition_shape[1] = P_x.shape[1]

        # Decide which workers will be used to store sum and affine parameters
        index = [0] * self.num_dimensions
        index[1] = slice(0, P_x.shape[1])
        index = tuple(index)
        storage_workers = worker_layout(P_x.shape)[index].tolist()

        self.P_x = P_x
        P_sum_base = P_x.create_partition_inclusive(storage_workers)
        self.P_sum = P_sum_base.create_cartesian_topology_partition(internal_partition_shape)

        # Release temporary resources
        P_sum_base.deactivate()

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(internal_data_shape, **factory_kwargs))
            self.register_buffer('running_var', torch.ones(internal_data_shape, **factory_kwargs))
        else:
            self.running_mean = None
            self.running_var = None

        self.sr = SumReduce(P_x, self.P_sum)
        self.bc = Broadcast(self.P_sum, P_x)
        self.bc_affine = Broadcast(self.P_sum, P_x, scale_backward=scale_backward)

        if self.affine:
            if self.P_sum.active:
                self.gamma = torch.nn.Parameter(torch.ones(internal_data_shape, **factory_kwargs))
                self.beta = torch.nn.Parameter(torch.zeros(internal_data_shape, **factory_kwargs))
            else:
                self.register_buffer('gamma', zero_volume_tensor(device=device, requires_grad=True))
                self.register_buffer('beta', zero_volume_tensor(device=device, requires_grad=True))

            # Buffers for weight prefetching
            self.gamma_buffer = None
            self.beta_buffer = None

            # CUDA streams for weight prefetching
            if not self.P_x.device == 'cpu':
                self.stream_gamma = torch.cuda.Stream(device=self.P_x.device)
                self.stream_beta = torch.cuda.Stream(device=self.P_x.device)
            else:
                self.stream_gamma = None
                self.stream_beta = None

        # State dict hooks for gather/scattering distributed weights
        self._register_state_dict_hook(self.gather_state_dict)
        self._register_load_state_dict_pre_hook(self.scatter_state_dict)

        # Functions to collect/scatter weights
        P_root_base = self.P_sum.create_partition_inclusive([0])
        self.P_root = P_root_base.create_cartesian_topology_partition([1] * self.P_sum.dim)
        if self.affine:
            self.gather_affine = Repartition(self.P_sum, self.P_root, preserve_batch=False)
            self.scatter_affine = Repartition(self.P_root, self.P_sum, preserve_batch=False)

    def gather_state_dict(self, module, destination, prefix, *args):
        if self.collect_state and self.P_x.active:
            if self.affine:

                # Collect parameters
                beta_key = next(reversed(destination))
                beta = self.gather_affine(destination.pop(beta_key))

                gamma_key = next(reversed(destination))
                gamma = self.gather_affine(destination.pop(gamma_key))

            # Remove entries for running stats everywhere but from root
            if not self.P_root.active and self.track_running_stats:

                mean_key = next(iter(destination))
                destination.pop(mean_key)

                var_key = next(iter(destination))
                destination.pop(var_key)

            # On root, keep track of collected parameters
            if self.P_root.active and self.affine:
                destination[gamma_key] = gamma
                destination[beta_key] = beta

        return destination

    def scatter_state_dict(self, destination, prefix, *args):
        if self.collect_state and self.P_x.active:
            if self.affine:

                # Scatter parameters
                gamma_key = next(iter(destination))
                gamma = destination.pop(gamma_key)

                beta_key = next(iter(destination))
                beta = destination.pop(beta_key)

                if self.P_sum.active:
                    gamma = self.scatter_affine(gamma)
                    beta = self.scatter_affine(beta)
                else:
                    gamma = zero_volume_tensor(device=self.P_x.device, requires_grad=True, dtype=gamma.dtype)
                    beta = zero_volume_tensor(device=self.P_x.device, requires_grad=True, dtype=beta.dtype)

            # Load statitics on all ranks
            mean_key = next(iter(destination))
            mean = destination.pop(mean_key)

            var_key = next(iter(destination))
            var = destination.pop(var_key)

            destination[var_key] = var
            destination[mean_key] = mean

            # Replace with zero-volume tensors
            if self.affine:
                destination[beta_key] = beta
                destination[gamma_key] = gamma

        return destination

    def _distdl_module_setup(self, input):
        r"""Distributed batch norm module setup function.

        This function is called every time something changes in the input
        tensor structure.  It should not be called manually.

        Parameters
        ----------
        input :
            Tuple of forward inputs.  See
            `torch.nn.Module.register_forward_pre_hook` for more details.

        """

        self.global_input_shape = assemble_global_tensor_structure(input[0], self.P_x).shape

    def _compute_mean(self, input, feature_volume):
        r"""
        Compute global feature mean (i.e., across all dimensions except feature).
        Ensures all ranks have the mean tensor.

        Parameters
        ----------
        input :
            PyTorch Tensor of values that should be summed.
        feature_volume :
            Integer volume of a single feature.

        """

        x = input
        for dim in range(self.num_dimensions):
            if dim != 1:
                x = x.sum(dim, keepdim=True)
        x = self.sr(x)
        x /= feature_volume
        x = self.bc(x)
        return x

    def _compute_var(self, input, mean, feature_volume):
        r"""
        Compute global variance across all dimensions except feature.
        Ensures all ranks have the variance tensor.

        Parameters
        ----------
        input :
            PyTorch Tensor of values for which variance should be computed.
        mean :
            PyTorch Tensor of feature means of shape [1, num_features, 1, ...].
        feature_volume :
            Integer volume of a single feature.

        """

        x = (input - mean) ** 2
        return self._compute_mean(x, feature_volume)

    def _update_running_stats(self, mean, var):
        r"""
        Updates the running statistics given the new batch mean and variance.

        Parameters
        ----------
        mean :
            PyTorch Tensor of feature means of shape [1, num_features, 1, ...].
        var :
            PyTorch Tensor of feature variances of shape [1, num_features, 1, ...].

        """

        with torch.no_grad():
            # Since this is no grad, there's no adjoint step, so we can avoid
            # using a communication here and instead copy the work.
            # So, each rank holds a copy of the running statistics.
            # See: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
            if self.momentum:
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            else:
                # use a cumulative moving average instead
                self.running_mean = (mean + self.inputs_seen * self.running_mean) / (self.inputs_seen + 1)
                self.running_var = (var + self.inputs_seen * self.running_var) / (self.inputs_seen + 1)
                self.inputs_seen += 1

    def prefetch_weights(self):
        if self.stream_gamma is not None:
            with ppe.cuda.stream(self.stream_gamma):
                self.gamma_buffer = self.bc_affine(self.gamma)
        else:
            self.gamma_buffer = self.bc_affine(self.gamma)

        if self.stream_beta is not None:
            with ppe.cuda.stream(self.stream_beta):
                self.beta_buffer = self.bc_affine(self.beta)
        else:
            self.beta_buffer = self.bc_affine(self.beta)

    def forward(self, input):
        r"""Forward function interface.

        Parameters
        ----------
        input :
            Input tensor to be normalized.

        """

        if self.global_input_shape[1] != self.num_features:
            raise ValueError('num_features does not match global input shape.')

        # Because of the correctness of slicing, we may assume that:
        #   input.shape[1] == self.local_num_features

        # compute the volume of a feature
        feature_volume = self.global_input_shape[0]
        for k in self.global_input_shape[2:]:
            feature_volume *= k

        # mini-batch statistics
        if self.training:
            mean = self._compute_mean(input, feature_volume)
            var = self._compute_var(input, mean, feature_volume)
            if self.track_running_stats:
                self._update_running_stats(mean, var)
        else:
            if self.track_running_stats:
                # use the tracked batch statistics
                mean = self.running_mean
                var = self.running_var
            else:
                mean = self._compute_mean(input, feature_volume)
                var = self._compute_var(input, mean, feature_volume)

        # normalize
        x = (input - mean) / torch.sqrt(var + self.eps)

        # scale and shift
        if self.affine:
            if self.gamma_buffer is None:
                gamma = self.bc_affine(self.gamma)
                beta = self.bc_affine(self.beta)
            else:
                gamma = self.gamma_buffer
                beta = self.beta_buffer
                self.gamma_buffer = None
                self.beta_buffer = None
            if self.stream_gamma is not None and self.stream_beta is not None:
                torch.cuda.current_stream().wait_stream(self.stream_gamma)
                torch.cuda.current_stream().wait_stream(self.stream_beta)

            x = gamma * x + beta

        return x
