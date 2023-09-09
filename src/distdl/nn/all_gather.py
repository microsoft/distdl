from distdl.nn.module import Module
from distdl.utilities.torch import TensorStructure
from distdl.utilities.slicing import compute_subshape_along_axis, compute_subshape
import numpy as np
import torch

class AllGather(Module):
    r"""A distributed allgather layer.

    This class provides the user interface to the allgather distributed data 
    movement primitive.  Implementation details are back-end specific.

    The AllGather algorithm performs an allgather within a single partition. 
    Thus, the standard DistDL sum-reduction/broadcast rules are implicitly 
    satisfied.

    Functionally, the input tensor is gathered along dimensions specified by
    the `axes_all_gather` field and the result of that reduction is scattered
    along the same dimensions.  However, the underlying implementation will
    not typically apply these two operations directly.

    One of `axes_all_gather` or `axes_keep`, only, may be set.

    Parameters
    ----------
    P_x :
        Partition of input and output tensor.
    axes_all_gather : tuple, optional
        Partition dimensions along which the allreduction and scattering takes place.
        Currently, only supportes all-gather operation along single dimension.
    axes_keep : tuple, optional
        Partition dimensions to reduce-scatter to.  Complement of `axes_all_gather`.
        Currently, only supportes all-gather operation along single dimension.

    """

    def __init__(self, P_x, axes_all_gather=None, axes_keep=None, use_frontend=False):

        super(AllGather, self).__init__()

        # Partition of input and output tensor.
        self.P_x = P_x

        # Partition dimensions along which the reduce-scatter takes place.
        # While we compute both terms, `axes_all_gather` is used internally.
        if axes_all_gather is None and axes_keep is None:
            raise ValueError("One of `axes_all_gather` or `axes_keep` must be specified.")
        elif axes_all_gather is not None and axes_keep is not None:
            raise ValueError("Only one of `axes_all_gather` or `axes_keep` may be specified.")
        elif axes_all_gather is not None:
            self.axes_all_gather = axes_all_gather
            self.axes_keep = [d for d in range(P_x.dim) if d not in axes_all_gather]
        elif axes_keep is not None:
            self.axes_all_gather = [d for d in range(P_x.dim) if d not in axes_keep]
            self.axes_keep = axes_keep

        # Indicates if broadcast requires any data movement.
        self.identity = False

        # Use frontend?
        self.use_frontend = use_frontend

        # Partition for performing reduce-scatter.
        self.P_allgather = self._distdl_backend.Partition()

        # Structure of the input tensor (shape, dtype, requires_grad, etc).
        self.input_tensor_structure = TensorStructure()

        # Structure of the output tensor (shape, dtype, requires_grad, etc).
        self.output_tensor_structure = TensorStructure()

        # Store slices to correctly reshape input/output
        self.slices = None

        # Variables for tracking input changes and buffer construction
        self._distdl_is_setup = False
        self._input_tensor_structure = TensorStructure()

        # The identity case is if the partition is of size 1,
        if self.P_x.size == 1:
            self.identity = True

    @staticmethod
    def _assemble_slices(out_local_shape, P_x, axis):

        # Reshape target array to correct form
        axis = axis[0]
        cart_slices = []
        flat_slices = []

        # We slice the input data along the dimension of the reduce-scatter
        # operation and insert it into the right location in the 1D array.
        for i in range(P_x.shape[axis]):
            
            # Slice source array
            target_slice = []
            origin_slice = slice(i*np.prod(out_local_shape), (i+1)*np.prod(out_local_shape))
            for j in range(len(out_local_shape)):
                if j == axis:
                    target_slice.append(slice(int(i*out_local_shape[axis]), int((i+1)*out_local_shape[axis])))
                else:
                    target_slice.append(slice(None))        
            cart_slices.append(tuple(target_slice))
            flat_slices.append(origin_slice)

        return cart_slices, flat_slices

    def _distdl_module_setup(self, input):
        r"""AllGather module setup function.

        Constructs the necessary partition functions to implement the above
        described allgather pattern.  This function performs collective
        communication across the input and output partitions.

        This function is called every time something changes in the input
        tensor structure.  It should not be called manually.

        Parameters
        ----------
        input :
            Tuple of forward inputs.  See
            `torch.nn.Module.register_forward_pre_hook` for more details.

        """

        if not (self.P_x.active):
            return

        # If it is not an identity, we need actual Partitions to do the work.
        if not self.identity:

            self.P_allgather = self.P_x.create_allreduction_partition(self.axes_all_gather, 
                initialize_backend_comm=True, use_frontend=self.use_frontend)
            self.input_tensor_structure = TensorStructure(input[0])
            self.output_tensor_structure = TensorStructure(input[0])
            self._distdl_backend.assemble_global_tensor_structure_along_axis(self.input_tensor_structure,
                                                                             self.P_x,
                                                                             self.axes_all_gather)
            self.slices = self._assemble_slices(self.input_tensor_structure.shape, 
                self.P_x, self.axes_all_gather)

        self._distdl_is_setup = True
        self._input_tensor_structure = TensorStructure(input[0])

    def _distdl_module_teardown(self, input):
        r"""AllGather module teardown function.

        Nullifies the necessary partition functions.

        This function is called every time something changes in the input
        tensor structure.  It should not be called manually.

        Parameters
        ----------
        input :
            Tuple of forward inputs.  See
            `torch.nn.Module.register_forward_pre_hook` for more details.

        """

        # Reset all of the buffers and communication objects
        self.P_allgather.deactivate()

        # Reset any data stored about the tensor
        self.input_tensor_structure = TensorStructure()
        self.output_tensor_structure = TensorStructure()

        # Reset any info about the input
        self._distdl_is_setup = False
        self._input_tensor_structure = TensorStructure()

    def _distdl_input_changed(self, input):
        r"""Determine if the structure of inputs has changed.

        Parameters
        ----------
        input :
            Tuple of forward inputs.  See
            `torch.nn.Module.register_forward_pre_hook` for more details.

        """

        new_tensor_structure = TensorStructure(input[0])

        return self._input_tensor_structure != new_tensor_structure

    def forward(self, input):
        """Forward function interface.

        Parameters
        ----------
        input :
            Input tensor to be reduce-scattered.

        """

        Function = self._distdl_backend.functional.all_gather.AllGatherFunction

        if self.identity:
            return input#.clone()

        if not (self.P_x.active):
            return input#.clone()

        return Function.apply(input,
                              self.P_allgather,
                              self.input_tensor_structure,
                              self.output_tensor_structure,
                              self.slices)
