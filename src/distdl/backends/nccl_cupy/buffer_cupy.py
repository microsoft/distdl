import numpy as np
#import cupy as cp
import torch

from ..common.buffer import MPIBufferManager, MPIExpandableBuffer
from distdl import backends

class MPIExpandableCupyBuffer(MPIExpandableBuffer):
    r"""NumPy (mpi4py compatible) implementation of expandable buffers.

    For use as intermediate communication buffers, as is common in MPI-based
    applications.

    An expandable buffer stores a linear (1D) array, from which contiguous
    blocks can be viewed with arbitrary shape.  This way, buffers can be
    re-used by multiple layers, for memory efficiency.

    Parameters
    ----------
    dtype : cupy.dtype
        The data type of the buffer
    initial_capacity : integer, optional
        The initial capacity of the raw buffer

    Attributes
    ----------
    dtype : cupy.dtype
        The data type of the buffer
    capacity : integer, optional
        The capacity of the raw buffer
    raw_buffer : cupy.ndarray
        The underlying 1D storage buffer
    views : dict
        Dictionary mapping shapes to contiguous numpy array views
    """

    def __init__(self, dtype, initial_capacity=0):

        # Data type of this buffer
        self.dtype = dtype

        # Current capacity
        self.capacity = initial_capacity

        # The actual storage buffer
        self.raw_buffer = torch.zeros(self.capacity, dtype=dtype, 
            device=backends.backend.get_device())

        # Map between array shapes and numpy views of contiguous chunks of the
        # raw buffer
        self.views = dict()

    def expand(self, new_capacity):
        r"""Expands the underlying buffer, creating a new view map.

        If the requested new capacity is more than the current capacity, the
        buffer is reallocated, the old buffer is copied in, and all current
        views are mapped to the new buffer.

        Parameters
        ----------
        new_capacity : int
            Proposed new capacity of the buffer.

        """

        # If the current capacity is large enough, do nothing.
        if new_capacity <= self.capacity:
            return

        # Otherwise, create a new buffer.
        new_buffer = torch.empty(new_capacity, dtype=self.dtype, 
            device=backends.backend.get_device())

        # And copy the contents of the old buffer into the new one.
        new_buffer[:len(self.raw_buffer)].copy_(self.raw_buffer) 

        # The new buffer is now the current buffer
        self.capacity = new_capacity
        self.raw_buffer = new_buffer

        # Loop over all existing views and recreate them in the new buffer.
        new_views = dict()
        for view_shape, view in self.views.items():
            view_volume = np.prod(view_shape)
            new_views[view_shape] = self.raw_buffer[:view_volume].reshape(view_shape)

        self.views = new_views

class MPICupyBufferManager(MPIBufferManager):
    r"""NumPy (mpi4py compatible) implementation of an expandable buffer
    manager.

    Provides user interface for re-using expandable buffer objects.

    Parameters
    ----------
    buffers : list
        List of buffer objects

    Attributes
    ----------
    buffers : list
        List of buffer objects
    """
    def __init__(self):
        super().__init__()

    def request_buffers(self, n_buffers, dtype, **kwargs):
        r"""Acquire a list of buffers of a specific dtype, creating them if
        required.

        Parameters
        ----------
        n_buffers : int
            Number of buffers to acquire.
        dtype : numpy.dtype
            Data type of the requested buffers.

        Returns
        -------
        List of `n_buffers` buffers with `dtype` data type.

        """

        if dtype not in self.buffers_map:
            self.buffers_map[dtype] = list()

        # Extract a list of all existing buffers with matching dtype
        dtype_buffers = self.buffers_map[dtype]

        # If there are not enough, create more buffers with that dtype
        for i in range(n_buffers - len(dtype_buffers)):
            dtype_buffers.append(MPIExpandableCupyBuffer(dtype, **kwargs))

        # Return the requested number of buffers
        return dtype_buffers[:n_buffers]
