import os
import numpy as np
import cupy as cp
import torch

from abc import ABC, abstractmethod


class MPIExpandableBuffer(ABC):
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

    @abstractmethod
    def __init__(self, dtype, initial_capacity=0):
        pass

    @abstractmethod
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
        pass

    def allocate_view(self, view_shape):
        r"""Reserves a new view, with provided the shape, into the buffer.

        A NumPy `ndarray` view of the 1D internal buffer with the desired
        shape will be created.

        Parameters
        ----------
        view_shape : iterable
            Shape of the desired view.

        """

        # Ensure the shape is hashable
        view_shape = tuple(view_shape)

        self.get_view(view_shape)

    def get_view(self, view_shape):
        r"""Returns a view, with the provided shape, into the buffer.

        A NumPy `ndarray` view of the 1D internal buffer with the desired
        shape will be returned.  If the view does not exist it will be
        created.

        Parameters
        ----------
        view_shape : iterable
            Shape of the desired view.

        Returns
        -------
        The requested view.

        """

        # Ensure the shape is hashable
        view_shape = tuple(view_shape)

        # If there is already a view of this shape, then return it
        if view_shape in self.views:
            return self.views[view_shape]

        # If new shape is larger than the buffer, expand the buffer
        view_volume = np.prod(view_shape)

        if view_volume > self.capacity:
            self.expand(view_volume)

        # Create the new view and return it
        # TODO: ask from Russel
        self.views[view_shape] = self.raw_buffer[:view_volume].reshape(view_shape)

        return self.views[view_shape]


class MPIBufferManager(ABC):
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

        self.buffers_map = dict()

    @abstractmethod
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
        pass
