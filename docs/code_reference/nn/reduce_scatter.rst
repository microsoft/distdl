===================
ReduceScatter Layer
===================

.. contents::
    :local:
    :depth: 2

Overview
========

The ReduceScatter distributed data movement primitive reduces a distributed
tensor and then partitions it along a specified dimension or set of dimensions.

In DistDL, reduce-scatter collects data from (sub)tensors along slices of a
partition.  The reduce-scatter operation applies for partitions with and
without a (Cartesian) topology.

For the purposes of this documentation, we will assume that an arbitrary
global input tensor :math:`{x}` is partitioned by :math:`P_x`.

.. note::
   The definition of a reduce-scatter in DistDL goes beyond the classical parallel
   collective operation, for example, ``MPI_Reducescatter()`` in MPI.  Such reductions 
   typically assume 1-dimensional arrays, scattered *within* a group of workers, and 
   neither impose nor exploit topological structure on the set of workers.

Motivation
==========

In distributed deep learning, there are many applications of the reduce-scatter
primitive.  For example, linear layers in which weights are distributed along 
the input channel dimension, use the reduce-scatter primitive after the matrix
multiplication to sum-reduce the output data and partition it along the same
dimension along which the input is distributed. In addition, reduce-scatter is
the adjoint of the all-gather primitive, and is therefore used in the backward
pass of distributed layers based on all-gather (such as layers using fully-sharded
data parallelism).

Implementation
==============

A back-end functional implementation supporting DistDL
:class:`~distdl.nn.ReduceScatter` allows users to specify which dimensions
of the partition the gathering happens along.  No other options are
required because the all-gather occurs within the input partition.

Assumptions
-----------

* The reduce-scatter operation is *not* in-place.  Even if the operation is
  equivalent to an identity (no dimensions are used in the reduction), a
  Torch ``.clone()`` of the tensor is returned.

* The current implementation only supports reduce-scattter along a *single*
  partitioned dimension.

Forward
-------

The forward operation sums subtensors within :math:`P_x` and then partitions
the output along a specified dimension.

* A worker that is active in :math:`P_x` will take a subtensor of :math:`x` 
  as input and return the sum of all subtensors as output, partitioned along
  the specified dimension.
* A worker that is not active in :math:`P_x` will take a zero-volume tensor
  as input and return a clone of that tensor as output.

This class provides only an interface to the back-end implementation of the
forward algorithm.  This interface does not impose any mechanism for
performing the reduction and scattering. Performance details and optimizations 
are back-end dependent.

The back-end forward operation is implemented through the `PyTorch autograd
<https://pytorch.org/docs/stable/autograd.html>`_ functional interface and
called through the ReduceScatter :math:`~distdl.nn.ReduceScatter.forward` function.

Adjoint
-------

The adjoint of the reduce-scatter primitive is the all-gather operation, which
collects subtensors within :math:`P_x` and concatenates them along the given 
data dimension.

This class provides only an interface to the back-end implementation of the
adjoint algorithm. This interface does not impose any mechanism for
performing this gathering. Performance details and optimizations are
back-end dependent.

The adjoint operation (PyTorch grad function class) is generated automatically
via autograd and calls the ``backward()`` function implemented by the back-end
functional interface.

Examples
========

To reduce-scatter a 3-dimensional tensor that lives on a ``2 x 2 x 3`` partition
along the last dimension:

>>> P_x_base = P_world.create_partition_inclusive(np.arange(0, 12))
>>> P_x = P_x_base.create_cartesian_topology_partition([2, 2, 3])
>>>
>>> x_local_shape = np.array([3, 7, 12])
>>>
>>> axes_reduce_scatter = (2,)
>>> layer = ReduceScatter(P_x, axes_reduce_scatter)
>>>
>>> x = zero_volume_tensor()
>>> if P_x.active:
>>>     x = torch.rand(*x_local_shape)
>>>
>>> y = layer(x)

The output tensor :math:`{y}` will have shape ``[3, 7, 4]``. 

API
===

.. currentmodule:: distdl.nn

.. autoclass:: ReduceScatter
    :members:

