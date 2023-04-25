===============
AllGather Layer
===============

.. contents::
    :local:
    :depth: 2

Overview
========

The AllGather distributed data movement primitive gathers data along the partitioned
dimension within a set of workers of a single Partition.

In DistDL, all-gather collects data from (sub)tensors along slices of a
partition.  The all-sum-reduce operation applies for partitions with and
without a (Cartesian) topology.

For the purposes of this documentation, we will assume that an arbitrary
global input tensor :math:`{x}` is partitioned by :math:`P_x`.

.. note::
   The definition of a all-gather in DistDL goes beyond the classical parallel
   collective operation, for example, ``MPI_Allgather()`` in MPI.  Such primitives 
   typically assume 1-dimensional arrays, scattered *within* a group of workers, and 
   neither impose nor exploit topological structure on the set of workers.

Motivation
==========

In distributed deep learning, there are many applications of the all-gather
primitive.  For example, for fully-sharded data parallelism (FSDP), weights
are partitioned along the data-parallel workers to reduce the memory imprint
per worker. During forward (or backward) pass, the full weight tensor is 
temporarily collected on each worker via the all-gather primitive. Another
example of the all-gather primitive is for linear layers, in which the input
data is partitioned and all-gather is called on the data prior to the subsequent
matrix multiplication.

Implementation
==============

A back-end functional implementation supporting DistDL
:class:`~distdl.nn.AllGather` allows users to specify which dimensions
of the partition the gathering happens along.  No other options are
required because the all-gather occurs within the input partition.

Assumptions
-----------

* The all-gather operation is *not* in-place.  Even if the operation is
  equivalent to an identity (no dimensions are used in the reduction), a
  Torch ``.clone()`` of the tensor is returned.

* The current implementation only supports all-gather along a *single*
  partitioned dimension.

Forward
-------

The forward operation gathers subtensors within :math:`P_x` along a specified
dimension.

* A worker that is active in :math:`P_x` will take a subtensor of :math:`x` 
  as input and return a copy of the global version of :math:`x` as output.
* A worker that is not active in :math:`P_x` will take a zero-volume tensor
  as input and return a clone of that tensor as output.

This class provides only an interface to the back-end implementation of the
forward algorithm.  This interface does not impose any mechanism for
performing the reduction.  Performance details and optimizations are back-end
dependent.

The back-end forward operation is implemented through the `PyTorch autograd
<https://pytorch.org/docs/stable/autograd.html>`_ functional interface and
called through the AllGather :math:`~distdl.nn.AllGather.forward` function.

Adjoint
-------

The adjoint of the all-gather primitive is the reduce-scatter operation, which
sums subtensors within :math:`P_x` and then partitions it along the specified
dimension.

This class provides only an interface to the back-end implementation of the
adjoint algorithm. This interface does not impose any mechanism for
performing this reduce-scatter. Performance details and optimizations are
back-end dependent.

The adjoint operation (PyTorch grad function class) is generated automatically
via autograd and calls the ``backward()`` function implemented by the back-end
functional interface.

Examples
========

To all-gather a 3-dimensional tensor that lives on a ``2 x 2 x 3`` partition
along the last dimension:

>>> P_x_base = P_world.create_partition_inclusive(np.arange(0, 12))
>>> P_x = P_x_base.create_cartesian_topology_partition([2, 2, 3])
>>>
>>> x_local_shape = np.array([3, 7, 4])
>>>
>>> axes_all_gather = (2,)
>>> layer = AllGather(P_x, axes_all_gather)
>>>
>>> x = zero_volume_tensor()
>>> if P_x.active:
>>>     x = torch.rand(*x_local_shape)
>>>
>>> y = layer(x)

The output tensor :math:`{y}` will have shape ``[3, 7, 12]``. 

API
===

.. currentmodule:: distdl.nn

.. autoclass:: AllGather
    :members:

