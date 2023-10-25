===========
MPI Backend
===========

.. contents::
    :local:
    :depth: 3

Overview
========

Tensor Partitions
=================

.. currentmodule:: distdl.backends.common

.. autoclass:: Partition

.. autoclass:: CartesianPartition

.. automodule:: distdl.backends.common.partition
    :members:
    :undoc-members:

.. automodule:: distdl.backends.common.tensor_comm
    :members:
    :undoc-members:

Primitive Functionals
=====================

All-Gather
----------

.. automodule:: distdl.backends.mpi_numpy.functional.all_gather

.. autoclass:: distdl.backends.mpi_numpy.functional.AllGatherFunction
    :members:
    :undoc-members:

All-Sum-Reduce
--------------

.. automodule:: distdl.backends.mpi_numpy.functional.all_sum_reduce

.. autoclass:: distdl.backends.mpi_numpy.functional.AllSumReduceFunction
    :members:
    :undoc-members:

Broadcast
---------

.. automodule:: distdl.backends.mpi_numpy.functional.broadcast

.. autoclass:: distdl.backends.mpi_numpy.functional.BroadcastFunction
    :members:
    :undoc-members:

Halo Exchange
-------------

.. automodule:: distdl.backends.mpi_numpy.functional.halo_exchange

.. autoclass:: distdl.backends.mpi_numpy.functional.HaloExchangeFunction
    :members:
    :undoc-members:

Reduce-Scatter
--------------

.. automodule:: distdl.backends.mpi_numpy.functional.reduce_scatter

.. autoclass:: distdl.backends.mpi_numpy.functional.ReduceScatterFunction
    :members:
    :undoc-members:

Sum-Reduce
----------

.. automodule:: distdl.backends.mpi_numpy.functional.sum_reduce

.. autoclass:: distdl.backends.mpi_numpy.functional.SumReduceFunction
    :members:
    :undoc-members:

Repartition
-----------

.. automodule:: distdl.backends.mpi_numpy.functional.repartition

.. autoclass:: distdl.backends.mpi_numpy.functional.RepartitionFunction
    :members:
    :undoc-members:

Misc
====

.. automodule:: distdl.backends.mpi_numpy.device
    :members:
    :undoc-members:

.. automodule:: distdl.backends.mpi_numpy.get_device
    :members:
    :undoc-members:

.. automodule:: distdl.backends.mpi_numpy.set_device
    :members:
    :undoc-members:
