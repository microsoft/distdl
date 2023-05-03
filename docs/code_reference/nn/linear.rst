============
Linear Layer
============

.. contents::
    :local:
    :depth: 2


Overview
========

The Distributed Linear (or affine) layer uses distributed primitive layers
to build a distributed version of the PyTorch ``Linear`` layer.  That is,
it implements

.. math::
   y = Wx + b

where the tensors :math:`x`, :math:`y`, :math:`W`, and :math:`b` are
partitioned over a number of workers.

Implementation
==============

DistDL provides different implementations of the distributed linear layer,
that generally differ in the way the input/output and weight tensors are 
partitioned. The partitioning scheme in turn induces the choice of the 
primitive used for communication. The following versions are currently 
supported:

* *Linear layer with all-gather*: Weights are partitioned along the output feature dimension. The layer performs an all-gather of the input along the model-parallel workers, followed by a local GEMM. A fully-sharded data parallel version of this layer is available as well.

* *Linear layer with reduce-scatter*: Weights are partitioned along the input feature dimension. The layer performs a local GEMM first and then applies a reduce-scatter along the model-parallel workers. A fully-sharded data parallel version of this layer is supported.

* *General linear layer*: Weights are partitioned along both the input and output feature dimensions. This version first performs a broadcast of the data, followed by a local GEMM and sum-reduction.


Public interfaces
-----------------

DistDL provides a public interface to the many distributed linear layer
implementations that follows the same pattern as other public interfaces
and keeps in line with the PyTorch interface.  The ``distdl.nn`` module provides the
:class:`distdl.nn.DistributedLinearAllGather`,
:class:`distdl.nn.DistributedLinearReduceScatter`,
:class:`distdl.nn.DistributedLinearAllGatherZero`,
:class:`distdl.nn.DistributedLinearReduceScatterZero`, and
:class:`distdl.nn.DistributedLinear` classes, based on the structure 
of :math:`P_x`, :math:`P_y`, and the choice of the underlying primitive.

Linear layer with all-gather
----------------------------

The all-gather version of the linear layer is based on the all-gather primitive, which is called on the input tensor prior to performing the matrix multiplication. This version of the linear layer is preferred over the reduce-scatter variant when the number of input features/channels is smaller than the number of output features/channels.

Weights in the all-gather version are partitioned along the output feature dimension. The respective tensor partition :math:`P_w` is created internally. Two options for partitioning the input data are available, whereas the output tensor is always partitioned along the first (batch) and last (feature/channel/embedding) dimension.

Single input/output partition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: /_images/linear_example_ag_p_x.png
   :alt: Example for linear layer with all-gather and single input/output partition.

Both the input and output tensors are partitioned on the same partition :math:`P_x`, whose shape is :math:`P_d \times 1 \times ... \times P_m`, where :math:`P_d` is the number of data-parallel workers and :math:`P_m` is the number of model-parallel workers. Weights are partitioned internally on :math:`P_w` along the output channel/feature dimension. The bias (not shown in the figure) is created on one of the model-parallel partitions only. During the forward pass, the input tensor is all-gathered along the model-parallel dimension, followed by a local GEMM. The output tensor is then partitioned along the same dimension as the input tensor.


>>> P_x_base = P_world.create_partition_inclusive(np.arange(0, 8))
>>> P_x = P_x_base.create_cartesian_topology_partition([2, 1, 4])
>>>
>>> in_features = 16
>>> out_features = 12
>>>
>>> x_local_shape = np.array([2, 8, 4])
>>>
>>> layer = DistributedLinearAllGather(P_x, in_features, out_features)
>>>
>>> x = zero_volume_tensor()
>>> if P_x.active:
>>>     x = torch.rand(*x_local_shape)
>>>
>>> y = layer(x)


Separate input/output partitions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: /_images/linear_example_ag_p_x_p_y.png
   :alt: Example for linear layer with all-gather and separate input/output partitions.

The input and output tensor are partitioned on separate partitions. The input tensor is partitioned on :math:`P_x`, whose shape is :math:`P_d \times ... \times P_m \times 1`, where :math:`P_d` is the number of data-parallel workers and :math:`P_m` is the number of model-parallel workers. The output tensor is partitioned on :math:`P_y`, whose shape is :math:`P_d \times 1 \times ... \times P_m`. Weights are partitioned internally on :math:`P_w` along the input channel/feature dimension. During the forward pass, the input tensor is all-gathered along the model-parallel dimension, followed by a local GEMM. 

>>> P_x_base = P_world.create_partition_inclusive(np.arange(0, 8))
>>> P_x = P_x_base.create_cartesian_topology_partition([2, 1, 4])
>>>
>>> P_y = P_x_base.create_cartesian_topology_partition([2, 4, 1])
>>>
>>> in_features = 16
>>> out_features = 12
>>>
>>> x_local_shape = np.array([2, 8, 4])
>>>
>>> layer = DistributedLinearAllGather(P_y, in_features, out_features, P_x=P_x)
>>>
>>> x = zero_volume_tensor()
>>> if P_y.active:
>>>     x = torch.rand(*x_local_shape)
>>>
>>> y = layer(x)

Fully-sharded data parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both conventional data parallelism and fully-sharded data parallelism (FSDP or ZeRO-3) are supported. In standard data parallelism, weights are only initialized on one set of model-parallel parallel workers and they are broadcasted to their respective data-parallel counterparts during the forward pass (see figure below). During the backward pass, the broadcast induces a sum-reduce operation. Note that DistDL differs in this regard from other frameworks for data parallelism, in which each worker (or set of model-parallel workers) keep a copy of the weights and then perform an all-reduce operation along (data-parallel) workers during the backward pass.

.. figure:: /_images/linear_example_ag_dp.png
   :alt: Example for linear layer with all-gather and standard data parallelism.

The FSDP version of the linear all-gather layer partitions weights along the data-parallel workers in addition to the model-parallel workers. During the forward pass, model-parallel workers collect their full local tensor through an all-gather along the data-parallel workers.

.. figure:: /_images/linear_example_ag_zero.png
   :alt: Example for linear layer with all-gather and fully-sharded data parallelism.

Note that the linear all-gather layer with FSDP is implemented as a separate module, rather than being a wrapper that is called on the distributed all-gather version. The reason for this is that the FSDP layer directly initializes the fully-sharded weights on each worker, rather than partitioning the weights after the initialization.

>>> P_x_base = P_world.create_partition_inclusive(np.arange(0, 8))
>>> P_x = P_x_base.create_cartesian_topology_partition([2, 1, 4])
>>>
>>> P_y = P_x_base.create_cartesian_topology_partition([2, 4, 1])
>>>
>>> in_features = 16
>>> out_features = 12
>>>
>>> x_local_shape = np.array([2, 8, 4])
>>>
>>> layer = DistributedLinearAllGatherZero(P_y, in_features, out_features, P_x=P_x)
>>>
>>> x = zero_volume_tensor()
>>> if P_y.active:
>>>     x = torch.rand(*x_local_shape)
>>>
>>> y = layer(x)

Examples
~~~~~~~~

Linear layer with reduce-scatter
--------------------------------

The reduce-scatter version of the linear layer is based on the reduce-scatter primitive, which is called on the output tensor after performing the matrix multiplication. The reduce-scatter version of the linear layer is preferred over the all-gather version when the number of output features/channels is smaller than the number of input features/channels.

Weights in the reduce-scatter version are partitioned along the input feature dimension. The respective tensor partition :math:`P_w` is created internally. Two options for partitioning the output data are available, whereas the input tensor is always partitioned along the first (batch) and last (feature/channel/embedding) dimension.


Single input/output partition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: /_images/linear_example_rs_p_x.png
   :alt: Example for linear layer with reduce-scatter and single input/output partition.

Both the input and output tensors are partitioned on the same partition :math:`P_x`, whose shape is :math:`P_d \times 1 \times ... \times P_m`, where :math:`P_d` is the number of data-parallel workers and :math:`P_m` is the number of model-parallel workers. Weights are partitioned internally on :math:`P_w` along the input channel/feature dimension. The bias (not shown in the figure) is partitioned along the model-parallel partitions as well. During the forward pass, the GEMM is carried out first, followed by a reduce-scatter to sum the output across the model-parallel workers and partition it along the last dimension. 

>>> P_x_base = P_world.create_partition_inclusive(np.arange(0, 8))
>>> P_x = P_x_base.create_cartesian_topology_partition([2, 1, 4])
>>>
>>> in_features = 16
>>> out_features = 12
>>>
>>> x_local_shape = np.array([2, 8, 4])
>>>
>>> layer = DistributedLinearReduceScatter(P_x, in_features, out_features)
>>>
>>> x = zero_volume_tensor()
>>> if P_x.active:
>>>     x = torch.rand(*x_local_shape)
>>>
>>> y = layer(x)


Separate input/output partitions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: /_images/linear_example_rs_p_x_p_y.png
   :alt: Example for linear layer with reduce-scatter and single input/output partition.

The input and output tensor are partitioned on two distinct partitions. The input tensor is partitioned on :math:`P_x`, whose shape is :math:`P_d \times ... \times 1 \times P_m`, where :math:`P_d` is the number of data-parallel workers and :math:`P_m` is the number of model-parallel workers. The output tensor is partitioned on :math:`P_y`, whose shape is :math:`P_d \times ... \times P_m \times 1`. Weights are partitioned internally on :math:`P_w` along the output channel/feature dimension. During the forward pass, the local GEMM is followed by a reduce-scatter operation, which sums the output across the model-parallel workers and partitions it along the 2nd last dimension.

>>> P_x_base = P_world.create_partition_inclusive(np.arange(0, 8))
>>> P_x = P_x_base.create_cartesian_topology_partition([2, 1, 4])
>>>
>>> P_y = P_x_base.create_cartesian_topology_partition([2, 4, 1])
>>>
>>> in_features = 16
>>> out_features = 12
>>>
>>> x_local_shape = np.array([2, 8, 4])
>>>
>>> layer = DistributedLinearReduceScatter(P_x, in_features, out_features, P_y=P_y)
>>>
>>> x = zero_volume_tensor()
>>> if P_x.active:
>>>     x = torch.rand(*x_local_shape)
>>>
>>> y = layer(x)

Fully-sharded data parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The FSDP version of the linear reduce-scatter version operates similar to the all-gather version. Weights are partitioned along the model-parallel workers across the input feature dimension and along data-parallel workers across the output feature dimension.

>>> P_x_base = P_world.create_partition_inclusive(np.arange(0, 8))
>>> P_x = P_x_base.create_cartesian_topology_partition([2, 1, 4])
>>>
>>> P_y = P_x_base.create_cartesian_topology_partition([2, 4, 1])
>>>
>>> in_features = 16
>>> out_features = 12
>>>
>>> x_local_shape = np.array([2, 8, 4])
>>>
>>> layer = DistributedLinearReduceScatterZero(P_x, in_features, out_features, P_y=P_y)
>>>
>>> x = zero_volume_tensor()
>>> if P_x.active:
>>>     x = torch.rand(*x_local_shape)
>>>
>>> y = layer(x)


General linear layer
--------------------


Assumptions
~~~~~~~~~~~

* The global input tensor :math:`x` has shape :math:`n_{\text{batch}} \times
  n_{\text{features in}}`.
* The input partition :math:`P_x` has shape :math:`1 \times P_{\text{f_in}}`,
  where :math:`P_{\text{f_in}}` is the number of workers partitioning the
  feature dimension of :math:`x`.
* The global output tensor :math:`y` has shape :math:`n_{\text{batch}} \times
  n_{\text{features out}}`.
* The output partition :math:`P_y` has shape :math:`1 \times P_{\text{f_out}}`,
  where :math:`P_{\text{f_out}}` is the number of workers partitioning the
  feature dimension of :math:`y`.

.. note::
   PyTorch admits input tensors of shape :math:`n_{\text{batch}} \times \dots
   \times n_{\text{features in}}` and output tensors of shape
   :math:`n_{\text{batch}} \times \dots \times n_{\text{features out}}`.
   DistDL does not explicitly support intermediate dimensions at this time.

* The weight tensor :math:`W` has shape :math:`n_{\text{features_out}} \times
  n_{\text{features_in}}`.  This follows PyTorch.
* The weight partition :math:`P_W` has shape :math:`P_{\text{f_out}} \times
  P_{\text{f_in}}`.

.. note::
   The bias vectors are stored on the 0th *column* of :math:`P_w`.  Hence, it
   is implicitly partitioned by a factor of :math:`P_{\text{f_in}}`.
   Following PyTorch, if the bias is turned off, no subtensors have bias
   terms.

.. figure:: /_images/linear_example_01.png
   :alt: Example setup for distributed linear layer.

   An example setup for a distributed linear layer, where :math:`P_x` has
   shape :math:`1 \times 4`, :math:`P_y` has shape :math:`1 \times 3`, and
   :math:`P_W` has shape :math:`3 \times 4`.

Forward
~~~~~~~

Under the above assumptions, the forward algorithm is:

1. Use a :ref:`code_reference/nn/broadcast:Broadcast Layer` to broadcast
   subtensors of :math:`x` from :math:`P_x` over the columns of :math:`P_W`.

.. figure:: /_images/linear_example_02.png
   :alt: Example forward broadcast in the distributed linear layer.

   Subtensors of :math:`x` are broadcast down the four columns of
   :math:`P_W`.

2. Perform the local forward linear layer application using a PyTorch Linear
   layer.  Note that the bias is only added on the 0th column of :math:`P_W`.
   Each worker now has a portion of the output vector :math:`y`.  In the rows
   of :math:`P_W` the results are partial contributions to the output feature
   degrees-of-freedom.

.. figure:: /_images/linear_example_03.png
   :alt: Example forward linear application in the distributed linear layer.

   Local application of linear layer.  Bias is present only in 0th column.

3. Use a :ref:`code_reference/nn/sum_reduce:SumReduce Layer` to reduce
   the subtensors of :math:`y` over the rows of :math:`P_W` into :math:`P_y`.
   Only one subtensor in each row of :math:`P_W` contains the a subtensor of
   the bias, so the output tensor correctly assimilates the bias.

   .. note::
      This sum-reduction requires one of the partitions to be transposed.

.. figure:: /_images/linear_example_04.png
   :alt: Example forward sum-reduction in the distributed linear layer.

   Subtensors of :math:`y` are assembled via sum-reduction along the three
   rows of :math:`P_W`.

Adjoint
~~~~~~~

The adjoint algorithm is not explicitly implemented.  PyTorch's ``autograd``
feature automatically builds the adjoint of the Jacobian of the distributed
linear forward application.  Essentially, the algorithm is as follows:

1. Broadcast the subtensors of the gradient output, :math:`\delta y` from
   :math:`P_y` along the rows of :math:`P_W`.

.. figure:: /_images/linear_example_05.png
   :alt: Example adjoint sum-reduction in the distributed linear layer.

   Subtensors of :math:`\delta y` are broadcast across the three rows of
   :math:`P_W`.

2. Each worker in :math:`P_W` computes its local part of :math:`\delta W` and
   :math:`\delta x` using the PyTorch implementation of the adjoint of the
   Jacobian of the local sequential linear layer.  If the bias is required,
   the 0th column of :math:`P_W` also computes :math:`\delta b` similarly.

.. figure:: /_images/linear_example_06.png
   :alt: Example adjoint linear application in the distributed linear layer.

   Local computation of subtensors of :math:`\delta x`, :math:`\delta W`, and
   :math:`\delta b`.

3. Sum-reduce the subtensors of the gradient input, :math:`\delta x`, along
   the rows of :math:`P_W` into :math:`P_x`.

.. figure:: /_images/linear_example_07.png
   :alt: Example adjoint broadcast in the distributed linear layer.

   Subtensors of :math:`\delta x` are assembled via sum-reduction along the
   four columns of :math:`P_W`.


Examples
~~~~~~~~

To apply a linear layer which maps a tensor on a ``1 x 4`` partition to a
tensor on a ``1 x 3`` partition:

>>> P_x_base = P_world.create_partition_inclusive(np.arange(0, 4))
>>> P_x = P_x_base.create_cartesian_topology_partition([1, 4])
>>>
>>> P_y_base = P_world.create_partition_inclusive(np.arange(4, 7))
>>> P_y = P_y_base.create_cartesian_topology_partition([1, 3])
>>>
>>> P_W_base = P_world.create_partition_inclusive(np.arange(0, 12))
>>> P_W = P_W_base.create_cartesian_topology_partition([3, 4])
>>>
>>> in_features = 16
>>> out_features = 12
>>>
>>> x_local_shape = np.array([1, 4])
>>>
>>> layer = DistributedLinear(P_x, P_y, P_W, in_features, out_features)
>>>
>>> x = zero_volume_tensor()
>>> if P_x.active:
>>>     x = torch.rand(*x_local_shape)
>>>
>>> y = layer(x)

API
===

.. currentmodule:: distdl.nn

.. autoclass:: DistributedLinear
    :members:
