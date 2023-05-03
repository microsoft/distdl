====================
Normalization Layers
====================

.. contents::
    :local:
    :depth: 2

Batch Normalization
===================

Overview
--------

DistDL supports distributed batch normalization through the ``distdl.nn.DistributedBatchNorm`` layer.
Unlike PyTorch, DistDL does not provide separate modules for 1D, 2D, and 3D batch normalization, but
instead the dimension is inferred automatically from the input partition :math:`P_x`.

Communication of the batch normalization layer depends of the input partitioning scheme. Tensors
that are partitioned along the channel-dimension only, do not induce communication, as batch
normalization does not average over the channel-dimension. Tensors that are partitioned over any other
dimensions are averaged locally first, and then a sum-reduction over the partitioned dimensions is 
performed.

.. figure:: /_images/batch_normalization_example.png
   :alt: Example for distributed batch normalization.

Implementation
--------------

DistDL's ``nn.DistributedBatchNorm`` layer is a generalization of PyTorch's ``nn.BatchNormNd`` layer
that supports arbitrary input partitioning schemes. If learnable affine parameters are used, weights
are broadcasted to all data-parallel workers during the forward pass. Mean and variances are computed
locally, and then (sum-)reduced over the partitioned dimensions.


Example
-------

>>> P_x_base = P_world.create_partition_inclusive(np.arange(0, 8))
>>> P_x = P_x_base.create_cartesian_topology_partition([2, 1, 4])
>>>
>>> batch_size = 6
>>> num_tokens = 7
>>> num_features = 32
>>>
>>> x_local_shape = np.array([3, 7, 8])
>>>
>>> batch_norm = DistributedBatchNorm(P_x, num_tokens)
>>>
>>> x = zero_volume_tensor()
>>> if P_x.active:
>>>     x = torch.rand(*x_local_shape)
>>>
>>> y = batch_norm(x)

API
---

.. currentmodule:: distdl.nn

.. autoclass:: DistributedBatchNorm
    :members:


Layer Normalization
===================

Overview
--------

DistDL's distributed layer normalization is a generalization of PyTorch's  ``nn.LayerNorm`` module.
The layer follows the same conventions as PyTorch's version. Namely, mean and variance are computed
over the last :math:`D` dimensions of the input partitioning scheme, where :math:`D` is the number of
dimensions of `normalized_shape`.

The communication patterns of the layer normalization module are induced by the partitioning scheme of
the input tensor. As for batch normalization, input partitions with any number of dimensions and sizes
in each dimension are supported. Tensors that are partitioned along the batch dimension only (i.e. data 
parallelism), do not induce communication for the mean and variances. Other partitioning schemes
require a local reduction of the mean and variances, followed by a sum-reduction.

.. figure:: /_images/layer_normalization_example.png
   :alt: Example for distributed layer normalization.


Implementation
--------------

The distributed layer normalization module is implemented as ``distdl.nn.DistributedLayerNorm``. The
module has the same function signature as PyTorch's version with the addition of the partitioning
scheme of the input tensor. The module supports learnable affine parameters, which are broadcasted to
all data-parallel workers during the forward pass. Mean and variances are computed locally, and then
(sum-)reduced over the partitioned dimensions.

Example
-------

>>> P_x_base = P_world.create_partition_inclusive(np.arange(0, 8))
>>> P_x = P_x_base.create_cartesian_topology_partition([2, 1, 4])
>>>
>>> batch_size = 6
>>> num_tokens = 7
>>> num_features = 32
>>> normalized_shape = (num_features)
>>>
>>> x_local_shape = np.array([3, 7, 8])
>>>
>>> layer_norm = DistributedLayerNorm(P_x, normalized_shape, elementwise_affine=True)
>>>
>>> x = zero_volume_tensor()
>>> if P_x.active:
>>>     x = torch.rand(*x_local_shape)
>>>
>>> y = layer_norm(x)

API
---

.. currentmodule:: distdl.nn

.. autoclass:: DistributedLayerNorm
    :members:
