===============
Embedding Layer
===============

.. contents::
    :local:
    :depth: 2

Overview
========

DistDL's embedding layer is a distributed implementation of the PyTorch ``nn.Embedding`` layer.
The distributed embedding layer essentially provides a distributed look-up table for a set of 
embedding vectors. This module is typically used for token embeddings in transformers.


Implementation
==============

The distributed embedding layer is implemented as ``distdl.nn.DistributedEmbedding`` and 
has the same function signature as the PyTorch version, with the only difference being that
the MPI partition is passed as the first input argument. The current implementation only
supports partitions of shape :math:`1 \times P_M`, where :math:`P_M` is the number of 
model-parallel workers. The first dimension corresponds number of embeddings and the second
dimension corresponds to the embedding vector size.

.. figure:: /_images/embedding_example_1_3.png
   :alt: Example for a distributed embedding layer.


During the forward pass, embedding weights are broadcasted to all data-parallel workers,
and a sum-reduction is applied to the updated weights during the backward pass. Along
model-parallel workers, the embedding layer does not perform any communication, as each
model-parallel worker maintains its own subset of each embedding vector.

Example
=======

The input partition :math:`P_x` can have any number of dimensions. The last dimension
of :math:`P_x` is the dimension along which the embedding vectors are distributed. The
second last dimension must be :math:`1`. During the forward pass, embedding vectors are
broadcasted to remaining dimensions (i.e. the first :math:`d-2` dimensions of :math:`P_x`).

>>> P_x_base = P_world.create_partition_inclusive(np.arange(0, 8))
>>> P_x = P_x_base.create_cartesian_topology_partition([2, 1, 4])
>>>
>>> num_embeddings = 16
>>> embedding_dim = 32
>>>
>>> input_idx = torch.arange(num_embeddings, device=P_x.device)
>>>
>>> layer = DistributedEmbedding(P_x, num_embeddings, embedding_dim)
>>>
>>> y = embedding(input_idx)


API
===

.. currentmodule:: distdl.nn

.. autoclass:: DistributedEmbedding
    :members:
