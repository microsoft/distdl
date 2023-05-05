======
DistDL
======

.. .. start-badges

.. .. list-table::
..     :stub-columns: 1

..     * - docs
..       - |docs|
..     * - tests
..       - | |ci| |codecov|
..     * - package
..       - | |version| |supported-implementations|

.. .. |docs| image:: https://readthedocs.org/projects/distdl/badge/?style=flat
..     :target: https://readthedocs.org/projects/distdl
..     :alt: Documentation Status

.. .. |ci| image:: https://github.com/distdl/distdl/workflows/package%20tests/badge.svg
..     :alt: DistDL Github Actions build status
..     :target: https://github.com/distdl/distdl/actions

.. .. |travis| image:: https://api.travis-ci.com/distdl/distdl.svg?branch=master
..     :alt: Travis-CI Build Status
..     :target: https://travis-ci.com/distdl/distdl

.. .. |appveyor| image:: https://ci.appveyor.com/api/projects/status/github/distdl/distdl?branch=master&svg=true
..     :alt: AppVeyor Build Status
..     :target: https://ci.appveyor.com/project/distdl/distdl

.. .. |requires| image:: https://requires.io/github/distdl/distdl/requirements.svg?branch=master
..     :alt: Requirements Status
..     :target: https://requires.io/github/distdl/distdl/requirements/?branch=master

.. .. |codecov| image:: https://codecov.io/gh/distdl/distdl/branch/master/graphs/badge.svg?branch=master
..     :alt: Coverage Status
..     :target: https://codecov.io/github/distdl/distdl

.. .. |version| image:: https://img.shields.io/pypi/v/distdl.svg
..     :alt: PyPI Package latest release
..     :target: https://pypi.org/project/distdl

.. .. |supported-versions| image:: https://img.shields.io/pypi/pyversions/distdl.svg
..     :alt: Supported versions
..     :target: https://pypi.org/project/distdl

.. .. |supported-implementations| image:: https://img.shields.io/pypi/implementation/distdl.svg
..     :alt: Supported implementations
..     :target: https://pypi.org/project/distdl



.. end-badges

Overview
========

DistDL is a package for tensor parallelism in PyTorch. From a user perspective, it provides two main features:

* An MPI-based partitioning system to distribute tensors across multiple processes and communication primitives to move data between or within partitions.

* Tensor-parallel implementations of PyTorch's nn modules such as linear layers, convolutional layers, layer norms, et cetera.

DistDL is designed to be used in conjunction with PyTorch and is not a replacement, but rather a complement to it. DistDL provides a set of distributed modules that are designed to be used in place of their serial PyTorch counterparts. These distributed modules are designed to be used in the same way as their PyTorch counterparts, but they are implemented using tensor-parallelism. This means that the user can write their code in the same way as they would for a single-GPU PyTorch implementation, but the code will run in a distributed fashion. DistDL also provides a set of communication primitives that can be used to implement custom tensor-parallel modules.

DistDL generally does not distinguish between data-parallelism and model-parallelism. Instead, the form of parallelism is induced by the partitioning of the tensors. For example, a tensor that is partitioned across two processes along the first dimension is data-parallel, while a tensor that is partitioned across two processes along the second dimension is model-parallel. Partitioning a tensor across multiple dimensions induces a hybrid form of parallelism.

The overall package structure is as follows:

* ``distdl.nn``: Contains distributed implementations of PyTorch's nn modules (linear layers, convolutions, etc.)

* ``distdl.backends``: Contains backend-specific implementations of communication primitives and their adjoints. Currently supported backends are MPI (CPU, GPU) and NCCL (GPU).

* ``distdl.utilities``: Contains utility functions for working with distributed tensors.

* ``distdl.functional``: Contains distributed implementations of PyTorch's functional modules.


Installation
============

CPU
---

DistDL is currently still in private development and needs to be installed from source.

.. code-block:: bash
        
    # Clone repository
    git clone git@github.com:microsoft/distdl.git

    # Install locally
    cd distdl
    pip install -e .

GPU
---

If a GPU is available on the system, CUDA support can be enabled by passing the cuda version as an argument to the install command. Use ``cuda11x`` for CUDA 11.* or ``cuda12x`` for CUDA 12.*.

.. code-block:: bash

    pip install -e .[cuda11x]

NCCL support must currently be enables manually by running the following command (using either ``11.x`` or ``12.x`` as appropriate):

.. code-block:: bash

    python3 -m cupyx.tools.install_library --cuda 11.x --library nccl


Getting started
===============

The examples directory contains a number of examples that demonstrate how to use DistDL.

* Examples for using the communication primitives can be found in `examples/primitives <https://github.com/microsoft/distdl/tree/main/examples/primitives>`_.

* Examples for using DistDL's distributed modules (linear layers, convolutions, etc.) are located in `examples/basics <https://github.com/microsoft/distdl/tree/main/examples/basics>`_.


Documentation
=============

The (private) documentation is available at the following link:

`DistDL Documentation <https://didactic-succotash-69z274m.pages.github.io/>`_

Development
===========

To run the all tests run::

    mpirun -np 20 python -m mpi4py -m pytest --with-mpi -rsa tests

Substitute ``mpiexec`` or ``srun`` as correct for your system.

.. Note, to combine the coverage data from all the tox environments run:

.. .. list-table::
..     :widths: 10 90
..     :stub-columns: 1

..     - - Windows
..       - ::

..             set PYTEST_ADDOPTS=--cov-append
..             tox

..     - - Other
..       - ::

..             PYTEST_ADDOPTS=--cov-append tox
