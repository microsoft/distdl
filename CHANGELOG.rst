
Changelog
=========

0.0.0 (2020-05-07)
------------------

* Package initialized.


0.0.1 (2020-06-09)
------------------

* Initial release.


0.2.0 (2020-08-18)
------------------

* Dramatically improved documentation.
* Added channel-distributed convolutional layer.
* Abstracted convolutional layer interface.  It now auto-selects
  implementation.
* Added pre-forward hooks so that communication buffers are only allocated
  when the shape of the input tensor changes.
* Improved general consistency of layer structure and member names.

0.3.0 (2020-12-01)
------------------
* Corrected use of dtype in internal buffers.
* Cleaned up partition API.
* Fixed a bug where MPI resources were not released.
* Removed assumption that transpose requires load-balanced input.
* Added smarter buffer re-use.
* Added distributed batch normalization layer.
* Added distributed upsampling interpolation layer.

0.4.0 (2021-09-01)
------------------
* Reorganized code to follow standard PyTorch naming
* Fixed bugs related to invalid convolution arguments
* Improved convolution and pooling implementations to reduce constraints on inputs
* Added all-sum-reduce
* Added distributed loss functions
* Added initial GPU support for MPI backend (experimental)
* Moved from Travis-CI to GitHub Actions
* Multiple documentation fixes


0.5.0 (2023-05-01)
------------------
* New version on Microsoft fork of original project
* Reorganized code to support multiple backends
* Add backends w/ GPU support (CUDA-aware MPI, NCCL, Torch)
* Added all-gather and reduce-scatter primitives
* Added new linear layers w/ all-gather and reduce-scatter
* Added linear layers with FSDP/ZeRO support
* Add new convolutional layers with all-gather reduce-scatter channel partitioning
* Added linear layers for mixture of experts (MoE) with all-gather and reduce-scatter
* Added layer normalization
* Added embedding layer
* Add additional examples for each layer
* Expand documentation (add new layers and primitives)