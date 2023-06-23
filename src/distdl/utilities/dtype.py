import numpy as np
import torch
import distdl.logger as logger

# -----------------------Extended From Pytorch -------------------------------
# https://github.com/pytorch/pytorch/blob/e180ca652f8a38c479a3eff1080efe69cbc11621/torch/testing/_internal/common_utils.py#L349

# The MIT License (MIT)

# Copyright (c) 2016 Outbrain Inc.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Dict of NumPy dtype -> torch dtype (when the correspondence exists)
numpy_to_torch_dtype_dict = {
    np.dtype(np.bool8): torch.bool,  # noqa E203
    np.dtype(np.uint8): torch.uint8,  # noqa E203
    np.dtype(np.int8): torch.int8,  # noqa E203
    np.dtype(np.int16): torch.int16,  # noqa E203
    np.dtype(np.int32): torch.int32,  # noqa E203
    np.dtype(np.int64): torch.int64,  # noqa E203
    np.dtype(np.float16): torch.float16,  # noqa E203
    np.dtype(np.float32): torch.float32,  # noqa E203
    np.dtype(np.float64): torch.float64,  # noqa E203
    np.dtype(np.complex64): torch.complex64,  # noqa E203
    np.dtype(np.complex128): torch.complex128,  # noqa E203
    np.bool8: torch.bool,  # noqa E203
    np.uint8: torch.uint8,  # noqa E203
    np.int8: torch.int8,  # noqa E203
    np.int16: torch.int16,  # noqa E203
    np.int32: torch.int32,  # noqa E203
    np.int64: torch.int64,  # noqa E203
    np.float16: torch.float16,  # noqa E203
    np.float32: torch.float32,  # noqa E203
    np.float64: torch.float64,  # noqa E203
    np.complex64: torch.complex64,  # noqa E203
    np.complex128: torch.complex128,  # noqa E203
}

# Dict of torch dtype -> NumPy dtype
torch_to_numpy_dtype_dict = {value: key for (key, value) in numpy_to_torch_dtype_dict.items()}

# -----------------------------End Extended From PyTorch ---------------------

# Get NumPy's unique numerical id numbers and map back to dtypes
numpy_to_intID_dtype_dict = {key: np.dtype(key).num for (key, value) in numpy_to_torch_dtype_dict.items()}
intID_to_numpy_dtype_dict = {value: key for (key, value) in numpy_to_intID_dtype_dict.items()}

# Also create the same mappings for torch dtypes
torch_to_intID_dtype_dict = {value: numpy_to_intID_dtype_dict[key] for (key, value) in numpy_to_torch_dtype_dict.items()}
intID_to_torch_dtype_dict = {value: key for (key, value) in torch_to_intID_dtype_dict.items()}


# TODO: where should we put this??
try:
    import cupy as cp
    from cupy.cuda import nccl

    # Dict of cupy dtype -> torch dtype (when the correspondence exists)
    cupy_to_torch_dtype_dict = {
        cp.dtype(cp.bool8): torch.bool,  # noqa E203
        cp.dtype(cp.uint8): torch.uint8,  # noqa E203
        cp.dtype(cp.int8): torch.int8,  # noqa E203
        cp.dtype(cp.int16): torch.int16,  # noqa E203
        cp.dtype(cp.int32): torch.int32,  # noqa E203
        cp.dtype(cp.int64): torch.int64,  # noqa E203
        cp.dtype(cp.float16): torch.float16,  # noqa E203
        cp.dtype(cp.float32): torch.float32,  # noqa E203
        cp.dtype(cp.float64): torch.float64,  # noqa E203
        cp.dtype(cp.complex64): torch.complex64,  # noqa E203
        cp.dtype(cp.complex128): torch.complex128,  # noqa E203
        cp.bool8: torch.bool,  # noqa E203
        cp.uint8: torch.uint8,  # noqa E203
        cp.int8: torch.int8,  # noqa E203
        cp.int16: torch.int16,  # noqa E203
        cp.int32: torch.int32,  # noqa E203
        cp.int64: torch.int64,  # noqa E203
        cp.float16: torch.float16,  # noqa E203
        cp.float32: torch.float32,  # noqa E203
        cp.float64: torch.float64,  # noqa E203
        cp.complex64: torch.complex64,  # noqa E203
        cp.complex128: torch.complex128,  # noqa E203
    }

    # Dict of torch dtype -> cupy dtype
    torch_to_cupy_dtype_dict = {value: key for (key, value) in cupy_to_torch_dtype_dict.items()}

    # Get Cupy's unique numerical id numbers and map back to dtypes
    cupy_to_intID_dtype_dict = {key: cp.dtype(key).num for (key, value) in cupy_to_torch_dtype_dict.items()}
    intID_to_cupy_dtype_dict = {value: key for (key, value) in cupy_to_intID_dtype_dict.items()}

    # Dict of torch dtype -> cupy NCCL dtype
    torch_to_nccl_dtype_dict = {
        torch.bfloat16: nccl.NCCL_BFLOAT16,  # noqa E203
        torch.double: nccl.NCCL_DOUBLE,  # noqa E203
        torch.float: nccl.NCCL_FLOAT,  # noqa E203
        torch.float16: nccl.NCCL_FLOAT16,  # noqa E203
        torch.float32: nccl.NCCL_FLOAT32,  # noqa E203
        torch.float64: nccl.NCCL_FLOAT64,  # noqa E203
        torch.half: nccl.NCCL_HALF,  # noqa E203
        torch.int: nccl.NCCL_INT,  # noqa E203
        torch.int32: nccl.NCCL_INT32,  # noqa E203
        torch.int64: nccl.NCCL_INT64,  # noqa E203
        torch.int8: nccl.NCCL_INT8,  # noqa E203
        torch.uint8: nccl.NCCL_UINT8,  # noqa E203
    }
    nccl_to_torch_dtype_dict = {value: key for (key, value) in torch_to_nccl_dtype_dict.items()}

except:
    logger.warning("Cupy dtype conversion not supported.")