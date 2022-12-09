import os
import distdl.logger as logger
import distdl.utilities.dtype as dtype_utils
import torch
from .. import config


def convert_torch_to_model_dtype(dtype):
    if config.array == "CUPY":
        return dtype_utils.torch_to_cupy_dtype_dict[dtype]
    if config.array == "NUMPY":
        return dtype_utils.torch_to_numpy_dtype_dict[dtype]
    if config.array == "TORCH":
        return dtype
    logger.error("Selected model doesn't exist!")


def convert_model_to_torch_dtype(dtype):
    if config.array == "CUPY":
        return dtype_utils.cupy_to_torch_dtype_dict[dtype]
    if config.array == "NUMPY":
        return dtype_utils.numpy_to_torch_dtype_dict[dtype]
    if config.array == "TORCH":
        return dtype
    logger.error("Selected model doesn't exist!")


def convert_intID_to_model_dtype_dict(intID):
    if config.array == "CUPY":
        return dtype_utils.intID_to_cupy_dtype_dict[intID]
    if config.array == "NUMPY":
        return dtype_utils.intID_to_numpy_dtype_dict[intID]
    if config.array == "TORCH":
        return dtype_utils.intID_to_torch_dtype_dict[intID]
    logger.error("Selected model doesn't exist!")


def convert_model_to_intID_dtype_dict(dtype):
    if config.array == "CUPY":
        return dtype_utils.cupy_to_intID_dtype_dict[dtype]
    if config.array == "NUMPY":
        return dtype_utils.numpy_to_intID_dtype_dict[dtype]
    if config.array == "TORCH":
        return dtype_utils.torch_to_intID_dtype_dict[dtype]
    logger.error("Selected model doesn't exist!")
