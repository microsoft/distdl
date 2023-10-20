import distdl.logger as logger
import distdl.utilities.dtype as dtype_utils
import distdl.backends as backends


def convert_torch_to_model_dtype(dtype):
    if backends.backend.__name__ == "nccl_cupy":
        return dtype
    elif backends.backend.__name__ == "mpi_cupy":
        return dtype_utils.torch_to_cupy_dtype_dict[dtype]
    elif backends.backend.__name__ == "mpi_numpy":
        return dtype_utils.torch_to_numpy_dtype_dict[dtype]
    else:
        logger.error("Selected model doesn't exist!")


def convert_model_to_torch_dtype(dtype):
    if backends.backend.__name__ == "nccl_cupy":
        return dtype
    elif backends.backend.__name__ == "mpi_cupy":
        return dtype_utils.cupy_to_torch_dtype_dict[dtype]
    elif backends.backend.__name__ == "mpi_numpy":
        return dtype_utils.numpy_to_torch_dtype_dict[dtype]
    else:
        logger.error("Selected model doesn't exist!")


def convert_intID_to_model_dtype_dict(intID):
    if backends.backend.__name__ == "nccl_cupy":
        return dtype_utils.intID_to_torch_dtype_dict[intID]
    elif backends.backend.__name__ == "mpi_cupy":
        return dtype_utils.intID_to_cupy_dtype_dict[intID]
    elif backends.backend.__name__ == "mpi_numpy":
        return dtype_utils.intID_to_numpy_dtype_dict[intID]
    else:
        logger.error("Selected model doesn't exist!")


def convert_model_to_intID_dtype_dict(dtype):
    if backends.backend.__name__ == "nccl_cupy":
        return dtype_utils.torch_to_intID_dtype_dict[dtype]
    elif backends.backend.__name__ == "mpi_cupy":
        return dtype_utils.cupy_to_intID_dtype_dict[dtype]
    elif backends.backend.__name__ == "mpi_numpy":
        return dtype_utils.numpy_to_intID_dtype_dict[dtype]
    else:
        logger.error("Selected model doesn't exist!")
