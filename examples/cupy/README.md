# Some tips and tricks

## Run python (with CUDA) with proper environment vars

```bash
$ CUPY_ACCELERATORS=cub,cutensor python ex_cupy.py 
```

## Run MPI-python applications (in docker environment)

```bash
$ mpiexec -np 4 --allow-run-as-root python example
```

## Profile applications

To profile non-MPI applications:

```bash
$ CUPY_ACCELERATORS=cub,cutensor nsys profile --trace=cuda,nvtx,cublas,cublas-verbose,cusparse,cudnn \
  --cuda-memory-usage=true --force-overwrite=true --cudabacktrace=all --sampling-period=500000 \
  -o profile/profile_ex_cupy_broadcast python ex_cupy_broadcast.py
```

To profile MPI applications:

```bash
$ CUPY_ACCELERATORS=cub,cutensor nsys profile --trace=mpi,ucx,cuda,nvtx,cublas,cublas-verbose,cusparse,cudnn \
  --cuda-memory-usage=true --force-overwrite=true --sampling-period=500000 \
  -o profile/profile_ex_cupy_broadcast mpirun -np 2 --allow-run-as-root python ex_cupy_broadcast.py
```

## Basic differences between numpy and cupy

The cupy.asnumpy() method returns a NumPy array(array on the host), whereas cupy.asarray() method returns a CuPy array(array on the current device). Both methods can accept arbitrary input, meaning that they can be applied to any data that is located on either the host or device and can be converted to an array.

## References

- https://docs.cupy.dev/en/stable/user_guide/basic.html
- https://docs.cupy.dev/en/stable/user_guide/difference.html
- https://docs.cupy.dev/en/stable/reference/cuda.html
- https://docs.cupy.dev/en/stable/user_guide/memory.html
- https://docs.cupy.dev/en/stable/user_guide/interoperability.html
