# Some tips and tricks

## Run python (with CUDA) with proper environment vars

```bash
$ CUPY_ACCELERATORS=cub,cutensor python ex_cupy.py 
```

## Run MPI-python applications (in docker environment)

```bash
$ CUPY_ACCELERATORS=cub,cutensor mpirun -np 4 --allow-run-as-root python example
```

## Profile applications

To profile non-MPI applications:

```bash
$ CUPY_ACCELERATORS=cub,cutensor nsys profile --trace=cuda,nvtx,cublas,cublas-verbose,cusparse,cudnn \
  --cuda-memory-usage=true --force-overwrite=true --cudabacktrace=all --sampling-period=200000 \
  -o profile/profile_ex_cupy_broadcast python ex_cupy_broadcast.py
```

To profile MPI applications:

```bash
$ CUPY_ACCELERATORS=cub,cutensor nsys profile --trace=mpi,ucx,cuda,nvtx,cublas,cublas-verbose,cusparse,cudnn \
  --cuda-memory-usage=true --force-overwrite=true --sampling-period=200000 \
  -o profile/profile_ex_cupy_broadcast mpirun -np 2 --allow-run-as-root python ex_cupy_broadcast.py
```

## Basic differences between numpy and cupy

The cupy.asnumpy() method returns a NumPy array(array on the host), whereas cupy.asarray() method returns a CuPy array(array on the current device). Both methods can accept arbitrary input, meaning that they can be applied to any data that is located on either the host or device and can be converted to an array.

## How to debug multi-process Python code in VS Code

Create a debug configuration file and store it at `.vscode/launch.json`.

```json
{
  // Use IntelliSense to learn about possible attributes.
  // Hover to view descriptions of existing attributes.
  // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Remote Attach",
      "type": "python",
      "request": "attach",
      "connect": {
        "host": "localhost",
        "port": 5678
      },
      "pathMappings": [
        {
          "localRoot": "${workspaceFolder}",
          "remoteRoot": "."
        }
      ],
      "justMyCode": false
    },
    {
      "name": "Python Attach (local) proc 0",
      "type": "python",
      "request": "attach",
      "port": 5678,
      "host": "localhost",
      "justMyCode": false
    },
    {
      "name": "Python Attach (local) proc 1",
      "type": "python",
      "request": "attach",
      "port": 5679,
      "host": "localhost",
      "justMyCode": false
    },
    {
      "name": "Python Attach (local) proc 2",
      "type": "python",
      "request": "attach",
      "port": 5680,
      "host": "localhost",
      "justMyCode": false
    },
    {
      "name": "Python Attach (local) proc 3",
      "type": "python",
      "request": "attach",
      "port": 5681,
      "host": "localhost",
      "justMyCode": false
    },
    {
      "name": "GDB Attach proc 0",
      "type": "cppdbg",
      "request": "attach",
      "program": "/usr/bin/python3",
      "processId": "${command:pickProcess}",
      "MIMode": "gdb"
    },
    {
      "name": "GDB Attach proc 1",
      "type": "cppdbg",
      "request": "attach",
      "program": "/usr/bin/python3",
      "processId": "${command:pickProcess}",
      "MIMode": "gdb"
    }
  ]
}
```

Then simply install `debugpy` with pip, and use it like this:

```python
import debugpy
from mpi4py import MPI

P_world = MPIPartition(MPI.COMM_WORLD)
P_world._comm.Barrier()

debugpy.listen(('localhost', 5678 + P_world.rank))
debugpy.wait_for_client()
debugpy.breakpoint()
```

Then, attach whichever process you would like with the VS Code configurations you have created.

If the code has C/C++ parts, you can attach GDB to the running process to debug the the rest of the code.

This method also works for debugging on docker images. Just perform all these steps on the docker image, using VS Code docker extension.

For more information visit [here](https://gist.github.com/kongdd/f49fabdbf0af20ec7fd6b4f8cd1f450d) and [there](https://github.com/microsoft/ptvsd/issues/1427).

## References

- https://docs.cupy.dev/en/stable/user_guide/basic.html
- https://docs.cupy.dev/en/stable/user_guide/difference.html
- https://docs.cupy.dev/en/stable/reference/cuda.html
- https://docs.cupy.dev/en/stable/user_guide/memory.html
- https://docs.cupy.dev/en/stable/user_guide/interoperability.html
