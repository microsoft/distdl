#!/bin/bash

# Get local cuda version
CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d "," -f1 | cut -c 2-)
echo "CUDA version: $CUDA_VERSION"

# Load MPI
module load mpi/hpcx

# Install requirements
python3 -m pip install -r requirements.txt

# Install cupy (cupy-cuda11x or cupy-cuda12x depending on CUDA version)
if [[ $CUDA_VERSION == 11.* ]]; then
    echo "Install cupy version 11.x"
    python3 -m pip install cupy-cuda11x
    python3 -m cupyx.tools.install_library --cuda 11.x --library nccl

elif [[ $CUDA_VERSION == 12.* ]]; then
    echo "Install cupy version 12.x"
    python3 -m pip install cupy-cuda12x
    python3 -m cupyx.tools.install_library --cuda 12.x --library nccl

else
    echo "CUDA versions below 11.x are not supported."
fi

# Install DistDL 
python3 -m pip install -e .