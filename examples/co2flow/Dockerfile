FROM nvcr.io/nvidia/pytorch:22.03-py3

RUN apt-get update \
    && apt-get install -y --no-install-recommends hdf5-tools 
    
RUN pip3 install --no-cache-dir azure-common azure-storage-blob h5py zarr mpi4py

# Clone DistDL
WORKDIR /workspace/home

# Environment variables
ENV CONTAINER="sleipner"
ENV DATA_PATH="sleipner64"
ENV ACCOUNT_URL="https://pwitteblob.blob.core.windows.net"