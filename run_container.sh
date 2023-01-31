#!/bin/bash

docker run --privileged=true \
    --gpus all --sig-proxy=false --cap-add=ALL \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /home/azureuser/pwitte:/workspace/home \
    -v /shared/pwitte:/shared \
    -it distdl:v1.0