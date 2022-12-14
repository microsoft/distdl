#!/bin/bash

docker run --privileged=true \
    --gpus all --sig-proxy=false --cap-add=ALL \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /home/azureuser:/workspace/home -it distdl:v2.0