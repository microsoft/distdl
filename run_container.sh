#!/bin/bash

docker run --privileged=true --gpus all -v /home/azureuser:/workspace/home -it distdl:v2.0

