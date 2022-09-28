#!/bin/bash

docker run --privileged=true --gpus all -v /home/pwitte:/workspace/home -it distdl:v1.1

