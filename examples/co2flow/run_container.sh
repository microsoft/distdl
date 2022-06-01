#!/bin/bash

docker run --gpus all -v /home/pwitte:/workspace/home -e SLEIPNER_CREDENTIALS=$SLEIPNER_CREDENTIALS -it distdl:v1.0