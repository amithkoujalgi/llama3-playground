#!/bin/bash

docker run \
  --gpus=all \
  -it \
  -p 8883:8070 \
  -p 8884:9001 \
  -p 8885:8885 \
  -p 8886:8886 \
  -p 8887:8887 \
  -p 8888:8888 \
  ~/llama3-playground-data:/app/data \
  llama3-finetuning:0.1


#docker run \
#  --gpus=all \
#  -it \
#  -p 8883:8070 \
#  -p 8884:9001 \
#  -p 8885:8885 \
#  -p 8886:8886 \
#  -p 8887:8887 \
#  -p 8888:8888 \
#  llama3-finetuning:0.1