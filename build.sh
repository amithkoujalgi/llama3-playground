#!/bin/bash

docker image prune -f
docker build -t llama3-playground:0.1 .