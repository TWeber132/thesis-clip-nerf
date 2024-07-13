#!/bin/bash
##############################################################################
##                            Build the image                               ##
##############################################################################
USER_ID=$(id -u)
USER_GID=$(id -g)
BASE_NAME=$(basename "$PWD")

docker build \
  --build-arg UID=$USER_ID \
  --build-arg GID=$USER_GID \
  -f dev.Dockerfile \
  -t thesis/$BASE_NAME .


