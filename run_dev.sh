#!/bin/bash
##############################################################################
##                            Build the image                               ##
##############################################################################
UID=$(eval "id -u")
GID=$(eval "id -g")
BASE_NAME=$(basename "$PWD")

docker build \
  --build-arg UID="$UID" \
  --build-arg GID="$GID" \
  -f dev.Dockerfile \
  -t thesis/$BASE_NAME .

##############################################################################
##                            Run the container                             ##
##############################################################################
DIR_NAME=$(dirname "$PWD")
SRC_CONTAINER=/home/jovyan/workspace/src
SRC_HOST="$(pwd)"/src

DATA_CONTAINER=/home/jovyan/data
DATA_HOST=$DIR_NAME/shared_docker_volume

docker run \
  --name tf \
  --rm \
  -it \
  --net=host \
  -v "$SRC_HOST":"$SRC_CONTAINER":rw \
  -v "$DATA_HOST":"$DATA_CONTAINER":rw \
  -e DISPLAY="$DISPLAY" \
  --gpus all \
  thesis/$BASE_NAME
