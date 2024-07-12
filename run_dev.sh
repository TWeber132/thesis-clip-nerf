#!/bin/bash
##############################################################################
##                            Build the image                               ##
##############################################################################
USER_ID=$(id -u)
USER_GID=$(id -g)
BASE_NAME=$(basename "$PWD")
DIR_NAME=$(dirname "$PWD")

docker build \
  --build-arg UID=$USER_ID \
  --build-arg GID=$USER_GID \
  -f dev.Dockerfile \
  -t thesis/$BASE_NAME .

##############################################################################
##                            Run the container                             ##
##############################################################################
docker run \
  --name tf \
  --rm \
  -it \
  --net=host \
  -v $DIR_NAME/$BASE_NAME/src:/home/$USER_NAME/workspace/src:rw \
  -v $DIR_NAME/shared_docker_volume:/home/$USER_NAME/data:rw \
  -e DISPLAY="$DISPLAY" \
  --gpus all \
  thesis/$BASE_NAME
