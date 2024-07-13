##############################################################################
##                            Run the container                             ##
##############################################################################
BASE_NAME=$(basename "$PWD")
DIR_NAME=$(dirname "$PWD")
USER_NAME=jovyan

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