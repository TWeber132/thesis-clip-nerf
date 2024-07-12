#!/bin/bash
##############################################################################
##                            Run the container                             ##
##############################################################################
DATA_CONTAINER=/home/jovyan/data
DATA_HOST=/home/iras-admin/project_data/2024-2/kiro2024

docker run \
  --name tf \
  --rm \
  -it \
  --net=host \
  -v "$DATA_HOST":"$DATA_CONTAINER":rw \
  -e DISPLAY="$DISPLAY" \
  --gpus all \
  gergelysoti/inference_server:2024-2-0.9
