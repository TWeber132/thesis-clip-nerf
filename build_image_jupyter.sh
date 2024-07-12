#!/bin/bash
##############################################################################
##                            Build the image                               ##
##############################################################################
uid=$(eval "id -u")
gid=$(eval "id -g")
docker build \
  --build-arg UID="$uid" \
  --build-arg GID="$gid" \
  -f jupyter.Dockerfile \
  -t gergelysoti/tf-jupyter-nerf:2024-2-0.1 .

