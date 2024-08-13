#!/bin/bash
##############################################################################
##                            Build the image                               ##
##############################################################################
docker build \
  -f training_server.Dockerfile \
  -t thesis/training-server:tf-v2.11.0 .

