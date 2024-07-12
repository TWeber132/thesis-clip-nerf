#!/bin/bash
##############################################################################
##                            Build the image                               ##
##############################################################################
docker build \
  -f inference_server.Dockerfile \
  -t gergelysoti/inference_server:2024-2-0.9 .

