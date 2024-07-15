#!/bin/bash
##############################################################################
##                            Build the image                               ##
##############################################################################
docker build \
  -f kubeflow.Dockerfile \
  -t thesis/kubeflow-jupyter-tensorflow-full:v1.9.0-rc.2 .

