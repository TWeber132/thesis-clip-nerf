#!/bin/bash
##############################################################################
##                            Build the image                               ##
##############################################################################
docker build \
  -f kubeflow_v160rc0.Dockerfile \
  -t thesis/kubeflow-jupyter-tensorflow-full:v1.6.0-rc.0 .

