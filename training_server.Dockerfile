##############################################################################
##                                 Base Image                               ##
##############################################################################
FROM kubeflownotebookswg/jupyter:v1.6.0-rc.0 AS kubeflow-base

##############################################################################
##                          jupyter-tensorflow-cuda                         ##
##############################################################################
FROM kubeflow-base AS jupyter-tensorflow-cuda
# SEE https://github.com/kubeflow/kubeflow/blob/v1.6.0-rc.0/components/example-notebook-servers/jupyter-tensorflow/cuda.Dockerfile

USER root

ARG TENSORFLOW_VERSION=2.11.0
# needed for LIBNVINFER
ARG OLD_CUDA_VERSION=11.1
# args - software versions
ARG CUDA_VERSION=11.2
ARG CUDA_COMPAT_VERSION=460.73.01-1
ARG CUDA_CUDART_VERSION=11.2.152-1
ARG CUDNN_VERSION=8.1.0.77-1
ARG LIBNVINFER_VERSION=7.2.3-1

# we need bash's env var character substitution
SHELL ["/bin/bash", "-c"]

# install - cuda
# for `cuda-compat-*`: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
RUN curl -sL "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub" | apt-key add - \
 && echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" > /etc/apt/sources.list.d/cuda.list \
 && apt-get -yq update \
 && apt-get -yq install --no-install-recommends \
    cuda-compat-${CUDA_VERSION/./-}=${CUDA_COMPAT_VERSION} \
    cuda-cudart-${CUDA_VERSION/./-}=${CUDA_CUDART_VERSION} \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* \
 && ln -s /usr/local/cuda-${CUDA_VERSION} /usr/local/cuda

# envs - cuda
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=${CUDA_VERSION}"

# install - other nvidia stuff
RUN curl -sL "https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub" | apt-key add - \
 && echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list \
 && echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list \
 && apt-get -yq update \
 && apt-get -yq install --no-install-recommends \
    cm-super \
    cuda-command-line-tools-${CUDA_VERSION/./-} \
    cuda-nvrtc-${CUDA_VERSION/./-} \
    libcublas-${CUDA_VERSION/./-} \
    libcudnn8=${CUDNN_VERSION}+cuda${CUDA_VERSION} \
    libcufft-${CUDA_VERSION/./-} \
    libcurand-${CUDA_VERSION/./-} \
    libcusolver-${CUDA_VERSION/./-} \
    libcusparse-${CUDA_VERSION/./-} \
    libfreetype6-dev \
    libhdf5-serial-dev \
    libnvinfer7=${LIBNVINFER_VERSION}+cuda${OLD_CUDA_VERSION} \
    libnvinfer-plugin7=${LIBNVINFER_VERSION}+cuda${OLD_CUDA_VERSION} \
    libzmq3-dev \
    pkg-config \
    # can't be used until NVIDIA updates (requires python < 3.7)
    # python3-libnvinfer=${LIBNVINFER_VERSION}+cuda${CUDA_VERSION} \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

 # tensorflow fix - CUDA profiling, tensorflow requires CUPTI
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# tensorflow fix - wrong libcuda lib path (+ reconfigure dynamic linker run-time bindings)
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
 && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf \
 && ldconfig

# tensorflow fix - wrong libcusolver lib path
# https://github.com/tensorflow/tensorflow/issues/43947#issuecomment-748273679
RUN ln -s /usr/local/cuda-${CUDA_VERSION}/targets/x86_64-linux/lib/libcusolver.so.11 /usr/local/cuda-${CUDA_VERSION}/targets/x86_64-linux/lib/libcusolver.so.10

# tensorflow fix - some tensorflow tools expect a `python` binary
RUN ln -s $(which python3) /usr/local/bin/python

USER $NB_UID

RUN python3 -m pip install --quiet --no-cache-dir tensorflow-gpu==${TENSORFLOW_VERSION}


##############################################################################
##                          jupyter-tensorflow-cuda                         ##
##############################################################################
FROM jupyter-tensorflow-cuda AS jupyter-tensorflow-cuda-full

RUN pip install --no-cache-dir \
                kfp==1.6.3 \
                kfp-server-api==1.6.0 \
                kfserving==0.5.1 \
                bokeh==2.3.2 \
                cloudpickle==1.6.0 \
                dill==0.3.4 \
                ipympl==0.7.0 \
                ipywidgets==7.6.3 \
                jupyterlab-git==0.30.1 \
                matplotlib==3.4.2 \
                pandas==1.2.4 \
                scikit-image==0.18.1 \
                scikit-learn==0.24.2 \
                scipy==1.7.0 \
                seaborn==0.11.1 \
                xgboost==1.4.2 \
                kfp==1.6.3 \
                kfp-server-api==1.6.0 \
                kfserving==0.5.1

##############################################################################
##                                 Dependencies                             ##
##############################################################################
FROM jupyter-tensorflow-cuda-full AS tf-dependencies

USER root
RUN DEBIAN_FRONTEND=noninteractive \
	apt-get update && \
	apt install -y mesa-utils libgl1-mesa-glx libglu1-mesa-dev freeglut3-dev mesa-common-dev libopencv-dev python3-opencv python3-tk
RUN apt-get update && apt-get install -y screen git

USER $NB_USER
RUN pip install --no-cache-dir opencv-contrib-python
RUN pip install --no-cache-dir transforms3d tensorflow_addons
RUN pip install --no-cache-dir scipy numpy
RUN pip install --no-cache-dir scikit-learn einops
RUN pip install --upgrade tensorflow-probability
RUN pip install --no-cache-dir wandb pandas
RUN pip install --no-cache-dir imageio
RUN pip install --no-cache-dir msgpack colortrans
RUN pip install --no-cache-dir fastapi uvicorn
RUN pip install --no-cache-dir matplotlib
RUN pip install --upgrade hydra-core 
RUN pip install --no-cache-dir tensorflow-graphics --no-deps
RUN pip install --no-cache-dir loguru
RUN pip install --no-cache-dir ftfy regex

##############################################################################
##                                 Manipulation Tasks                       ##
##############################################################################
FROM tf-dependencies AS tf-manipulation-tasks

COPY --chown=$NB_UID:$NB_UID ./dependencies /opt/dependencies
RUN cd /opt/dependencies/manipulation_tasks && \
    pip install .
