##############################################################################
##                                 Base Image                               ##
##############################################################################
FROM kubeflownotebookswg/jupyter-tensorflow-cuda-full:v1.9.0-rc.2 as kubeflow-base
##############################################################################
##                                 Dependencies                             ##
##############################################################################
FROM kubeflow-base as tf-dependencies

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
FROM tf-dependencies as tf-manipulation-tasks

COPY --chown=$NB_UID:$NB_UID ./dependencies /opt/dependencies
RUN cd /opt/dependencies/manipulation_tasks && \
    pip install .
