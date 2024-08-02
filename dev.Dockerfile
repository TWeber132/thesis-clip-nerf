##############################################################################
##                                 Base Image                               ##
##############################################################################
ARG RENDER=base
FROM tensorflow/tensorflow:2.11.0-gpu AS tf-base
USER root
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

##############################################################################
##                                 Dependencies                             ##
##############################################################################
FROM tf-base AS tf-dependencies
USER root
RUN apt update \
  && apt install -y -qq --no-install-recommends \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libxext6 \
    libx11-6 \
  && rm -rf /var/lib/apt/lists/*# Env vars for the nvidia-container-runtime.

RUN DEBIAN_FRONTEND=noninteractive \
	apt update && \
	apt install -y mesa-utils libgl1-mesa-glx libglu1-mesa-dev freeglut3-dev mesa-common-dev libopencv-dev python3-opencv python3-tk
RUN /usr/bin/python3 -m pip install --upgrade pip
RUN /usr/bin/python3 -m pip install --no-cache-dir opencv-contrib-python
RUN /usr/bin/python3 -m pip install --no-cache-dir transforms3d tensorflow_addons
RUN /usr/bin/python3 -m pip install --no-cache-dir scipy numpy
RUN /usr/bin/python3 -m pip install --no-cache-dir scikit-learn einops
RUN /usr/bin/python3 -m pip install --upgrade tensorflow-probability
RUN /usr/bin/python3 -m pip install --no-cache-dir wandb pandas
RUN /usr/bin/python3 -m pip install --no-cache-dir imageio
RUN /usr/bin/python3 -m pip install --no-cache-dir msgpack colortrans
RUN /usr/bin/python3 -m pip install --no-cache-dir fastapi uvicorn
RUN /usr/bin/python3 -m pip install --no-cache-dir tensorflow-graphics
RUN /usr/bin/python3 -m pip install --no-cache-dir ftfy regex

# USER root
# RUN add-apt-repository --remove ppa:vikoadi/ppa
# RUN rm /etc/apt/sources.list.d/cuda.list
USER $USER

##############################################################################
##                                  User                                    ##
##############################################################################
FROM tf-dependencies AS tf-user
#FROM base-render as user

# install sudo
RUN apt-get update && apt-get install -y sudo

# Create user
ARG USER=jovyan
ARG PASSWORD=automaton
ARG UID=1000
ARG GID=1000
ENV USER=$USER
RUN groupadd -g $GID $USER \
    && useradd -m -u $UID -g $GID -p "$(openssl passwd -1 $PASSWORD)" \
    --shell $(which bash) $USER -G sudo
RUN echo "%sudo ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/sudogrp

USER $USER
RUN mkdir -p /home/$USER/workspace/src
RUN mkdir -p /home/$USER/data
WORKDIR /home/$USER/workspace
CMD ["bash"]

##############################################################################
##                                 Manipulation Tasks                       ##
##############################################################################
FROM tf-user AS tf-manipulation-tasks

COPY --chown=$USER:$USER ./dependencies /home/$USER/workspace/dependencies
RUN cd /home/$USER/workspace/dependencies/manipulation_tasks && \
    pip install .

RUN pip install --no-cache-dir loguru
RUN pip install --no-cache-dir matplotlib
RUN pip install hydra-core --upgrade

##############################################################################
##                                 Simulation                               ##
##############################################################################
FROM tf-manipulation-tasks AS tf-gym

RUN pip install --no-cache-dir gym==0.17.3
RUN pip install --no-cache-dir pybullet>=3.0.4
RUN pip install --no-cache-dir tdqm
RUN pip install --no-cache-dir transformers==4.3.2
RUN pip install --no-cache-dir imageio-ffmpeg
# RUN pip install --no-cache-dir meshcat>=0.0.18
# RUN pip install --no-cache-dir kornia==0.5.11