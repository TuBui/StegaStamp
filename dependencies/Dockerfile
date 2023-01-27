# FROM tensorflow/tensorflow:1.13.0-devel-gpu-py3
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -y && apt-get install -y \
    ca-certificates \
    git \
    wget \
    curl \
    bzip2 \
    libgtk2.0-dev \
    libopenblas-dev \
    libmagickwand-dev \
    libgl1-mesa-glx \
    sshfs \
    && rm -rf /var/lib/apt/lists/*

# sshfs
RUN echo "user_allow_other" >> /etc/fuse.conf
RUN mkdir /mnt/fast && chmod 777 /mnt/fast

# custom user
ARG NB_USER="tb0035"
ARG NB_UID="260846"
ARG NB_GID="5000092"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV NB_UID=${NB_UID} \
    NB_GID=${NB_GID} \
    HOME="/home/${NB_USER}"
RUN useradd -l -m -s /bin/bash -N -u "${NB_UID}" "${NB_USER}" && \
    mkdir -p "${HOME}" && \
    chown "${NB_USER}:${NB_GID}" "${HOME}"
RUN adduser ${NB_USER} sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER ${NB_UID}

# conda
WORKDIR "${HOME}"
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH="${HOME}"/miniconda/bin:$PATH

RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py37_4.9.2-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda install -y python==3.7 \
 && conda clean -ya

# jupyter lab
RUN conda install -y -c conda-forge jupyterlab \
    && conda clean -ya

# lmdb
RUN conda install -y lmdb python-lmdb \
    && conda clean -ya

# cupy
ENV CFLAGS=-I/usr/include
ENV LDFLAGS=-L/usr/lib/x86_64-linux-gnu/
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib:/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH

# imagenet-c
RUN git clone https://github.com/hendrycks/robustness.git && cd robustness && git checkout 3050544 && cd ImageNet-C/imagenet_c/ && pip install -e .

# pytorch
# RUN conda install -y pytorch torchvision cudatoolkit=10.0 -c pytorch && conda clean -ya

ADD ./requirements.txt ./
RUN pip install -r requirements.txt
