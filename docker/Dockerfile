FROM nvcr.io/nvidia/tensorflow:20.02-tf2-py3

# Install apt packages
RUN apt-get update && apt-get install -y --no-install-recommends \
vim build-essential cmake autoconf automake autopoint libglib2.0-dev libtool libsm6 libxrender-dev libxext6 && apt-get clean && rm -fr /var/lib/apt/lists/*

# Install keras_retinanet
COPY . /keras_retinanet/.
WORKDIR /keras_retinanet
RUN python setup.py develop
