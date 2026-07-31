# Dockerfile may have following Arguments:
# tag - tag for the Base image, (e.g. 2.9.1 for tensorflow)
# branch - user repository branch to clone (default: master, another option: test)
#
# To build the image:
# $ docker build -t <dockerhub_user>/<dockerhub_repo> --build-arg arg=value .
# or using default args:
# $ docker build -t <dockerhub_user>/<dockerhub_repo> .

ARG tag=2.9.1-cuda12.6-cudnn9-runtime

FROM pytorch/pytorch:${tag}

LABEL maintainer='Jesus Soriano-Gonzalez, Josep Oliver-Sanso'
LABEL version='1.0.0'

# Install Ubuntu packages
# - gcc is needed in Pytorch images because deepaas installation might break otherwise
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        git \
        libgl1-mesa-glx \
        curl \
        libglib2.0-0\
    && rm -rf /var/lib/apt/lists/*

# Set LANG environment
ENV LANG=C.UTF-8

# Set the working directory
WORKDIR /srv/socib-beach-wracks-identification

# Copy local repository files to the container
COPY . .

# Install user app
RUN pip3 install --no-cache-dir -e .

# Open ports (deepaas, monitoring, ide)
EXPOSE 5000

# Launch deepaas
CMD ["deepaas-run", "--listen-ip", "0.0.0.0", "--listen-port", "5000", "--config-file", "/srv/socib-beach-wracks-identification/deepaas.conf"]
