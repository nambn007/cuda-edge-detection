# Use NVIDIA's CUDA base image
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install required packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libboost-all-dev \
    libopencv-dev \
    libfftw3-dev \
    libcurl4-openssl-dev \
    libgoogle-glog-dev \
    libfmt-dev \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app
