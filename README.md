# CUDA Edge Detection (Sobel Filter)

This project implements SobelFilter for detecting edge in individual images or directories containing multiple images.

## System Requirements

- NVIDIA GPU with CUDA support
- NVIDIA Driver (tested with version 550.120)
- Docker
- nvidia-container-toolkit

## Installation and Building

### Using Docker (Recommended)

1. Run the script to build the Docker image:
```bash
./build_docker.sh
```

2. Start the container:
```bash
./start.sh
```

3. Inside the container, build the project:
```bash
mkdir build && cd build
cmake .. && make -j11
```

### Manual Installation

If you're not using Docker, you need to install the following dependencies:
- build-essential
- cmake
- git
- libboost-all-dev
- libopencv-dev
- libfftw3-dev
- libcurl4-openssl-dev
- libgoogle-glog-dev
- libfmt-dev
- CUDA Toolkit

## Usage

The project provides:

### 2. Edge Detection

Used to detect edge on an image or set of images (Vertical or Horizontal)

```bash
Usage: ./cuda_edge_detection
  -h [ --help ]                        Show help message
  -i [ --input ] arg                   Input image file
  -d [ --input-dir ] arg               Input directory
  -D [ --output-dir ] arg (=./outputs) Output directory
  -v [ --vertical ] arg (=0)           Detect vertical edges
  -h [ --horizontal ] arg (=0)         Detect horizontal edges
  -s [ --num-streams ] arg             Number of CUDA streams to use
```

### Usage Examples:

Process a single image with vertical sobel filter:
```bash
./cuda_edge_detection --input ../data/cat.jpg --output-dir ./outputs --vertical true
# Output image will be saved to ./outputs/cat.jpg
```

Process a single image with horizontal sobel filter:
```bash
./cuda_edge_detection --input ../data/cat.jpg --output-dir ./outputs --horizontal true
# Output image will be saved to ./outputs/cat.jpg
```


Process multiple images from a directory using CUDA:
```bash
./cuda_edge_detection --input-dir ../data/input_100images --output-dir ../data/output_horizontal  --horizontal true
```

## Project Structure

- **edge_detection.h**: Define process edege detection
- **main.cpp**: Main image processing program
- **utils.h**: logging and check status cuda

