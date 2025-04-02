# FFT Convolution 2D

This project implements FilterSobel for detection edge in individual images or directories containing multiple images.

## Performance

The project has been tested on an NVIDIA RTX 3060 GPU with Driver Version: 550.120 with the following results:

- **CUDA FFT Convolution**: Average time for 100 iterations: 1.13191 ms
- **CPU FFT Convolution**: Average time for 100 iterations: 23.1025 ms

Thus, the CUDA version is approximately 20 times faster than the CPU version.

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

The project provides two main executables:

### 1. benchmark_fft_convolution

Used to evaluate the performance of both methods (CUDA and CPU).

```bash
./benchmark_fft_convolution <path_image> <kernel_file>
```

Example:
```bash
./benchmark_fft_convolution ../data/cat.jpg ../benchmarks/kernel1.csv
```

### 2. fft_convolution

Used to apply a kernel to an image or set of images.

```bash
Usage: ./fft_convolution parameters
  --help                    Print help message
  --cuda arg (=1)           Use CUDA (1=enabled, 0=disabled)
  --image arg               Path to the image (process a single image)
  --folder arg              Path to folder of images (process multiple images)
  --kernel arg              Path to the kernel file
  --output arg (=./outputs) Path to output folder
```

### Usage Examples:

Process a single image with a Gaussian blur kernel:
```bash
./fft_convolution -k ../kernels/gaussian_blur_15.csv -f ../data/cat.jpg
# Output image will be saved to ./outputs/cat.jpg
```

Process multiple images from a directory using CUDA:
```bash
./fft_convolution -k ../kernels/gaussian_blur_15.csv -d ../data/input_100images -o ../data/output_cuda_100images
```

Process multiple images from a directory using CPU:
```bash
./fft_convolution -k ../kernels/gaussian_blur_15.csv -d ../data/input_100images -o ../data/output_cpu_100images -c false
```

## Project Structure

- **benchmark.cpp**: Performance evaluation program
- **main.cpp**: Main image processing program
- **fft.h/fft.cuh**: Function declarations and definitions for FFT Convolution
- **utils.h**: Image processing utilities

## Kernels

The project comes with some predefined kernels in the `kernels/` directory:
- **gaussian_blur_15.csv**: Gaussian blur kernel of size 15x15

## Notes

- All input images are resized to 640x640 before processing
- Applied kernels are automatically normalized
- Logs are stored in the `./logs` directory
