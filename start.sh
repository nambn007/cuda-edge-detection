docker run -it --rm --gpus=all \
    -v $(pwd):/app \
    cuda-fft-convolution:latest \
    bash