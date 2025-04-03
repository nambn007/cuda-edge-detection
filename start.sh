#!/bin/bash
docker run -it --rm --device /dev/dri --privileged --gpus all \
    -v $(pwd):/app \
    --env NVIDIA_DISABLE_REQUIRE=1 \
    cuda-edge-detection:latest \
    bash