#pragma once

#include <iostream>
#include <cuda_runtime.h>
#include <npp.h>
#include <glog/logging.h>

inline 
void checkCuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        LOG(ERROR) << "CUDA error: " << msg << " - " << cudaGetErrorString(err);
        exit(EXIT_FAILURE);
    }
}

inline 
void checkNpp(NppStatus status, const char *msg) {
    if (status != NPP_SUCCESS) {
        LOG(ERROR) << "NPP error: " << msg;
        exit(EXIT_FAILURE);
    }
}
