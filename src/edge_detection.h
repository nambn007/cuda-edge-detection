#pragma once 

#include <fmt/core.h>
#include <filesystem>
#include <glog/logging.h>
#include <vector>
#include <cuda_runtime.h>
#include <npp.h>
#include <opencv2/opencv.hpp>
#include "utils.h"

namespace fs = std::filesystem;

void process_edege_detection(const std::vector<std::string> &images_path, 
                             const std::string &outputDir,
                             bool detect_vertical = true,
                             int num_streams = 16) 
{
    auto start = std::chrono::high_resolution_clock::now();

    const int numImages = images_path.size();
    if (numImages == 0) {
        std::cerr << "No images to process." << std::endl;
        return;
    }

    const int batchSize = num_streams;
    std::vector<cudaStream_t> streams(batchSize);

    LOG(INFO) << "Use batch-size " << batchSize << " for processing images.";

    for (int i = 0; i < batchSize; ++i) {
        checkCuda(cudaStreamCreate(&streams[i]), "Creating CUDA stream");
    }

    for (int b = 0; b < numImages; b += batchSize) {
        int actualBatch = std::min(batchSize, numImages - b);

        std::vector<Npp8u*> d_src(actualBatch), d_gray(actualBatch), d_dst(actualBatch);
        std::vector<Npp8u*> h_out(actualBatch);
        NppiSize roi;
        std::cout << "Start processing batch " << b << " to " << b + actualBatch - 1 << std::endl;

        for (int i = 0; i < actualBatch; ++i) {
            cv::Mat img = cv::imread(images_path[b + i]);
            if (img.empty()) continue;

            cv::Mat imgRGB;
            cv::cvtColor(img, imgRGB, cv::COLOR_BGR2RGB);
            
            roi.width = imgRGB.cols;
            roi.height = imgRGB.rows;

            size_t numPixels = roi.width * roi.height;
            size_t imgSize = numPixels * 3;
            size_t graySize = numPixels;

            Npp8u *d_rgb;
            checkCuda(cudaMalloc(&d_rgb, imgSize), "Malloc RGB");
            checkCuda(cudaMemcpyAsync(d_rgb, imgRGB.data, imgSize, cudaMemcpyHostToDevice, streams[i]), "Copy RGB to device");
            
            checkCuda(cudaMalloc(&d_gray[i], graySize), "Malloc Gray");
            checkCuda(cudaMalloc(&d_dst[i], graySize), "Malloc Edge");

            NppiSize size = {roi.width, roi.height};
            int step = roi.width * 3;
            int grayStep = roi.width;

            NppStreamContext streamCtx;
            streamCtx.hStream = streams[i];
            checkNpp(nppiRGBToGray_8u_C3C1R_Ctx(
                d_rgb, step, d_gray[i], grayStep, size, streamCtx
            ), "Convert RGB to Gray");

            if (detect_vertical) {
                checkNpp(nppiFilterSobelVertBorder_8u_C1R_Ctx(
                    d_gray[i], grayStep, roi, NppiPoint{0, 0},
                    d_dst[i], grayStep, roi, NppiBorderType::NPP_BORDER_REPLICATE, streamCtx
                ), "Sobel Filter Vert");
            } else {
                checkNpp(nppiFilterSobelHorizBorder_8u_C1R_Ctx(
                    d_gray[i], grayStep, roi, NppiPoint{0, 0},
                    d_dst[i], grayStep, roi, NppiBorderType::NPP_BORDER_REPLICATE, streamCtx
                ), "Sobel Filter Horiz");
            }

            h_out[i] = new Npp8u[graySize];
            checkCuda(cudaMemcpyAsync(h_out[i], d_dst[i], graySize, cudaMemcpyDeviceToHost, streams[i]), "Copy Gray to host");     
            
            cudaFreeAsync(d_rgb, streams[i]);
        }

        for (int i = 0; i < actualBatch; ++i) {
            cudaStreamSynchronize(streams[i]);
            std::cout << "Processing image: " << images_path[b + i] << std::endl;
            std::cout << "Output size: " << roi.width << "x" << roi.height << std::endl;
            cv::Mat outputImg(roi.height, roi.width, CV_8UC1, h_out[i]);
            std::string output_path = fmt::format("{}/{}", outputDir, fs::path(images_path[b + i]).filename().string());
            cv::imwrite(output_path, outputImg);

            delete[] h_out[i];
            cudaFree(d_gray[i]);
            cudaFree(d_dst[i]);
            std::cout << "Done processing image: " << images_path[b + i] << std::endl;
        }
    }

    for (auto &stream : streams) cudaStreamDestroy(stream);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Processing completed in " << elapsed.count() << " ms." << std::endl;
}
    