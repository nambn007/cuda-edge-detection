#pragma once

#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include <vector>
#include <string>

// Non-Interleaved Image
struct PlanarImage {
    std::vector<float> data;
    int height;
    int width;
    int channel;

    PlanarImage();
    PlanarImage(int height, int width, int channels);

    int load(const std::string &img_path, bool is_rgb = false);
    int load(const cv::Mat &img);
    bool empty() const;
    cv::Mat toImage() const;
    const float* getData() const;
    const float* getData(int channelID) const;
};

// Kernel for convolution
struct Kernel {
    std::vector<float> data;
    int height;
    int width;

    bool empty();
    int load(const std::string &file_name);
    void normalize();
};

// Utility functions
PlanarImage resize_image(const PlanarImage &image, int outH, int outW);
PlanarImage pad_image(const PlanarImage &image, int padded_h, int padded_w);
PlanarImage crop(const PlanarImage &image, int outH, int outW);
Kernel pad_kernel(const Kernel &kernel, int padded_h, int padded_w);