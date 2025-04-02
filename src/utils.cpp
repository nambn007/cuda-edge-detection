#include "utils.h"
#include <fstream>
#include <sstream>
#include <numeric>
#include <cstring>

// PlanarImage implementation
PlanarImage::PlanarImage() : height(0), width(0), channel(0) {}

PlanarImage::PlanarImage(int height, int width, int channels) 
    : height(height), width(width), channel(channels) {
    data.resize(height * width * channels);
}

int PlanarImage::load(const std::string &img_path, bool is_rgb) {
    cv::Mat img;
    if (!is_rgb) {
        img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
    } else {
        img = cv::imread(img_path, cv::IMREAD_COLOR);
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    }

    if (img.empty()) {
        LOG(ERROR) << "Cannot read image: " << img_path;
        return -1;
    }
    return load(img);
}

int PlanarImage::load(const cv::Mat &img) {
    height = img.rows;
    width = img.cols;
    channel = img.channels();
    data.resize(height * width * channel);

    for (int row = 0; row < img.rows; row++) {
        for (int col = 0; col < img.cols; col++) {
            if (channel == 1) {
                uchar pixel = img.at<uchar>(row, col);
                data[row * width + col] = float(pixel);
            } else {
                cv::Vec3b pixel = img.at<cv::Vec3b>(row, col);
                for (int c = 0; c < channel; ++c) {
                    data[c * height * width + row * width + col] = float(pixel[c]);
                }
            }
        }
    }
    return 0;
}

bool PlanarImage::empty() const {
    return data.empty();
}

cv::Mat PlanarImage::toImage() const {
    auto flag = (channel == 1) ? CV_8UC1 : CV_8UC3;
    cv::Mat img(height, width, flag);

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            if (channel == 1) {
                img.at<uchar>(row, col) = static_cast<uchar>(
                    std::min(std::max(data[row * width + col], 0.0f), 255.0f));
            } else {
                uchar r = static_cast<uchar>(
                    std::min(std::max(data[0 * height * width + row * width + col], 0.0f), 255.0f));
                uchar g = static_cast<uchar>(
                    std::min(std::max(data[1 * height * width + row * width + col], 0.0f), 255.0f));
                uchar b = static_cast<uchar>(
                    std::min(std::max(data[2 * height * width + row * width + col], 0.0f), 255.0f));
                img.at<cv::Vec3b>(row, col) = cv::Vec3b(b, g, r);
            }
        }
    }
    return img;
}

const float* PlanarImage::getData() const {
    return data.data();
}

const float* PlanarImage::getData(int channelID) const {
    if (channelID < 0 || channelID >= channel) {
        LOG(ERROR) << "Invalid channel ID: " << channelID;
        return nullptr;
    }
    return data.data() + channelID * height * width;
}

// Kernel implementation
bool Kernel::empty() {
    return data.empty();
}

int Kernel::load(const std::string &file_name) {
    std::ifstream file(file_name);
    if (!file.is_open()) {
        std::cerr << "Cannot open kernel file: " << file_name << "\n";
        return -1;
    }

    std::vector<std::vector<float>> m_data;
    std::string line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<float> row;
        while (std::getline(ss, value, ',')) {
            row.push_back(std::stof(value));
        }
        m_data.push_back(row);
    }

    height = m_data.size();
    width = m_data[0].size();
    data.resize(height * width);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            data[i * width + j] = m_data[i][j];
        }
    }
    return 0;
}

void Kernel::normalize() {
    float sum = std::accumulate(data.begin(), data.end(), 0.0f);
    for (float &val : data) {
        val /= sum;
    }
}

// Utility functions
cv::Mat resizeWithAspectRatio(const cv::Mat &img, int target_width, int target_height) {
    cv::Mat resized;
    float aspect_ratio = (float)img.cols / img.rows;
    int new_width, new_height;

    if (aspect_ratio > 1) { 
        new_width = target_width;
        new_height = target_width / aspect_ratio;
    } else { 
        new_height = target_height;
        new_width = target_height * aspect_ratio;
    }

    cv::resize(img, resized, cv::Size(new_width, new_height));

    cv::Mat output = cv::Mat::zeros(target_height, target_width, img.type());

    int x_offset = (target_width - new_width) / 2;
    int y_offset = (target_height - new_height) / 2;
    resized.copyTo(output(cv::Rect(x_offset, y_offset, new_width, new_height)));

    return output;
}


PlanarImage resize_image(const PlanarImage &image, int outH, int outW) {
    cv::Mat mat_image = image.toImage();
    cv::Mat mat_resize_image = resizeWithAspectRatio(mat_image, outW, outH);
    
    PlanarImage resized_image(mat_resize_image.rows, mat_resize_image.cols, mat_resize_image.channels());
    resized_image.load(mat_resize_image);
    return resized_image;
}


PlanarImage pad_image(const PlanarImage &image, int padded_h, int padded_w) {
    PlanarImage padded_image(padded_h, padded_w, image.channel);
    for (int c = 0; c < image.channel; ++c) {
        for (int r = 0; r < image.height; ++r) {
            float *out_data = padded_image.data.data() + c * padded_h * padded_w;
            memcpy(out_data + r * padded_w, 
                   image.getData(c) + r * image.width,
                   image.width * sizeof(float));
        }
    }
    return padded_image;
}

PlanarImage crop(const PlanarImage &image, int outH, int outW) {
    int offsetH = (image.height - outH) / 2;
    int offsetW = (image.width - outW) / 2;
    PlanarImage cropped_image(outH, outW, image.channel);
    for (int c = 0; c < image.channel; ++c) {
        for (int r = 0; r < outH; ++r) {
            float *out_data = cropped_image.data.data() + c * outH * outW;
            memcpy(out_data + r * outW, 
                   image.getData(c) + (r + offsetH) * image.width + offsetW,
                   outW * sizeof(float));
        }
    }
    return cropped_image;
}

Kernel pad_kernel(const Kernel &kernel, int padded_h, int padded_w) {
    Kernel padded_kernel;
    padded_kernel.height = padded_h;
    padded_kernel.width = padded_w;
    padded_kernel.data.resize(padded_h * padded_w);
    for (int r = 0; r < kernel.height; ++r) {
        memcpy(padded_kernel.data.data() + r * padded_w, 
               kernel.data.data() + r * kernel.width,
               kernel.width * sizeof(float));
    }
    return padded_kernel;
}