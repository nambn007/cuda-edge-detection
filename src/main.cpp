#include <iostream>
#include <cuda_runtime.h>
#include <npp.h>
#include <filesystem>
#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>

namespace po = boost::program_options;
namespace fs = std::filesystem;

const int NUM_STREAMS = 32;

void checkCuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void process_edege_detection(const std::vector<std::string> &images_path, const std::string &outputDir) {
    const int numImages = images_path.size();
    if (numImages == 0) {
        std::cerr << "No images to process." << std::endl;
        return;
    }

    const int batchSize = NUM_STREAMS;
    std::vector<cudaStream_t> streams(batchSize);

    for (int i = 0; i < batchSize; ++i) {
        checkCuda(cudaStreamCreate(&streams[i]), "Creating CUDA stream");
    }

    for (int b = 0; b < numImages; b += batchSize) {
        int actualBatch = std::min(batchSize, numImages - b);

        std::vector<Npp8u*> d_src(actualBatch), d_gray(actualBatch), d_dst(actualBatch);
        std::vector<Npp8u*> h_out(actualBatch);
        NppiSize roi;

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
            NppStatus status = nppiRGBToGray_8u_AC4C1R_Ctx(
                d_rgb, step, d_gray[i], grayStep, size, streamCtx
            );
            

        }
    }

    for (auto &stream : streams) cudaStreamDestroy(stream);
}
    

int main(int argc, char* argv[]) {
    // Check if CUDA is available
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        return -1;
    }

    po::options_description desc("CUDA edge detection");
    desc.add_options()
        ("help,h", "Show help message")
        ("input,i", po::value<std::string>(), "Input image file")
        ("input-dir,d", po::value<std::string>(), "Input directory")
        ("output-dir,D", po::value<std::string>()->default_value("./outputs"), "Output directory");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    
    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }
    
    if (!vm.count("input") || !vm.count("input-dir")) {
        std::cerr << "Input file or input-dir are required." << std::endl;
        return -1;
    }

    std::string inputFile = vm["input"].as<std::string>();
    std::string inputDir = vm["input-dir"].as<std::string>();
    std::string outputDir = vm["output-dir"].as<std::string>();

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int sm_count = prop.multiProcessorCount;
    std::cout << "SM count: " << sm_count << std::endl;

    std::vector<std::string> images_path;

    if (fs::exists(inputFile)) {
        images_path.push_back(inputFile);    
    }

    if (fs::exists(inputDir)) {
        for (const auto& entry : fs::directory_iterator(inputDir)) {
            if (entry.is_regular_file()) {
                images_path.push_back(entry.path().string());
            }
        }
    }

    if (!fs::exists(outputDir)) {
        fs::create_directories(outputDir);
    }

    process_edege_detection(images_path, outputDir);
    
    return 0;
}