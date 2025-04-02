#include <iostream>
#include <filesystem>
#include <boost/program_options.hpp>
#include <fmt/core.h>
#include <glog/logging.h>
#include "utils.h"
#include "edge_detection.h"

namespace po = boost::program_options;
namespace fs = std::filesystem;


int main(int argc, char* argv[]) {
    // Check if CUDA is available
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        LOG(ERROR) << "No CUDA devices found." << std::endl;
        return -1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int sm_count = prop.multiProcessorCount;

    // Initialize Google Logging
    google::InitGoogleLogging(argv[0]);
    FLAGS_log_dir = "./logs";
    FLAGS_colorlogtostderr = 1;
    FLAGS_alsologtostderr = 1;
    FLAGS_max_log_size = 10; // Maximum log file size in MB
    FLAGS_stop_logging_if_full_disk = 1;

    if (!fs::exists(FLAGS_log_dir)) {
        fs::create_directories(FLAGS_log_dir);
    }
    

    po::options_description desc("CUDA edge detection");
    desc.add_options()
        ("help,h", "Show help message")
        ("input,i", po::value<std::string>(), "Input image file")
        ("input-dir,d", po::value<std::string>(), "Input directory")
        ("output-dir,D", po::value<std::string>()->default_value("./outputs"), "Output directory")
        ("vertical,v", po::value<bool>()->default_value(true), "Detect vertical edges")
        ("horizontal,h", po::value<bool>()->default_value(false), "Detect horizontal edges")
        ("num-streams,s", po::value<int>(), "Number of CUDA streams to use");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    
    if (vm.count("help")) {
        LOG(INFO) << desc;
        return 0;
    }
    
    if (!vm.count("input") and !vm.count("input-dir")) {
        LOG(ERROR) << "Input file or input-dir are required.";
        return -1;
    }

    std::string inputFile;
    if (vm.count("input")) {
        inputFile = vm["input"].as<std::string>();
    }

    std::string inputDir;
    if (vm.count("input-dir")) {
        inputDir = vm["input-dir"].as<std::string>();
    }

    std::string outputDir = vm["output-dir"].as<std::string>();
    bool use_vertical = vm["vertical"].as<bool>();
    bool use_horizontal = vm["horizontal"].as<bool>();
    if (use_vertical && use_horizontal) {
        LOG(ERROR) << "Cannot use both vertical and horizontal edge detection.";
        return -1;
    }


    int num_streams = sm_count;
    if (vm.count("num-streams")) {
        num_streams = vm["num-streams"].as<int>();
    }
    

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

    LOG(INFO) << "Start processing images...";
    LOG(INFO) << "Number of images: " << images_path.size();
    LOG(INFO) << "Number of streams: " << num_streams;
    LOG(INFO) << "Output directory: " << outputDir;
    if (use_vertical) {
        LOG(INFO) << "Using vertical edge detection.";
    } else if (use_horizontal) {
        LOG(INFO) << "Using horizontal edge detection.";
    }

    process_edege_detection(images_path, outputDir, false, num_streams);
    
    return 0;
}