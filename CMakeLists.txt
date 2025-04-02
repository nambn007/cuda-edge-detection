cmake_minimum_required(VERSION 3.10)
project(cuda_edge_detection LANGUAGES CXX CUDA)

# Enable CUDA
enable_language(CUDA)

# Set CUDA standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)
find_package(fmt REQUIRED)
find_package(CURL REQUIRED)
find_package(glog REQUIRED)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)

# Include directories
include_directories(
    include
)

link_directories(/lib/x86_64-linux-gnu/)

link_libraries(
    ${Boost_LIBRARIES}
    ${CURL_LIBRARIES}
    ${OpenCV_LIBS}
    CUDA::cufft CUDA::cublas glog::glog
)

# add executable 
add_executable(cuda_edge_detection 
    src/main.cpp
)

target_link_libraries(cuda_edge_detection 
    ${Boost_LIBRARIES}
    ${CURL_LIBRARIES}
    ${OpenCV_LIBS}
    CUDA::cudart CUDA::cufft CUDA::cublas glog::glog CUDA::nppicc CUDA::nppif
    fmt::fmt
)

set_target_properties(cuda_edge_detection PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

