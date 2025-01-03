cmake_minimum_required(VERSION 3.8)
project(cuda_examples CUDA CXX)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES must be set. Example: -DCMAKE_CUDA_ARCHITECTURES=75")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")
endif()

add_subdirectory(vendor/fmt)

add_executable(device_info device_info.cu)
add_subdirectory(mnist)
add_subdirectory(gpu-puzzles)
add_subdirectory(mm)
