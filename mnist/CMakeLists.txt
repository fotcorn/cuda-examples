enable_language(CUDA)

add_executable(mnist main.cu cppdl/mnist_utils.cpp)

target_include_directories(mnist PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mnist PRIVATE fmt::fmt)

set_target_properties(mnist PROPERTIES 
    CUDA_ARCHITECTURES "70;75;80;86"
    CUDA_SEPARABLE_COMPILATION ON)
