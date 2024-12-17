add_executable(mnist main.cpp cppdl/mnist_utils.cpp)

target_include_directories(mnist PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mnist PRIVATE fmt::fmt)
