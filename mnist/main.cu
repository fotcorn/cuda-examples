#include "cppdl/mnist_utils.h"
#include "cppdl/serialization.h"
#include "wmma_kernel.cuh"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char *argv[]) {
  if (argc < 2) {
    fmt::println(stderr, "Usage: {} <data_dir>\n", argv[0]);
    return 1;
  }

  // Check if device supports WMMA
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  if (prop.major < 7) {
    std::cerr << "Device needs to support CUDA compute capability 7.0 or higher\n";
    return 1;
  }

  std::string dataDir = argv[1];
  std::ifstream file(dataDir + "/mnist-20-91.82.cppdl", std::ios::binary);
  if (!file.is_open()) {
    fmt::println(stderr, "Error opening weights file: {}\n", dataDir);
    return 1;
  }

  if (!readFileHeader(file)) {
    fmt::println(stderr, "Invalid weights file format.\n");
    return 1;
  }

  // Load validation dataset
  auto rawValidationImages = loadImages(dataDir + "/t10k-images-idx3-ubyte");
  auto rawValidationLabels = loadLabels(dataDir + "/t10k-labels-idx1-ubyte");

  // Load weights
  auto l0w = deserializeTensor<float>(file).value().transpose();
  auto l0b = deserializeTensor<float>(file).value();
  auto l1w = deserializeTensor<float>(file).value();
  auto l1b = deserializeTensor<float>(file).value();
  auto l2w = deserializeTensor<float>(file).value();
  auto l2b = deserializeTensor<float>(file).value();

  fmt::println("l0w shape: {}", l0w.shape);

  // Stack first 25 images
  std::vector<Tensor<float>> first25Images(rawValidationImages.begin(), rawValidationImages.begin() + 25);
  auto stackedImages = Tensor<float>::stack(first25Images.begin(), first25Images.end());
  fmt::println("Stacked images shape: {}", stackedImages.shape);

  // Get matrix dimensions from tensor shapes
  const int M = stackedImages.shape[0];  // Number of images (25)
  const int K = stackedImages.shape[1];  // Image size (784)
  const int N = l0w.shape[1];           // Output features (16)

  fmt::println("Matrix dimensions: {}x{} * {}x{} = {}x{}", M, K, K, N, M, N);

  // Convert tensors to half precision and allocate device memory
  std::vector<half> h_images(stackedImages.size);
  std::vector<half> h_weights(l0w.size);
  for (size_t i = 0; i < stackedImages.size; i++) {
    h_images[i] = __float2half(stackedImages.data[i]);
  }
  for (size_t i = 0; i < l0w.size; i++) {
    h_weights[i] = __float2half(l0w.data[i]);
  }

  half *d_images, *d_weights;
  float *d_output;
  CUDA_CHECK(cudaMalloc(&d_images, stackedImages.size * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_weights, l0w.size * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_output, M * N * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_images, h_images.data(), h_images.size() * sizeof(half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_weights, h_weights.data(), h_weights.size() * sizeof(half), cudaMemcpyHostToDevice));

  // Launch kernel
  dim3 gridDim((M + WMMA_M - 1) / WMMA_M, (N + WMMA_N - 1) / WMMA_N);
  dim3 blockDim(32, 1);  // One warp per block

  fmt::println("Grid dimensions: {}x{}", gridDim.x, gridDim.y);
  fmt::println("Block dimensions: {}x{}", blockDim.x, blockDim.y);

  wmma_matrix_multiply<<<gridDim, blockDim>>>(d_images, d_weights, d_output, M, K, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy result back to host
  std::vector<float> h_output(M * N);
  CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost));

  // Print first few elements of result
  fmt::println("First few elements of result:");
  for (int i = 0; i < 5; ++i) {
    fmt::print("{} ", h_output[i]);
  }
  fmt::println("");

  // Cleanup
  CUDA_CHECK(cudaFree(d_images));
  CUDA_CHECK(cudaFree(d_weights));
  CUDA_CHECK(cudaFree(d_output));

  return 0;
}
