#include "cppdl/mnist_utils.h"
#include "cppdl/serialization.h"
#include "linear_layer.cuh"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <utility>

// Helper function to upload weights and bias to GPU
std::pair<half*, float*> uploadWeightsAndBias(const Tensor<float>& weights, const Tensor<float>& bias) {
    std::vector<half> h_weights(weights.size);
    for (size_t i = 0; i < weights.size; i++) {
        h_weights[i] = __float2half(weights.data[i]);
    }

    half* d_weights;
    float* d_bias;
    CUDA_CHECK(cudaMalloc(&d_weights, weights.size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_bias, bias.size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_weights, h_weights.data(), h_weights.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, bias.data.get(), bias.size * sizeof(float), cudaMemcpyHostToDevice));

    return {d_weights, d_bias};
}

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
  auto l1w = deserializeTensor<float>(file).value().transpose();
  auto l1b = deserializeTensor<float>(file).value();
  auto l2w = deserializeTensor<float>(file).value().transpose();
  auto l2b = deserializeTensor<float>(file).value();

  auto [d_weights_l0, d_bias_l0] = uploadWeightsAndBias(l0w, l0b);
  auto [d_weights_l1, d_bias_l1] = uploadWeightsAndBias(l1w, l1b);
  auto [d_weights_l2, d_bias_l2] = uploadWeightsAndBias(l2w, l2b);

  // Stack first 25 images
  std::vector<Tensor<float>> first25Images(rawValidationImages.begin(), rawValidationImages.begin() + 25);
  auto stackedImages = Tensor<float>::stack(first25Images.begin(), first25Images.end());

  // Convert tensors to half precision and allocate device memory
  std::vector<half> h_images(stackedImages.size);
  for (size_t i = 0; i < stackedImages.size; i++) {
    h_images[i] = __float2half(stackedImages.data[i]);
  }

  half *d_images;
  CUDA_CHECK(cudaMalloc(&d_images, stackedImages.size * sizeof(half)));
  CUDA_CHECK(cudaMemcpy(d_images, h_images.data(), h_images.size() * sizeof(half), cudaMemcpyHostToDevice));

  const int BATCH_SIZE = stackedImages.shape[0];  // Number of images (25)
  const int IMAGE_SIZE = stackedImages.shape[1];  // Image size (784)
  const int NUM_FEATURES = l0w.shape[1];           // Output features (16)

  // Linear layer 0
  float *d_output_l0;
  CUDA_CHECK(cudaMalloc(&d_output_l0, BATCH_SIZE * NUM_FEATURES * sizeof(float)));

  dim3 gridDim((BATCH_SIZE + WMMA_M - 1) / WMMA_M, (NUM_FEATURES + WMMA_N - 1) / WMMA_N);
  dim3 blockDim(32, 1);

  linear_layer_forward<true><<<gridDim, blockDim>>>(d_images, d_weights_l0, d_bias_l0, d_output_l0, BATCH_SIZE, IMAGE_SIZE, NUM_FEATURES);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy result back to host
  std::vector<float> h_output(BATCH_SIZE * NUM_FEATURES);
  CUDA_CHECK(cudaMemcpy(h_output.data(), d_output_l0, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost));

  // Print first few elements of result
  fmt::println("First few elements of result:");
  for (int i = 0; i < 5; ++i) {
    fmt::print("{} ", h_output[i]);
  }
  fmt::println("");

  // Cleanup
  CUDA_CHECK(cudaFree(d_images));
  CUDA_CHECK(cudaFree(d_weights_l0));
  CUDA_CHECK(cudaFree(d_output_l0));
  CUDA_CHECK(cudaFree(d_bias_l0));

  return 0;
}
