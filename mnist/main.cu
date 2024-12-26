#include "cppdl/mnist_utils.h"
#include "cppdl/serialization.h"
#include "linear_layer.cuh"

#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

// Helper function to upload weights and bias to GPU
std::pair<half *, float *> uploadWeightsAndBias(const Tensor<float> &weights,
                                                const Tensor<float> &bias) {
  // Calculate padded dimensions to be multiples of 16
  size_t padded_rows = (weights.shape[0] + WMMA_M - 1) / WMMA_M * WMMA_M;
  size_t padded_cols = (weights.shape[1] + WMMA_N - 1) / WMMA_N * WMMA_N;
  size_t padded_weights_size = padded_rows * padded_cols;

  std::vector<half> h_weights(padded_weights_size, __float2half(0.0f));
  
  // Copy weights row by row, padding each row to multiple of 16
  for (size_t i = 0; i < weights.shape[0]; i++) {
    for (size_t j = 0; j < weights.shape[1]; j++) {
      h_weights[i * padded_cols + j] = 
        __float2half(weights.data[i * weights.shape[1] + j]);
    }
  }

  // Calculate padded bias size to be multiple of 16
  size_t padded_bias_size = (bias.size + WMMA_N - 1) / WMMA_N * WMMA_N;
  std::vector<float> padded_bias(padded_bias_size, 0.0f);
  std::copy(bias.data.get(), bias.data.get() + bias.size, padded_bias.begin());

  half *d_weights;
  float *d_bias;
  CUDA_CHECK(cudaMalloc(&d_weights, padded_weights_size * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_bias, padded_bias_size * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_weights, h_weights.data(),
                        padded_weights_size * sizeof(half),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_bias, padded_bias.data(), padded_bias_size * sizeof(float),
                        cudaMemcpyHostToDevice));

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
    std::cerr
        << "Device needs to support CUDA compute capability 7.0 or higher\n";
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
  std::vector<Tensor<float>> first25Images(rawValidationImages.begin(),
                                           rawValidationImages.begin() + 25);
  auto stackedImages =
      Tensor<float>::stack(first25Images.begin(), first25Images.end());

  const int NUM_CLASSES = 10;

  // Allocate device memory for input images and copy data to device
  float *d_images;
  CUDA_CHECK(cudaMalloc(&d_images, stackedImages.size * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_images, stackedImages.data.get(),
                        stackedImages.size * sizeof(float),
                        cudaMemcpyHostToDevice));

  const int BATCH_SIZE = stackedImages.shape[0]; // Number of images (25)
  const int IMAGE_SIZE = stackedImages.shape[1]; // Image size (784)
  const int NUM_FEATURES = l0w.shape[1];         // Output features (16)

  // Linear layer 0
  float *d_output_l0;
  CUDA_CHECK(
      cudaMalloc(&d_output_l0, BATCH_SIZE * NUM_FEATURES * sizeof(float)));

  dim3 gridDim((BATCH_SIZE + WMMA_M - 1) / WMMA_M,
               (NUM_FEATURES + WMMA_N - 1) / WMMA_N);
  dim3 blockDim(32, 1);

  linear_layer_forward<true>
      <<<gridDim, blockDim>>>(d_images, d_weights_l0, d_bias_l0, d_output_l0,
                              BATCH_SIZE, IMAGE_SIZE, NUM_FEATURES);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  float *d_output_l1;
  CUDA_CHECK(
      cudaMalloc(&d_output_l1, BATCH_SIZE * NUM_FEATURES * sizeof(float)));
  linear_layer_forward<true>
      <<<gridDim, blockDim>>>(d_output_l0, d_weights_l1, d_bias_l1, d_output_l1,
                              BATCH_SIZE, NUM_FEATURES, NUM_FEATURES);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  float *d_output_l2;
  CUDA_CHECK(
      cudaMalloc(&d_output_l2, BATCH_SIZE * /*NUM_CLASSES*/ NUM_FEATURES * sizeof(float)));
  linear_layer_forward<false>
      <<<gridDim, blockDim>>>(d_output_l1, d_weights_l2, d_bias_l2, d_output_l2,
                              BATCH_SIZE, NUM_FEATURES, /*NUM_CLASSES*/ NUM_FEATURES);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Add this code to print layer 0 output
  std::vector<float> h_output_l0(BATCH_SIZE * NUM_FEATURES);
  CUDA_CHECK(cudaMemcpy(h_output_l0.data(), d_output_l0,
                        h_output_l0.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  fmt::println("Layer 0 first 5 outputs:");
  for (int i = 0; i < 32; ++i) {
    if (i == 16) {
      fmt::println("");
    }
    fmt::print("{:.4f} ", h_output_l0[i]);
  }
  fmt::println("\n");

  // Add this code to print layer 1 output
  std::vector<float> h_output_l1(BATCH_SIZE * NUM_FEATURES);
  CUDA_CHECK(cudaMemcpy(h_output_l1.data(), d_output_l1,
                        h_output_l1.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  fmt::println("Layer 1 first 5 outputs:");
  for (int i = 0; i < 32; ++i) {
    if (i == 16) {
      fmt::println("");
    }
    fmt::print("{:.4f} ", h_output_l1[i]);
  }
  fmt::println("\n");

  // Add this code to print layer 2 output (before the existing output copy)
  std::vector<float> h_output_l2(BATCH_SIZE * /*NUM_CLASSES*/ NUM_FEATURES);
  CUDA_CHECK(cudaMemcpy(h_output_l2.data(), d_output_l2,
                        h_output_l2.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  fmt::println("Layer 2 first 5 outputs:");
  for (int i = 0; i < 32; ++i) {
    if (i == 16) {
      fmt::println("");
    }
    fmt::print("{:.4f} ", h_output_l2[i]);
  }
  fmt::println("\n");

  // Print first few elements of result
  fmt::println("Predictions vs Labels:");
  for (int i = 0; i < BATCH_SIZE; ++i) {
    // Find argmax for predictions
    int pred_class = 0;
    float max_val = h_output_l2[i * NUM_FEATURES];
    for (int j = 1; j < NUM_FEATURES; ++j) {
      float val = h_output_l2[i * NUM_FEATURES + j];
      //fmt::print("{} ", val);
      if (val > max_val) {
        max_val = val;
        pred_class = j;
      }
    }
    //fmt::println("");

    fmt::print("Image {}: Predicted {} | Actual {}\n", i, pred_class, rawValidationLabels[i].argmax(0).data[0]);
  }

  // Cleanup
  CUDA_CHECK(cudaFree(d_images));
  CUDA_CHECK(cudaFree(d_weights_l0));
  CUDA_CHECK(cudaFree(d_weights_l1));
  CUDA_CHECK(cudaFree(d_weights_l2));
  CUDA_CHECK(cudaFree(d_output_l0));
  CUDA_CHECK(cudaFree(d_output_l1));
  CUDA_CHECK(cudaFree(d_output_l2));
  CUDA_CHECK(cudaFree(d_bias_l0));
  CUDA_CHECK(cudaFree(d_bias_l1));
  CUDA_CHECK(cudaFree(d_bias_l2));

  return 0;
}
