#include "cppdl/mnist_utils.h"
#include "cppdl/serialization.h"

#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char *argv[]) {
  if (argc < 2) {
    fmt::println(stderr, "Usage: {} <data_dir>\n", argv[0]);
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
  auto l0w = deserializeTensor<float>(file).value();
  auto l0b = deserializeTensor<float>(file).value();
  auto l1w = deserializeTensor<float>(file).value();
  auto l1b = deserializeTensor<float>(file).value();
  auto l2w = deserializeTensor<float>(file).value();
  auto l2b = deserializeTensor<float>(file).value();

  fmt::println("l0w shape: {}", l0w.shape);

  fmt::println("rawValidationImages[0] shape: {}", rawValidationImages.front().shape);
  fmt::println("rawValidationLabels[0] shape: {}", rawValidationLabels.front().shape);

  return 0;
}
