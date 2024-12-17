
#include "cppdl/mnist_utils.h"
#include <fmt/format.h>

#include <fstream>
#include <stdexcept>

std::vector<Tensor<float>> loadImages(std::string path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error(fmt::format("Cannot load image file {}", path));
  }

  file.seekg(16, std::ios::beg); // Skip header.
  constexpr size_t imageSize = 784;
  std::vector<Tensor<float>> images;

  unsigned char buffer[imageSize];
  while (file.read(reinterpret_cast<char *>(buffer), imageSize)) {
    float image[imageSize];
    for (size_t i = 0; i < imageSize; ++i) {
      image[i] = buffer[i] / 255.0f;
    }
    images.push_back(Tensor<float>::vector(std::begin(image), std::end(image)));
  }
  return images;
}

std::vector<Tensor<float>> loadLabels(std::string path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error(fmt::format("Cannot load labels file {}", path));
  }

  file.seekg(8, std::ios::beg); // Skip header.

  std::vector<Tensor<float>> labels;

  char label;
  while (file.read(&label, 1)) {
    float labelOneHot[10] = {};
    labelOneHot[static_cast<size_t>(label)] = 1.0f;
    labels.emplace_back(
        Tensor<float>::vector(std::begin(labelOneHot), std::end(labelOneHot)));
  }
  return labels;
}
