#pragma once

#include "cppdl/tensor.h"

#include <string>
#include <vector>

std::vector<Tensor<float>> loadImages(std::string path);
std::vector<Tensor<float>> loadLabels(std::string path);
