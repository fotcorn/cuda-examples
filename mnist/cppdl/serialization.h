#pragma once

#include "cppdl/tensor.h"

#include <istream>
#include <optional>
#include <ostream>

const char MagicNumber[] = "CPPDL";
const char Version = 1;

void writeFileHeader(std::ostream &out) {
  out.write(MagicNumber, sizeof(MagicNumber));
  out.write(&Version, sizeof(Version));
}

[[nodiscard]] bool readFileHeader(std::istream &in) {
  char magicNumber[sizeof(MagicNumber)];
  in.read(magicNumber, sizeof(magicNumber));
  if (std::memcmp(magicNumber, MagicNumber, sizeof(MagicNumber)) != 0) {
    return false;
  }
  char version;
  in.read(&version, sizeof(version));
  if (version != Version) {
    return false;
  }
  return true;
}

template <typename T>
void serializeTensor(const Tensor<T> &t, std::ostream &out) {
  std::uint8_t elementSize = sizeof(T);
  out.write(reinterpret_cast<const char *>(&elementSize), sizeof(elementSize));

  std::uint8_t rank = static_cast<uint8_t>(t.shape.size());
  out.write(reinterpret_cast<const char *>(&rank), sizeof(rank));

  for (const auto &dim : t.shape) {
    std::uint32_t dimension = static_cast<std::uint32_t>(dim);
    out.write(reinterpret_cast<const char *>(&dimension), sizeof(dimension));
  }

  switch (t.shape.size()) {
  case 1:
    for (size_t dim0 = 0; dim0 < t.shape[0]; dim0++) {
      T element = t.data[t.offset + dim0 * t.strides[0]];
      out.write(reinterpret_cast<const char *>(&element), sizeof(element));
    }
    break;
  case 2:
    for (size_t dim0 = 0; dim0 < t.shape[0]; dim0++) {
      for (size_t dim1 = 0; dim1 < t.shape[1]; dim1++) {
        T element =
            t.data[t.offset + dim0 * t.strides[0] + dim1 * t.strides[1]];
        out.write(reinterpret_cast<const char *>(&element), sizeof(element));
      }
    }
    break;
  case 3:
    for (size_t dim0 = 0; dim0 < t.shape[0]; dim0++) {
      for (size_t dim1 = 0; dim1 < t.shape[1]; dim1++) {
        for (size_t dim2 = 0; dim2 < t.shape[2]; dim2++) {
          T element = t.data[t.offset + dim0 * t.strides[0] +
                             dim1 * t.strides[1] + dim2 * t.strides[2]];
          out.write(reinterpret_cast<const char *>(&element), sizeof(element));
        }
      }
    }
    break;
  case 4:
    for (size_t dim0 = 0; dim0 < t.shape[0]; dim0++) {
      for (size_t dim1 = 0; dim1 < t.shape[1]; dim1++) {
        for (size_t dim2 = 0; dim2 < t.shape[2]; dim2++) {
          for (size_t dim3 = 0; dim3 < t.shape[3]; dim3++) {
            T element =
                t.data[t.offset + dim0 * t.strides[0] + dim1 * t.strides[1] +
                       dim2 * t.strides[2] + dim3 * t.strides[3]];
            out.write(reinterpret_cast<const char *>(&element),
                      sizeof(element));
          }
        }
      }
    }
    break;
  }
}

template <typename T>
std::optional<Tensor<T>> deserializeTensor(std::istream &in) {
  uint8_t elementSize;
  in.read(reinterpret_cast<char *>(&elementSize), sizeof(elementSize));
  if (elementSize != sizeof(T)) {
    return std::nullopt;
  }

  uint8_t rank;
  in.read(reinterpret_cast<char *>(&rank), sizeof(rank));

  size_t totalElements = 1;
  std::vector<size_t> shape;
  shape.resize(rank);
  for (size_t i = 0; i < rank; ++i) {
    uint32_t dimSize;
    in.read(reinterpret_cast<char *>(&dimSize), sizeof(dimSize));
    shape[i] = dimSize;
    totalElements *= dimSize;
  }

  std::vector<size_t> strides(rank);
  calculateStridesFromShape(shape, strides);

  Tensor<T> tensor(shape);
  for (size_t i = 0; i < totalElements; ++i) {
    T element;
    in.read(reinterpret_cast<char *>(&element), sizeof(element));
    tensor.data.get()[i] = element;
  }

  return tensor;
}
