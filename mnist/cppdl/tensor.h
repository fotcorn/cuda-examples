#pragma once

#include <cassert>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <vector>

#include <fmt/format.h>
#include <fmt/ranges.h>

inline void calculateStridesFromShape(const std::vector<size_t> &shape,
                                      std::vector<size_t> &strides) {
  strides.resize(shape.size());
  strides.back() = 1;
  for (int i = shape.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
}

namespace {
template <typename T>
T generateUniformRandom() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_real_distribution<T> dis(-1.0, 1.0);
  return dis(gen);
}
} // namespace

template <typename T>
struct Tensor final {
  Tensor(const std::vector<size_t> &shape, T init = 0) : shape(shape) {
    offset = 0;
    size = 1;
    for (int dim : shape) {
      if (dim <= 0)
        throw std::runtime_error("Dimension size must be greater than 0");
      size *= dim;
    }
    data = std::shared_ptr<T[]>(new T[size]);
    std::fill_n(data.get(), size, init);

    calculateStridesFromShape(shape, strides);
  };
  Tensor(std::shared_ptr<T[]> data, size_t offset, size_t size,
         std::vector<size_t> shape, std::vector<size_t> strides)
      : shape(std::move(shape)), data(data), offset(offset), size(size),
        strides(std::move(strides)) {}
  Tensor() = default;

  T item() const {
    if (shape.size() != 1 || shape[0] != 1) {
      throw std::runtime_error(
          "item() only works on tensors with one element.");
    }
    return data[offset];
  }

  static Tensor<T> ones(std::vector<size_t> shape) {
    Tensor<T> t(shape, 1);
    return t;
  }

  static Tensor<T> random(std::vector<size_t> shape) {
    Tensor<T> t(shape);
    for (size_t i = 0; i < t.size; i++) {
      t.data.get()[i] = generateUniformRandom<T>();
    }
    return t;
  }

  template <typename InputIt>
  static Tensor<T> vector(InputIt begin, InputIt end) {
    size_t size = std::distance(begin, end);
    std::vector<size_t> shape = {size};
    Tensor<T> t(shape);
    std::copy(begin, end, t.data.get());
    return t;
  }

  static Tensor<T> vector(std::initializer_list<T> data) {
    return vector(data.begin(), data.end());
  }

  static Tensor<T>
  matrix2d(std::initializer_list<std::initializer_list<T>> data) {
    if (data.size() == 0) {
      throw std::runtime_error("Input data cannot be empty.");
    }
    size_t subvectorSize = data.begin()->size();
    for (const auto &subvector : data) {
      if (subvector.size() != subvectorSize) {
        throw std::runtime_error("All subvectors must be the same size.");
      }
    }
    std::vector<size_t> shape = {data.size(), subvectorSize};
    Tensor<T> t(shape);
    T *ptr = t.data.get();
    for (const auto &subvector : data) {
      std::copy(subvector.begin(), subvector.end(), ptr);
      ptr += subvectorSize;
    }
    return t;
  }

  Tensor<T> operator[](const size_t index) const {
    if (index >= shape[0]) {
      throw std::runtime_error("index out of range");
    }
    size_t newOffset = this->offset + this->strides[0] * index;
    if (shape.size() == 1) {
      std::vector<size_t> newShape({1});
      std::vector<size_t> newStrides({1});
      return Tensor<T>(this->data, newOffset, 1, newShape, newStrides);
    }
    std::vector<size_t> newShape(this->shape.begin() + 1, this->shape.end());
    std::vector<size_t> newStrides(this->strides.begin() + 1,
                                   this->strides.end());
    size_t newSize = 1;
    for (int dim : newShape) {
      newSize *= dim;
    }
    return Tensor<T>(this->data, newOffset, newSize, newShape, newStrides);
  }

  // Elementwise ops.
  Tensor<T> apply(std::function<T(T)> func) const {
    assert(offset == 0);
    assert(strides.back() == 1);
    Tensor<T> result(this->shape);
    for (size_t i = 0; i < result.size; i++) {
      // TODO: take offset and stride into account.
      result.data.get()[i] = func(data.get()[i]);
    }
    return result;
  }

  Tensor<T> operator+(T op) const {
    return apply([op](T val) { return val + op; });
  }

  Tensor<T> operator-(T op) const {
    return apply([op](T val) { return val - op; });
  }

  Tensor<T> operator*(T op) const {
    return apply([op](T val) { return val * op; });
  }

  Tensor<T> operator/(T op) const {
    if (op == 0) {
      throw std::runtime_error("Division by zero is not allowed.");
    }
    return apply([op](T val) { return val / op; });
  }

  Tensor<T> relu() const {
    return apply([](T val) { return std::max<T>(0, val); });
  }

  // Tensor ops.
  Tensor<T> apply(const Tensor<T> &op, std::function<T(T, T)> func) const {
    auto &op1 = *this;
    auto &op2 = op;

    // Shape and strides are padded by 1's at the front.
    size_t length = std::max(op1.shape.size(), op2.shape.size());

    auto shapeOp1 = std::vector<size_t>(length, 1);
    auto shapeOp2 = std::vector<size_t>(length, 1);
    std::copy_backward(op1.shape.begin(), op1.shape.end(), shapeOp1.end());
    std::copy_backward(op2.shape.begin(), op2.shape.end(), shapeOp2.end());

    auto stridesOp1 = std::vector<size_t>(length, 1);
    auto stridesOp2 = std::vector<size_t>(length, 1);
    std::copy_backward(op1.strides.begin(), op1.strides.end(),
                       stridesOp1.end());
    std::copy_backward(op2.strides.begin(), op2.strides.end(),
                       stridesOp2.end());

    assert(shapeOp1.size() == shapeOp2.size());
    assert(stridesOp1.size() == stridesOp2.size());

    for (size_t i = 0; i < shapeOp1.size(); i++) {
      if (shapeOp1[i] != shapeOp2[i] && shapeOp1[i] != 1 && shapeOp2[i] != 1) {
        throw std::runtime_error(
            fmt::format("incompatible shapes for arithmetic operation "
                        "(broadcasting applied): {}, {}",
                        shapeOp1, shapeOp2));
      }
    }

    if (shapeOp1.size() == 1) {
      size_t dimOp1 = shapeOp1[0];
      size_t dimOp2 = shapeOp2[0];
      size_t maxDim = std::max(dimOp1, dimOp2);
      Tensor<T> res = Tensor<T>({maxDim});
      for (size_t i = 0; i < maxDim; i++) {
        res.data[i] = func(op1.data[op1.offset + stridesOp1[0] * (i % dimOp1)],
                           op2.data[op2.offset + stridesOp2[0] * (i % dimOp2)]);
      }
      return res;
    }

    if (shapeOp1.size() == 2) {
      size_t dim0Max = std::max(shapeOp1[0], shapeOp2[0]);
      size_t dim1Max = std::max(shapeOp1[1], shapeOp2[1]);

      Tensor<T> res = Tensor<T>({dim0Max, dim1Max});
      for (size_t dim0 = 0; dim0 < dim0Max; dim0++) {
        for (size_t dim1 = 0; dim1 < dim1Max; dim1++) {
          T index1 = op1.offset + (dim0 % shapeOp1[0]) * stridesOp1[0] +
                     (dim1 % shapeOp1[1]) * stridesOp1[1];
          T index2 = op2.offset + (dim0 % shapeOp2[0]) * stridesOp2[0] +
                     (dim1 % shapeOp2[1]) * stridesOp2[1];
          res.data[dim0 * res.strides[0] + dim1] =
              func(op1.data[index1], op2.data[index2]);
        }
      }
      return res;
    }

    if (shapeOp1.size() == 3) {
      size_t dim0Max = std::max(shapeOp1[0], shapeOp2[0]);
      size_t dim1Max = std::max(shapeOp1[1], shapeOp2[1]);
      size_t dim2Max = std::max(shapeOp1[2], shapeOp2[2]);

      Tensor<T> res = Tensor<T>({dim0Max, dim1Max, dim2Max});
      for (size_t dim0 = 0; dim0 < dim0Max; dim0++) {
        for (size_t dim1 = 0; dim1 < dim1Max; dim1++) {
          for (size_t dim2 = 0; dim2 < dim2Max; dim2++) {
            T index1 = op1.offset + (dim0 % shapeOp1[0]) * stridesOp1[0] +
                       (dim1 % shapeOp1[1]) * stridesOp1[1] +
                       (dim2 % shapeOp1[2]) * stridesOp1[2];
            T index2 = op2.offset + (dim0 % shapeOp2[0]) * stridesOp2[0] +
                       (dim1 % shapeOp2[1]) * stridesOp2[1] +
                       (dim2 % shapeOp2[2]) * stridesOp2[2];
            res.data[dim0 * res.strides[0] + dim1 * res.strides[1] + dim2] =
                func(op1.data[index1], op2.data[index2]);
          }
        }
      }
      return res;
    }

    throw std::runtime_error("unsupported shapes for arithmetic operation");
  }

  Tensor<T> operator+(const Tensor<T> &op) const {
    return apply(op, [](T v1, T v2) { return v1 + v2; });
  }

  Tensor<T> operator-(const Tensor<T> &op) const {
    return apply(op, [](T v1, T v2) { return v1 - v2; });
  }

  Tensor<T> operator*(const Tensor<T> &op) const {
    return apply(op, [](T v1, T v2) { return v1 * v2; });
  }

  Tensor<T> operator/(const Tensor<T> &op) const {
    return apply(op, [](T v1, T v2) {
      if (v2 == 0) {
        throw std::runtime_error("Division by zero is not allowed.");
      }
      return v1 / v2;
    });
  }

  Tensor<T> matmul(const Tensor<T> &op) const {
    auto &op1 = *this;
    auto &op2 = op;

    if (op1.shape.size() != op2.shape.size()) {
      throw std::runtime_error("matmul requires same number of dimensions");
    }
    if (op1.shape.size() == 2) {
      if (op1.shape[1] != op2.shape[0]) {
        throw std::runtime_error(
            fmt::format("matmul: second dimension of first matrix does "
                        "not match first dimension of second matrix {}, {}",
                        op1.shape, op2.shape));
      }

      const size_t dim0Max = op1.shape[0];
      const size_t dim1Max = op2.shape[1];
      const size_t maxI = op1.shape[1];
      Tensor<T> res = Tensor<T>({dim0Max, dim1Max});
      for (size_t dim0 = 0; dim0 < dim0Max; dim0++) {
        for (size_t dim1 = 0; dim1 < dim1Max; dim1++) {
          T sum = 0;
          for (size_t i = 0; i < maxI; i++) {
            T o1 = op1.data[op1.offset + dim0 * op1.strides[0] +
                            i * op1.strides[1]];
            T o2 = op2.data[op2.offset + i * op2.strides[0] +
                            dim1 * op2.strides[1]];
            sum += o1 * o2;
          }
          res.data[dim0 * res.strides[0] + dim1] = sum;
        }
      }

      return res;
    }
    throw std::runtime_error("matmul only supports 2-dimensional matrices");
  }

  std::string toString() const {
    if (shape.size() > 2) {
      throw std::runtime_error(
          "toString() only works on tensors with one or two dimensions.");
    }
    std::stringstream ss;
    if (shape.size() == 1) {
      ss << "[";
      for (size_t i = 0; i < shape[0]; i++) {
        ss << data[offset + i];
        if (i != shape[0] - 1) {
          ss << ", ";
        }
      }
      ss << "]";
    } else if (shape.size() == 2) {
      ss << "[\n";
      for (size_t i = 0; i < shape[0]; i++) {
        ss << "  [";
        for (size_t j = 0; j < shape[1]; j++) {
          ss << data[offset + i * strides[0] + j * strides[1]];
          if (j != shape[1] - 1) {
            ss << ", ";
          }
        }
        ss << "]\n";
      }
      ss << "]";
    }
    return ss.str();
  }

  template <typename InputIt>
  static Tensor<T> stack(InputIt begin, InputIt end) {
    if (begin == end) {
      throw std::runtime_error("Cannot stack empty list of tensors.");
    }
    auto inputShape = begin->shape;
    auto firstDimension = static_cast<size_t>(std::distance(begin, end));
    auto outputShape = std::vector<size_t>({firstDimension});
    outputShape.insert(outputShape.end(), inputShape.begin(), inputShape.end());

    auto result = Tensor<T>(outputShape);
    size_t offset = 0;
    for (auto it = begin; it != end; ++it) {
      // todo: take offset and strides into account
      assert(it->offset == 0);
      assert(it->strides.back() == 1);

      if (it->shape != inputShape) {
        throw std::runtime_error("stack: mismatched shapes");
      }
      std::copy(it->data.get(), it->data.get() + it->size,
                result.data.get() + offset);
      offset += it->size;
    }
    return result;
  }

  static Tensor<T> stack(std::initializer_list<Tensor<T>> tensors) {
    return stack(tensors.begin(), tensors.end());
  }

  Tensor<T> reshape(std::initializer_list<size_t> newShape) {
    size_t oldShapeProduct =
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies());
    size_t newShapeProduct =
        std::accumulate(newShape.begin(), newShape.end(), 1, std::multiplies());

    if (oldShapeProduct != newShapeProduct) {
      throw std::runtime_error(fmt::format(
          "reshape: old and new shape do not match: {}, {}", shape, newShape));
    }

    assert(strides.back() == 1);

    std::vector<size_t> newStrides;
    calculateStridesFromShape(newShape, newStrides);

    return Tensor<T>(data, offset, size, newShape, newStrides);
  }

  Tensor<T> transpose() const {
    if (shape.size() == 1) {
      return *this;
    }
    std::vector<size_t> newShape;
    std::reverse_copy(shape.begin(), shape.end(), std::back_inserter(newShape));
    auto result = Tensor<T>(newShape);

    if (shape.size() == 2) {
      size_t newIndex = 0;
      for (size_t dim1 = 0; dim1 < shape[1]; dim1++) {
        for (size_t dim0 = 0; dim0 < shape[0]; dim0++) {
          result.data.get()[newIndex] =
              data.get()[offset + dim0 * strides[0] + dim1 * strides[1]];
          newIndex++;
        }
      }
      return result;
    }
    if (shape.size() == 3) {
      size_t newIndex = 0;
      for (size_t dim2 = 0; dim2 < shape[2]; dim2++) {
        for (size_t dim1 = 0; dim1 < shape[1]; dim1++) {
          for (size_t dim0 = 0; dim0 < shape[0]; dim0++) {
            result.data.get()[newIndex] =
                data.get()[offset + dim0 * strides[0] + dim1 * strides[1] +
                           dim2 * strides[2]];
            newIndex++;
          }
        }
      }
      return result;
    }
    throw std::runtime_error("unsupported shape for transpose()");
  }

  T meanSquareError(Tensor<T> labels) {
    if (labels.shape != shape) {
      throw std::runtime_error(
          "meanSquareError: shapes of operands do not match");
    }

    T total = 0;
    if (shape.size() == 1) {
      for (size_t dim0 = 0; dim0 < shape[0]; dim0++) {
        T observed = data[offset + dim0 * strides[0]];
        T label = labels.data[labels.offset + dim0 * labels.strides[0]];
        T val = observed - label;
        total += val * val;
      }
      return total / shape[0];
    }

    if (shape.size() == 2) {
      T total = 0;
      for (size_t dim1 = 0; dim1 < shape[1]; dim1++) {
        for (size_t dim0 = 0; dim0 < shape[0]; dim0++) {
          T observed = data[offset + dim0 * strides[0] + dim1 * strides[1]];
          T label = labels.data[labels.offset + dim0 * labels.strides[0] +
                                dim1 * labels.strides[1]];
          T val = observed - label;
          total += val * val;
        }
      }
      return total / (shape[0] * shape[1]);
    }

    throw std::runtime_error("meanSquareError: unsupported shape size");
  }

  Tensor<T> softmax(size_t dimension = 0) {
    Tensor<T> result(shape);

    if (shape.size() == 1) {
      T sum = 0;
      for (size_t dim0 = 0; dim0 < shape[0]; dim0++) {
        T expElem = std::exp(data[offset + dim0 * strides[0]]);
        result.data.get()[dim0] = expElem;
        sum += expElem;
      }
      for (size_t dim0 = 0; dim0 < shape[0]; dim0++) {
        result.data.get()[dim0] /= sum;
      }
      return result;
    }

    if (shape.size() == 2) {
      if (dimension != 0 && dimension != 1) {
        throw std::runtime_error(
            fmt::format("Invalid dimension in softmax: {}", dimension));
      }
      // Index into strides and shape, based on which dimension (row or column)
      // we calculate on.
      int i0 = 1;
      int i1 = 0;
      if (dimension == 1) {
        i0 = 0;
        i1 = 1;
      }
      for (size_t dim0 = 0; dim0 < shape[i0]; dim0++) {
        T sum = 0.0;
        for (size_t dim1 = 0; dim1 < shape[i1]; dim1++) {
          T expElem =
              std::exp(data[offset + dim0 * strides[i0] + dim1 * strides[i1]]);
          size_t resultIndex =
              dim0 * result.strides[i0] + dim1 * result.strides[i1];
          result.data.get()[resultIndex] = expElem;
          sum += expElem;
        }
        for (size_t dim1 = 0; dim1 < shape[i1]; dim1++) {
          size_t resultIndex =
              dim0 * result.strides[i0] + dim1 * result.strides[i1];
          result.data.get()[resultIndex] /= sum;
        }
      }
      return result;
    }

    throw std::runtime_error("softmax: unsupported shape size");
  }

  Tensor<size_t> argmax(size_t dimension) const {
    if (dimension >= shape.size()) {
      throw std::runtime_error(fmt::format(
          "argmax: invalid dimension {} for shape {}", dimension, shape));
    }
    std::vector newShape(shape);
    newShape.erase(newShape.begin() + dimension);
    if (newShape.empty()) {
      newShape.push_back(1);
    }
    Tensor<size_t> result(newShape);

    if (shape.size() == 1) {
      T maxValue = data[offset];
      size_t maxIndex = 0;
      for (size_t dim0 = 1; dim0 < shape[0]; dim0++) {
        T currentValue = data[offset + dim0 * strides[0]];
        if (currentValue > maxValue) {
          maxValue = currentValue;
          maxIndex = dim0;
        }
      }
      result.data[0] = maxIndex;
      return result;
    }

    if (shape.size() == 2) {
      // Index into strides and shape, based on which dimension (row or column)
      // we calculate on.
      int i0 = 1;
      int i1 = 0;
      if (dimension == 1) {
        i0 = 0;
        i1 = 1;
      }
      for (size_t dim0 = 0; dim0 < shape[i0]; dim0++) {
        T maxValue = std::numeric_limits<T>::min();
        size_t maxIndex = 0;
        for (size_t dim1 = 0; dim1 < shape[i1]; dim1++) {
          T currentValue =
              data[offset + dim0 * strides[i0] + dim1 * strides[i1]];
          if (currentValue > maxValue) {
            maxValue = currentValue;
            maxIndex = dim1;
          }
        }
        result.data[dim0] = maxIndex;
      }
      return result;
    }

    throw std::runtime_error("argmax: unsupported shape size");
  }

  Tensor<T> sum(size_t dimension = 0) const {
    if (dimension >= shape.size()) {
      throw std::runtime_error(fmt::format(
          "sum: invalid dimension {} for shape {}", dimension, shape));
    }
    std::vector newShape(shape);
    newShape.erase(newShape.begin() + dimension);
    if (newShape.empty()) {
      newShape.push_back(1);
    }
    Tensor<T> result(newShape);

    if (shape.size() == 1) {
      T sum = 0.0f;
      for (size_t dim0 = 0; dim0 < shape[0]; dim0++) {
        sum += data[offset + dim0 * strides[0]];
      }
      result.data.get()[0] = sum;
      return result;
    }

    if (shape.size() == 2) {
      // Index into strides and shape, based on which dimension (row or column)
      // we calculate on.
      int i0 = 1;
      int i1 = 0;
      if (dimension == 1) {
        i0 = 0;
        i1 = 1;
      }
      for (size_t dim0 = 0; dim0 < shape[i0]; dim0++) {
        T sum = 0.0f;
        for (size_t dim1 = 0; dim1 < shape[i1]; dim1++) {
          sum += data[offset + dim0 * strides[i0] + dim1 * strides[i1]];
        }
        result.data.get()[dim0] = sum;
      }
      return result;
    }

    throw std::runtime_error("sum: unsupported shape size");
  }

  friend std::ostream &operator<<(std::ostream &os, const Tensor<T> &t) {
    os << t.toString();
    return os;
  }

  bool operator==(const Tensor<T> &other) const {
    if (this->shape != other.shape) {
      return false;
    }
    for (size_t i = 0; i < this->size; i++) {
      if (this->data.get()[i] != other.data.get()[i]) {
        return false;
      }
    }
    return true;
  }

  std::vector<size_t> shape;
  std::shared_ptr<T[]> data;
  size_t offset = 0;
  size_t size = 0;
  std::vector<size_t> strides;
};

namespace fmt {
template <typename T>
struct formatter<Tensor<T>> {
  template <typename ParseContext>
  constexpr auto parse(ParseContext &ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(const Tensor<T> &t, FormatContext &ctx) {
    return format_to(ctx.out(), "{}", t.toString());
  }
};
} // namespace fmt
