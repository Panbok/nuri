#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <memory_resource>
#include <span>
#include <vector>

#include <glm/glm.hpp>
#include <stb_image_resize2.h>

#include "nuri/defines.h"

namespace nuri {

enum class BitmapType : uint8_t { Bitmap2D, BitmapCube, Count };

enum class BitmapFormat : uint8_t { U8, F32, Count };

class Bitmap final {
public:
  Bitmap() = default;

  Bitmap(int32_t width, int32_t height, int32_t components,
         BitmapFormat format,
         std::pmr::memory_resource *mem = std::pmr::get_default_resource())
      : width_(width), height_(height), components_(std::clamp(components, 1, 4)),
        format_(format), data_(mem) {
    initGetSetFuncs();
    resizeData();
  }

  Bitmap(int32_t width, int32_t height, int32_t depth, int32_t components,
         BitmapFormat format,
         std::pmr::memory_resource *mem = std::pmr::get_default_resource())
      : width_(width), height_(height), depth_(depth),
        components_(std::clamp(components, 1, 4)), format_(format), data_(mem) {
    initGetSetFuncs();
    resizeData();
  }

  Bitmap(int32_t width, int32_t height, int32_t components, BitmapFormat format,
         const void *srcData,
         std::pmr::memory_resource *mem = std::pmr::get_default_resource())
      : width_(width), height_(height), components_(std::clamp(components, 1, 4)),
        format_(format), data_(mem) {
    initGetSetFuncs();
    resizeData();
    if (srcData != nullptr && !data_.empty()) {
      std::memcpy(data_.data(), srcData, data_.size());
    }
  }

  ~Bitmap() = default;

  [[nodiscard]] static constexpr int32_t getBytesPerComponent(
      BitmapFormat format) noexcept {
    switch (format) {
    case BitmapFormat::U8:
      return 1;
    case BitmapFormat::F32:
      return 4;
    default:
      return 0;
    }
  }

  [[nodiscard]] int32_t width() const noexcept { return width_; }
  [[nodiscard]] int32_t height() const noexcept { return height_; }
  [[nodiscard]] int32_t depth() const noexcept { return depth_; }
  [[nodiscard]] int32_t components() const noexcept { return components_; }
  [[nodiscard]] BitmapFormat format() const noexcept { return format_; }
  [[nodiscard]] BitmapType type() const noexcept { return type_; }
  [[nodiscard]] bool empty() const noexcept { return data_.empty(); }

  [[nodiscard]] std::span<const uint8_t> data() const noexcept { return data_; }
  [[nodiscard]] std::span<uint8_t> data() noexcept { return data_; }

  void setPixel(int32_t x, int32_t y, const glm::vec4 &color);
  [[nodiscard]] glm::vec4 getPixel(int32_t x, int32_t y) const;

  [[nodiscard]] static float radicalInverseVdC(uint32_t bits) noexcept;
  [[nodiscard]] static glm::vec2 hammersley2D(uint32_t i,
                                              uint32_t sampleCount) noexcept;

  static void convolveLambertian(std::span<const glm::vec3> data, int32_t srcW,
                                 int32_t srcH, int32_t dstW, int32_t dstH,
                                 std::span<glm::vec3> output,
                                 int32_t numMonteCarloSamples);
  static void convolveGGX(std::span<const glm::vec3> data, int32_t srcW,
                          int32_t srcH, int32_t dstW, int32_t dstH,
                          std::span<glm::vec3> output,
                          int32_t numMonteCarloSamples);
  static void convolveGGX(std::span<const glm::vec3> data, int32_t srcW,
                          int32_t srcH, int32_t dstW, int32_t dstH,
                          std::span<glm::vec3> output,
                          int32_t numMonteCarloSamples, float roughness);

  [[nodiscard]] static glm::vec3 faceCoordsToXYZ(int32_t i, int32_t j,
                                                 int32_t faceID,
                                                 int32_t faceSize) noexcept;
  [[nodiscard]] Bitmap convertEquirectangularMapToVerticalCross() const;
  [[nodiscard]] Bitmap convertVerticalCrossToCubeMapFaces() const;
  [[nodiscard]] Bitmap convertEquirectangularMapToCubeMapFaces() const;

private:
  using SetPixelFn = void (Bitmap::*)(int32_t, int32_t, const glm::vec4 &);
  using GetPixelFn = glm::vec4 (Bitmap::*)(int32_t, int32_t) const;

  [[nodiscard]] bool isInside2D(int32_t x, int32_t y) const noexcept {
    return x >= 0 && y >= 0 && x < width_ && y < height_;
  }

  [[nodiscard]] size_t componentOffset2D(int32_t x, int32_t y) const noexcept {
    return static_cast<size_t>(components_) *
           (static_cast<size_t>(y) * static_cast<size_t>(width_) +
            static_cast<size_t>(x));
  }

  [[nodiscard]] size_t byteOffset2D(int32_t x, int32_t y) const noexcept {
    return componentOffset2D(x, y) *
           static_cast<size_t>(getBytesPerComponent(format_));
  }

  [[nodiscard]] std::pmr::memory_resource *memoryResource() const noexcept {
    return data_.get_allocator().resource();
  }

  void resizeData() {
    if (width_ <= 0 || height_ <= 0 || depth_ <= 0 || components_ <= 0) {
      data_.clear();
      return;
    }

    const int32_t bytesPerComponent = getBytesPerComponent(format_);
    if (bytesPerComponent <= 0) {
      data_.clear();
      return;
    }

    const size_t pixelCount = static_cast<size_t>(width_) *
                              static_cast<size_t>(height_) *
                              static_cast<size_t>(depth_);
    const size_t totalComponents =
        pixelCount * static_cast<size_t>(components_);
    const size_t totalBytes =
        totalComponents * static_cast<size_t>(bytesPerComponent);
    data_.resize(totalBytes);
  }

  void initGetSetFuncs() {
    switch (format_) {
    case BitmapFormat::U8:
      setPixelFn_ = &Bitmap::setPixelU8;
      getPixelFn_ = &Bitmap::getPixelU8;
      break;
    case BitmapFormat::F32:
      setPixelFn_ = &Bitmap::setPixelF32;
      getPixelFn_ = &Bitmap::getPixelF32;
      break;
    default:
      setPixelFn_ = &Bitmap::setPixelU8;
      getPixelFn_ = &Bitmap::getPixelU8;
      break;
    }
  }

  void setPixelF32(int32_t x, int32_t y, const glm::vec4 &color) {
    if (!isInside2D(x, y)) {
      return;
    }
    const size_t baseByteOffset = byteOffset2D(x, y);
    if (components_ > 0) {
      std::memcpy(data_.data() + baseByteOffset + 0U * sizeof(float), &color.x,
                  sizeof(float));
    }
    if (components_ > 1) {
      std::memcpy(data_.data() + baseByteOffset + 1U * sizeof(float), &color.y,
                  sizeof(float));
    }
    if (components_ > 2) {
      std::memcpy(data_.data() + baseByteOffset + 2U * sizeof(float), &color.z,
                  sizeof(float));
    }
    if (components_ > 3) {
      std::memcpy(data_.data() + baseByteOffset + 3U * sizeof(float), &color.w,
                  sizeof(float));
    }
  }

  [[nodiscard]] glm::vec4 getPixelF32(int32_t x, int32_t y) const {
    if (!isInside2D(x, y)) {
      return glm::vec4(0.0f);
    }

    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    float a = 0.0f;
    const size_t baseByteOffset = byteOffset2D(x, y);
    if (components_ > 0) {
      std::memcpy(&r, data_.data() + baseByteOffset + 0U * sizeof(float),
                  sizeof(float));
    }
    if (components_ > 1) {
      std::memcpy(&g, data_.data() + baseByteOffset + 1U * sizeof(float),
                  sizeof(float));
    }
    if (components_ > 2) {
      std::memcpy(&b, data_.data() + baseByteOffset + 2U * sizeof(float),
                  sizeof(float));
    }
    if (components_ > 3) {
      std::memcpy(&a, data_.data() + baseByteOffset + 3U * sizeof(float),
                  sizeof(float));
    }

    return glm::vec4(r, g, b, a);
  }

  void setPixelU8(int32_t x, int32_t y, const glm::vec4 &color) {
    if (!isInside2D(x, y)) {
      return;
    }
    const size_t baseComponentOffset = componentOffset2D(x, y);
    const glm::vec4 clamped = glm::clamp(color, glm::vec4(0.0f), glm::vec4(1.0f));
    if (components_ > 0) {
      data_[baseComponentOffset + 0U] = static_cast<uint8_t>(clamped.x * 255.0f);
    }
    if (components_ > 1) {
      data_[baseComponentOffset + 1U] = static_cast<uint8_t>(clamped.y * 255.0f);
    }
    if (components_ > 2) {
      data_[baseComponentOffset + 2U] = static_cast<uint8_t>(clamped.z * 255.0f);
    }
    if (components_ > 3) {
      data_[baseComponentOffset + 3U] = static_cast<uint8_t>(clamped.w * 255.0f);
    }
  }

  [[nodiscard]] glm::vec4 getPixelU8(int32_t x, int32_t y) const {
    if (!isInside2D(x, y)) {
      return glm::vec4(0.0f);
    }
    const size_t baseComponentOffset = componentOffset2D(x, y);
    return glm::vec4(
        components_ > 0 ? static_cast<float>(data_[baseComponentOffset + 0U]) /
                              255.0f
                        : 0.0f,
        components_ > 1 ? static_cast<float>(data_[baseComponentOffset + 1U]) /
                              255.0f
                        : 0.0f,
        components_ > 2 ? static_cast<float>(data_[baseComponentOffset + 2U]) /
                              255.0f
                        : 0.0f,
        components_ > 3 ? static_cast<float>(data_[baseComponentOffset + 3U]) /
                              255.0f
                        : 0.0f);
  }

  int32_t width_ = 0;
  int32_t height_ = 0;
  int32_t depth_ = 1;
  int32_t components_ = 3;
  BitmapFormat format_ = BitmapFormat::U8;
  BitmapType type_ = BitmapType::Bitmap2D;
  std::pmr::vector<uint8_t> data_ =
      std::pmr::vector<uint8_t>(std::pmr::get_default_resource());
  SetPixelFn setPixelFn_ = &Bitmap::setPixelU8;
  GetPixelFn getPixelFn_ = &Bitmap::getPixelU8;
};

inline void Bitmap::setPixel(int32_t x, int32_t y, const glm::vec4 &color) {
  if (data_.empty()) {
    return;
  }
  (this->*setPixelFn_)(x, y, color);
}

inline glm::vec4 Bitmap::getPixel(int32_t x, int32_t y) const {
  if (data_.empty()) {
    return glm::vec4(0.0f);
  }
  return (this->*getPixelFn_)(x, y);
}

inline float Bitmap::radicalInverseVdC(uint32_t bits) noexcept {
  bits = (bits << 16U) | (bits >> 16U);
  bits = ((bits & 0x55555555U) << 1U) | ((bits & 0xAAAAAAAAU) >> 1U);
  bits = ((bits & 0x33333333U) << 2U) | ((bits & 0xCCCCCCCCU) >> 2U);
  bits = ((bits & 0x0F0F0F0FU) << 4U) | ((bits & 0xF0F0F0F0U) >> 4U);
  bits = ((bits & 0x00FF00FFU) << 8U) | ((bits & 0xFF00FF00U) >> 8U);
  return static_cast<float>(bits) * 2.3283064365386963e-10F;
}

inline glm::vec2 Bitmap::hammersley2D(uint32_t i, uint32_t sampleCount) noexcept {
  if (sampleCount == 0U) {
    return glm::vec2(0.0F);
  }
  return glm::vec2(static_cast<float>(i) / static_cast<float>(sampleCount),
                   radicalInverseVdC(i));
}

inline void Bitmap::convolveLambertian(std::span<const glm::vec3> data, int32_t srcW,
                                       int32_t srcH, int32_t dstW, int32_t dstH,
                                       std::span<glm::vec3> output,
                                       int32_t numMonteCarloSamples) {
  if (srcW <= 0 || srcH <= 0 || dstW <= 0 || dstH <= 0 ||
      numMonteCarloSamples <= 0) {
    return;
  }
  if (srcW != (2 * srcH)) {
    return;
  }

  const size_t srcPixels =
      static_cast<size_t>(srcW) * static_cast<size_t>(srcH);
  const size_t dstPixels =
      static_cast<size_t>(dstW) * static_cast<size_t>(dstH);
  if (data.size() < srcPixels || output.size() < dstPixels) {
    return;
  }

  std::vector<glm::vec3> tmp(dstPixels);
  void *resized = stbir_resize(
      reinterpret_cast<const float *>(data.data()), srcW, srcH, 0,
      reinterpret_cast<float *>(tmp.data()), dstW, dstH, 0, STBIR_RGB,
      STBIR_TYPE_FLOAT, STBIR_EDGE_WRAP, STBIR_FILTER_CUBICBSPLINE);
  if (resized == nullptr) {
    return;
  }

  constexpr float kPi = 3.14159265358979323846F;
  constexpr float kTwoPi = 6.28318530717958647692F;

  const glm::vec3 *scratch = tmp.data();
  srcW = dstW;
  srcH = dstH;

  for (int32_t y = 0; y < dstH; ++y) {
    const float theta1 = (static_cast<float>(y) / static_cast<float>(dstH)) * kPi;
    for (int32_t x = 0; x < dstW; ++x) {
      const float phi1 =
          (static_cast<float>(x) / static_cast<float>(dstW)) * kTwoPi;
      const glm::vec3 v1(std::sin(theta1) * std::cos(phi1),
                         std::sin(theta1) * std::sin(phi1), std::cos(theta1));

      glm::vec3 color(0.0F);
      float weight = 0.0F;
      for (int32_t i = 0; i < numMonteCarloSamples; ++i) {
        const glm::vec2 h =
            hammersley2D(static_cast<uint32_t>(i),
                         static_cast<uint32_t>(numMonteCarloSamples));
        const int32_t x1 = std::clamp(static_cast<int32_t>(
                                          std::floor(h.x * static_cast<float>(srcW))),
                                      0, srcW - 1);
        const int32_t y1 = std::clamp(static_cast<int32_t>(
                                          std::floor(h.y * static_cast<float>(srcH))),
                                      0, srcH - 1);

        const float theta2 =
            (static_cast<float>(y1) / static_cast<float>(srcH)) * kPi;
        const float phi2 =
            (static_cast<float>(x1) / static_cast<float>(srcW)) * kTwoPi;
        const glm::vec3 v2(std::sin(theta2) * std::cos(phi2),
                           std::sin(theta2) * std::sin(phi2), std::cos(theta2));
        const float d = std::max(0.0F, glm::dot(v1, v2));
        if (d > 0.01F) {
          color += scratch[static_cast<size_t>(y1) * static_cast<size_t>(srcW) +
                           static_cast<size_t>(x1)] *
                   d;
          weight += d;
        }
      }

      output[static_cast<size_t>(y) * static_cast<size_t>(dstW) +
             static_cast<size_t>(x)] = (weight > 0.0F) ? (color / weight)
                                                       : glm::vec3(0.0F);
    }
  }
}

inline void Bitmap::convolveGGX(std::span<const glm::vec3> data, int32_t srcW,
                                int32_t srcH, int32_t dstW, int32_t dstH,
                                std::span<glm::vec3> output,
                                int32_t numMonteCarloSamples) {
  convolveGGX(data, srcW, srcH, dstW, dstH, output, numMonteCarloSamples, 0.5F);
}

inline void Bitmap::convolveGGX(std::span<const glm::vec3> data, int32_t srcW,
                                int32_t srcH, int32_t dstW, int32_t dstH,
                                std::span<glm::vec3> output,
                                int32_t numMonteCarloSamples, float roughness) {
  if (srcW <= 0 || srcH <= 0 || dstW <= 0 || dstH <= 0 ||
      numMonteCarloSamples <= 0) {
    return;
  }
  if (srcW != (2 * srcH)) {
    return;
  }

  const size_t srcPixels =
      static_cast<size_t>(srcW) * static_cast<size_t>(srcH);
  const size_t dstPixels =
      static_cast<size_t>(dstW) * static_cast<size_t>(dstH);
  if (data.size() < srcPixels || output.size() < dstPixels) {
    return;
  }

  constexpr float kPi = 3.14159265358979323846F;
  constexpr float kTwoPi = 6.28318530717958647692F;
  const float safeRoughness = std::clamp(roughness, 0.001F, 1.0F);
  const float a = safeRoughness * safeRoughness;
  const float a2 = a * a;

  const auto sampleEquirectBilinear = [&](const glm::vec3 &dir) -> glm::vec3 {
    const float u = (std::atan2(dir.y, dir.x) / kTwoPi) + 0.5F;
    const float v = std::acos(std::clamp(dir.z, -1.0F, 1.0F)) / kPi;

    const float wrappedU = u - std::floor(u);
    const float clampedV = std::clamp(v, 0.0F, 1.0F);
    const float fx = wrappedU * static_cast<float>(srcW);
    const float fy = clampedV * static_cast<float>(srcH - 1);

    const int32_t x0 = std::clamp(static_cast<int32_t>(std::floor(fx)), 0, srcW - 1);
    const int32_t y0 = std::clamp(static_cast<int32_t>(std::floor(fy)), 0, srcH - 1);
    const int32_t x1 = (x0 + 1) % srcW;
    const int32_t y1 = std::min(y0 + 1, srcH - 1);

    const float tx = fx - static_cast<float>(x0);
    const float ty = fy - static_cast<float>(y0);

    const glm::vec3 c00 = data[static_cast<size_t>(y0) * static_cast<size_t>(srcW) +
                               static_cast<size_t>(x0)];
    const glm::vec3 c10 = data[static_cast<size_t>(y0) * static_cast<size_t>(srcW) +
                               static_cast<size_t>(x1)];
    const glm::vec3 c01 = data[static_cast<size_t>(y1) * static_cast<size_t>(srcW) +
                               static_cast<size_t>(x0)];
    const glm::vec3 c11 = data[static_cast<size_t>(y1) * static_cast<size_t>(srcW) +
                               static_cast<size_t>(x1)];
    const glm::vec3 cx0 = glm::mix(c00, c10, tx);
    const glm::vec3 cx1 = glm::mix(c01, c11, tx);
    return glm::mix(cx0, cx1, ty);
  };

  for (int32_t y = 0; y < dstH; ++y) {
    const float theta = (static_cast<float>(y) / static_cast<float>(dstH)) * kPi;
    for (int32_t x = 0; x < dstW; ++x) {
      const float phi = (static_cast<float>(x) / static_cast<float>(dstW)) * kTwoPi;
      const glm::vec3 n(std::sin(theta) * std::cos(phi),
                        std::sin(theta) * std::sin(phi), std::cos(theta));
      const glm::vec3 v = n;

      const glm::vec3 up = (std::abs(n.z) < 0.999F) ? glm::vec3(0.0F, 0.0F, 1.0F)
                                                     : glm::vec3(1.0F, 0.0F, 0.0F);
      const glm::vec3 tangent = glm::normalize(glm::cross(up, n));
      const glm::vec3 bitangent = glm::cross(n, tangent);

      glm::vec3 prefiltered(0.0F);
      float totalWeight = 0.0F;

      for (int32_t i = 0; i < numMonteCarloSamples; ++i) {
        const glm::vec2 xi =
            hammersley2D(static_cast<uint32_t>(i),
                         static_cast<uint32_t>(numMonteCarloSamples));

        const float phiH = kTwoPi * xi.x;
        const float cosThetaH =
            std::sqrt((1.0F - xi.y) / (1.0F + ((a2 - 1.0F) * xi.y)));
        const float sinThetaH =
            std::sqrt(std::max(0.0F, 1.0F - (cosThetaH * cosThetaH)));

        const glm::vec3 hTangent(std::cos(phiH) * sinThetaH,
                                 std::sin(phiH) * sinThetaH, cosThetaH);
        const glm::vec3 h = glm::normalize((tangent * hTangent.x) +
                                           (bitangent * hTangent.y) +
                                           (n * hTangent.z));
        glm::vec3 l = (2.0F * glm::dot(v, h) * h) - v;
        const float lLenSq = glm::dot(l, l);
        if (lLenSq <= 0.0F) {
          continue;
        }
        l /= std::sqrt(lLenSq);

        const float nDotL = std::max(glm::dot(n, l), 0.0F);
        if (nDotL > 0.0F) {
          prefiltered += sampleEquirectBilinear(l) * nDotL;
          totalWeight += nDotL;
        }
      }

      output[static_cast<size_t>(y) * static_cast<size_t>(dstW) +
             static_cast<size_t>(x)] =
          (totalWeight > 0.0F) ? (prefiltered / totalWeight) : glm::vec3(0.0F);
    }
  }
}

inline glm::vec3 Bitmap::faceCoordsToXYZ(int32_t i, int32_t j, int32_t faceID,
                                         int32_t faceSize) noexcept {
  if (faceSize <= 0) {
    return glm::vec3(0.0F);
  }

  const float a = 2.0F * (static_cast<float>(i) / static_cast<float>(faceSize));
  const float b = 2.0F * (static_cast<float>(j) / static_cast<float>(faceSize));

  if (faceID == 0) {
    return glm::vec3(-1.0F, a - 1.0F, b - 1.0F);
  }
  if (faceID == 1) {
    return glm::vec3(a - 1.0F, -1.0F, 1.0F - b);
  }
  if (faceID == 2) {
    return glm::vec3(1.0F, a - 1.0F, 1.0F - b);
  }
  if (faceID == 3) {
    return glm::vec3(1.0F - a, 1.0F, 1.0F - b);
  }
  if (faceID == 4) {
    return glm::vec3(b - 1.0F, a - 1.0F, 1.0F);
  }
  if (faceID == 5) {
    return glm::vec3(1.0F - b, a - 1.0F, -1.0F);
  }

  return glm::vec3(0.0F);
}

inline Bitmap Bitmap::convertEquirectangularMapToVerticalCross() const {
  if (type_ != BitmapType::Bitmap2D || width_ <= 0 || height_ <= 0 || data_.empty()) {
    return Bitmap();
  }

  const int32_t faceSize = width_ / 4;
  if (faceSize <= 0) {
    return Bitmap();
  }

  const int32_t resultW = faceSize * 3;
  const int32_t resultH = faceSize * 4;
  Bitmap result(resultW, resultH, components_, format_, memoryResource());

  const std::array<glm::ivec2, 6> faceOffsets{
      glm::ivec2(faceSize, faceSize * 3), glm::ivec2(0, faceSize),
      glm::ivec2(faceSize, faceSize), glm::ivec2(faceSize * 2, faceSize),
      glm::ivec2(faceSize, 0), glm::ivec2(faceSize, faceSize * 2)};

  constexpr float kPi = 3.14159265358979323846F;

  const int32_t clampW = width_ - 1;
  const int32_t clampH = height_ - 1;

  for (int32_t face = 0; face < 6; ++face) {
    for (int32_t i = 0; i < faceSize; ++i) {
      for (int32_t j = 0; j < faceSize; ++j) {
        const glm::vec3 p = faceCoordsToXYZ(i, j, face, faceSize);
        const float r = std::hypot(p.x, p.y);
        const float theta = std::atan2(p.y, p.x);
        const float phi = std::atan2(p.z, r);

        const float uf =
            (2.0F * static_cast<float>(faceSize) * (theta + kPi)) / kPi;
        const float vf =
            (2.0F * static_cast<float>(faceSize) * ((kPi * 0.5F) - phi)) / kPi;

        const int32_t u1 =
            std::clamp(static_cast<int32_t>(std::floor(uf)), 0, clampW);
        const int32_t v1 =
            std::clamp(static_cast<int32_t>(std::floor(vf)), 0, clampH);
        const int32_t u2 = std::clamp(u1 + 1, 0, clampW);
        const int32_t v2 = std::clamp(v1 + 1, 0, clampH);

        const float s = uf - static_cast<float>(u1);
        const float t = vf - static_cast<float>(v1);

        const glm::vec4 a = getPixel(u1, v1);
        const glm::vec4 b = getPixel(u2, v1);
        const glm::vec4 c = getPixel(u1, v2);
        const glm::vec4 d = getPixel(u2, v2);
        const glm::vec4 color = a * (1.0F - s) * (1.0F - t) +
                                b * s * (1.0F - t) + c * (1.0F - s) * t +
                                d * s * t;
        result.setPixel(i + faceOffsets[static_cast<size_t>(face)].x,
                        j + faceOffsets[static_cast<size_t>(face)].y, color);
      }
    }
  }

  return result;
}

inline Bitmap Bitmap::convertVerticalCrossToCubeMapFaces() const {
  if (type_ != BitmapType::Bitmap2D || width_ <= 0 || height_ <= 0 || data_.empty()) {
    return Bitmap();
  }

  const int32_t faceWidth = width_ / 3;
  const int32_t faceHeight = height_ / 4;
  if (faceWidth <= 0 || faceHeight <= 0) {
    return Bitmap();
  }

  Bitmap cubemap(faceWidth, faceHeight, 6, components_, format_,
                 memoryResource());
  cubemap.type_ = BitmapType::BitmapCube;

  const uint8_t *src = data_.data();
  uint8_t *dst = cubemap.data_.data();
  const int32_t pixelSize = components_ * getBytesPerComponent(format_);
  if (pixelSize <= 0) {
    return Bitmap();
  }

  for (int32_t face = 0; face < 6; ++face) {
    for (int32_t j = 0; j < faceHeight; ++j) {
      for (int32_t i = 0; i < faceWidth; ++i) {
        int32_t x = 0;
        int32_t y = 0;

        switch (face) {
        case 0:
          x = (2 * faceWidth) + i;
          y = faceHeight + j;
          break;
        case 1:
          x = i;
          y = faceHeight + j;
          break;
        case 2:
          x = faceWidth + i;
          y = j;
          break;
        case 3:
          x = faceWidth + i;
          y = (2 * faceHeight) + j;
          break;
        case 4:
          x = faceWidth + i;
          y = faceHeight + j;
          break;
        case 5:
          x = (2 * faceWidth) - (i + 1);
          y = height_ - (j + 1);
          break;
        default:
          break;
        }

        const int32_t safeX = std::clamp(x, 0, width_ - 1);
        const int32_t safeY = std::clamp(y, 0, height_ - 1);
        const size_t srcOffset =
            (static_cast<size_t>(safeY) * static_cast<size_t>(width_) +
             static_cast<size_t>(safeX)) *
            static_cast<size_t>(pixelSize);
        std::memcpy(dst, src + srcOffset, static_cast<size_t>(pixelSize));
        dst += pixelSize;
      }
    }
  }

  return cubemap;
}

inline Bitmap Bitmap::convertEquirectangularMapToCubeMapFaces() const {
  Bitmap cross = convertEquirectangularMapToVerticalCross();
  return cross.convertVerticalCrossToCubeMapFaces();
}

} // namespace nuri
