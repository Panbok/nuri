#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <glm/glm.hpp>
#include <memory_resource>
#include <span>
#include <vector>
namespace nuri {

enum class BitmapType : uint8_t { Bitmap2D, BitmapCube, Bitmap3D };
enum class BitmapFormat : uint8_t { U8, F32 };

class Bitmap final {
public:
  Bitmap() = default;
  Bitmap(int32_t width, int32_t height, int32_t components, BitmapFormat format,
         std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : Bitmap(width, height, 1, components, format, memory) {}
  Bitmap(int32_t width, int32_t height, int32_t depth, int32_t components,
         BitmapFormat format,
         std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : width_(width), height_(height), depth_(depth),
        components_(std::clamp(components, 1, 4)), format_(format),
        type_(depth > 1 ? BitmapType::Bitmap3D : BitmapType::Bitmap2D),
        data_(memory) {
    if (width > 0 && height > 0 && depth > 0) {
      data_.resize(static_cast<size_t>(width) * static_cast<size_t>(height) *
                   static_cast<size_t>(depth) *
                   static_cast<size_t>(components_) * bytesPerComponent());
    }
  }
  Bitmap(int32_t width, int32_t height, int32_t components, BitmapFormat format,
         const void *source,
         std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : Bitmap(width, height, components, format, memory) {
    if (source != nullptr) {
      std::memcpy(data_.data(), source, data_.size());
    }
  }
  [[nodiscard]] static constexpr int32_t
  getBytesPerComponent(BitmapFormat format) noexcept {
    constexpr std::array<int32_t, 2> sizes{1, 4};
    return sizes[static_cast<size_t>(format)];
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
  void setPixel(int32_t x, int32_t y, const glm::vec4 &color) {
    if (inside(x, y)) {
      writePixel(static_cast<size_t>(y) * static_cast<size_t>(width_) +
                     static_cast<size_t>(x),
                 color);
    }
  }
  [[nodiscard]] glm::vec4 getPixel(int32_t x, int32_t y) const {
    return inside(x, y) ? readPixel(static_cast<size_t>(y) *
                                        static_cast<size_t>(width_) +
                                    static_cast<size_t>(x))
                        : glm::vec4(0.0f);
  }
  [[nodiscard]] Bitmap convertEquirectangularMapToCubeMapFaces() const {
    if (type_ != BitmapType::Bitmap2D || width_ < 4 || width_ != 2 * height_ ||
        data_.empty()) {
      return {};
    }
    const int32_t faceSize = width_ / 4;
    Bitmap cube(faceSize, faceSize, 6, components_, format_, memoryResource());
    cube.type_ = BitmapType::BitmapCube;
    constexpr std::array<uint8_t, 6> sourceFaces{3, 1, 4, 5, 2, 0};
    constexpr float pi = 3.14159265358979323846f;
    for (int32_t face = 0; face < 6; ++face) {
      for (int32_t y = 0; y < faceSize; ++y) {
        for (int32_t x = 0; x < faceSize; ++x) {
          const bool reverse = face == 5;
          const glm::vec3 direction = faceDirection(
              reverse ? faceSize - 1 - x : x, reverse ? faceSize - 1 - y : y,
              sourceFaces[face], faceSize);
          const float radius = std::hypot(direction.x, direction.y);
          const float u = 2.0f * static_cast<float>(faceSize) *
                          (std::atan2(direction.y, direction.x) + pi) / pi;
          const float v = 2.0f * static_cast<float>(faceSize) *
                          (pi * 0.5f - std::atan2(direction.z, radius)) / pi;
          const int32_t x0 =
              std::clamp(static_cast<int32_t>(std::floor(u)), 0, width_ - 1);
          const int32_t y0 =
              std::clamp(static_cast<int32_t>(std::floor(v)), 0, height_ - 1);
          const int32_t x1 = std::min(x0 + 1, width_ - 1);
          const int32_t y1 = std::min(y0 + 1, height_ - 1);
          const float tx = u - static_cast<float>(x0);
          const float ty = v - static_cast<float>(y0);
          const glm::vec4 color = getPixel(x0, y0) * (1.0f - tx) * (1.0f - ty) +
                                  getPixel(x1, y0) * tx * (1.0f - ty) +
                                  getPixel(x0, y1) * (1.0f - tx) * ty +
                                  getPixel(x1, y1) * tx * ty;
          cube.writePixel(
              (static_cast<size_t>(face) * faceSize + y) * faceSize + x, color);
        }
      }
    }
    return cube;
  }

private:
  struct FaceTransform {
    glm::vec3 base;
    glm::vec3 x;
    glm::vec3 y;
  };
  [[nodiscard]] static glm::vec3
  faceDirection(int32_t x, int32_t y, uint8_t face, int32_t size) noexcept {
    static const std::array<FaceTransform, 6> transforms{{
        {{-1.0f, -1.0f, -1.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}},
        {{-1.0f, -1.0f, 1.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, -1.0f}},
        {{1.0f, -1.0f, 1.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, -1.0f}},
        {{1.0f, 1.0f, 1.0f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, -1.0f}},
        {{-1.0f, -1.0f, 1.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}},
        {{1.0f, -1.0f, -1.0f}, {0.0f, 1.0f, 0.0f}, {-1.0f, 0.0f, 0.0f}},
    }};
    const float a = 2.0f * static_cast<float>(x) / static_cast<float>(size);
    const float b = 2.0f * static_cast<float>(y) / static_cast<float>(size);
    const FaceTransform &transform = transforms[face];
    return transform.base + transform.x * a + transform.y * b;
  }
  [[nodiscard]] bool inside(int32_t x, int32_t y) const noexcept {
    return x >= 0 && y >= 0 && x < width_ && y < height_;
  }
  [[nodiscard]] size_t bytesPerComponent() const noexcept {
    return static_cast<size_t>(getBytesPerComponent(format_));
  }
  [[nodiscard]] size_t componentOffset(size_t pixel) const noexcept {
    return pixel * static_cast<size_t>(components_);
  }
  void writePixel(size_t pixel, const glm::vec4 &color) {
    const size_t component = componentOffset(pixel);
    if (format_ == BitmapFormat::F32) {
      std::memcpy(data_.data() + component * sizeof(float), &color.x,
                  static_cast<size_t>(components_) * sizeof(float));
      return;
    }
    const glm::vec4 value = glm::clamp(color, glm::vec4(0.0f), glm::vec4(1.0f));
    for (int32_t index = 0; index < components_; ++index) {
      data_[component + static_cast<size_t>(index)] =
          static_cast<uint8_t>(glm::round(value[index] * 255.0f));
    }
  }
  [[nodiscard]] glm::vec4 readPixel(size_t pixel) const {
    const size_t component = componentOffset(pixel);
    glm::vec4 value(0.0f);
    if (format_ == BitmapFormat::F32) {
      std::memcpy(&value.x, data_.data() + component * sizeof(float),
                  static_cast<size_t>(components_) * sizeof(float));
      return value;
    }
    for (int32_t index = 0; index < components_; ++index) {
      value[index] = static_cast<float>(data_[component + index]) / 255.0f;
    }
    return value;
  }
  [[nodiscard]] std::pmr::memory_resource *memoryResource() const noexcept {
    return data_.get_allocator().resource();
  }
  int32_t width_ = 0;
  int32_t height_ = 0;
  int32_t depth_ = 1;
  int32_t components_ = 3;
  BitmapFormat format_ = BitmapFormat::U8;
  BitmapType type_ = BitmapType::Bitmap2D;
  std::pmr::vector<uint8_t> data_{};
};

} // namespace nuri
