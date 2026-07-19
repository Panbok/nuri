#include "nuri/resources/storage/texture/texture_processing.h"
#include "nuri/pch.h"
#include <stb_image_resize2.h>
namespace nuri {
namespace {
[[nodiscard]] uint8_t toByte(float value) {
  return static_cast<uint8_t>(
      std::round(std::clamp(value, 0.0f, 1.0f) * 255.0f));
}
[[nodiscard]] float toUnit(std::byte value) {
  return static_cast<float>(std::to_integer<uint32_t>(value)) / 255.0f;
}
[[nodiscard]] float alphaCutoff(float cutoff) {
  return std::isfinite(cutoff)
             ? std::clamp(cutoff, 1.0f / 255.0f, 254.0f / 255.0f)
             : 0.5f;
}
[[nodiscard]] float alphaCoverage(std::span<const std::byte> rgba, float cutoff,
                                  float scale = 1.0f) {
  uint32_t covered = 0;
  const size_t count = rgba.size() / 4u;
  for (size_t i = 3; i < rgba.size(); i += 4) {
    covered += std::min(toUnit(rgba[i]) * scale, 1.0f) >= cutoff;
  }
  return count ? static_cast<float>(covered) / count : 0.0f;
}
void preserveAlphaCoverage(std::vector<std::byte> &rgba, float cutoff,
                           float target) {
  if (rgba.empty() || target <= 0.0f) {
    return;
  }
  float lo = 0.0f;
  float hi = 1.0f;
  while (alphaCoverage(rgba, cutoff, hi) < target && hi < 16.0f) {
    hi *= 2.0f;
  }
  for (uint32_t i = 0; i < 10; ++i) {
    const float mid = (lo + hi) * 0.5f;
    (alphaCoverage(rgba, cutoff, mid) < target ? lo : hi) = mid;
  }
  for (size_t i = 3; i < rgba.size(); i += 4) {
    rgba[i] = static_cast<std::byte>(toByte(toUnit(rgba[i]) * hi));
  }
}
} // namespace

uint32_t textureMipLevelCount(uint32_t width, uint32_t height) noexcept {
  return std::max(
      1u, static_cast<uint32_t>(std::bit_width(std::max(width, height))));
}

uint32_t textureMipDimension(uint32_t base, uint32_t mip) noexcept {
  return std::max(1u, base >> std::min(mip, 31u));
}

Result<std::vector<std::byte>, std::string>
generateRgba8Mip(std::span<const std::byte> source, uint32_t width,
                 uint32_t height, const TextureLoadOptions &options) {
  const uint32_t dstWidth = std::max(1u, width >> 1u);
  const uint32_t dstHeight = std::max(1u, height >> 1u);
  std::vector<std::byte> output(size_t{dstWidth} * dstHeight * 4u);
  if (options.mipSemantic == TextureMipSemantic::NormalMap) {
    std::vector<float> sourceFloats(size_t{width} * height * 4u);
    std::vector<float> outputFloats(size_t{dstWidth} * dstHeight * 4u);
    for (size_t i = 0; i < size_t{width} * height; ++i) {
      const size_t offset = i * 4u;
      glm::vec3 normal(toUnit(source[offset]) * 2.0f - 1.0f,
                       toUnit(source[offset + 1]) * 2.0f - 1.0f,
                       toUnit(source[offset + 2]) * 2.0f - 1.0f);
      normal = glm::dot(normal, normal) > 1.0e-8f ? glm::normalize(normal)
                                                  : glm::vec3(0, 0, 1);
      sourceFloats[offset] = normal.x;
      sourceFloats[offset + 1] = normal.y;
      sourceFloats[offset + 2] = normal.z;
      sourceFloats[offset + 3] = toUnit(source[offset + 3]);
    }
    if (!stbir_resize_float_linear(sourceFloats.data(), width, height, 0,
                                   outputFloats.data(), dstWidth, dstHeight, 0,
                                   STBIR_RGBA)) {
      return Result<std::vector<std::byte>, std::string>::makeError(
          "failed to resize normal-map mip");
    }
    for (size_t i = 0; i < size_t{dstWidth} * dstHeight; ++i) {
      const size_t offset = i * 4u;
      glm::vec3 normal(outputFloats[offset], outputFloats[offset + 1],
                       outputFloats[offset + 2]);
      normal = glm::dot(normal, normal) > 1.0e-8f ? glm::normalize(normal)
                                                  : glm::vec3(0, 0, 1);
      output[offset] = static_cast<std::byte>(toByte(normal.x * 0.5f + 0.5f));
      output[offset + 1] =
          static_cast<std::byte>(toByte(normal.y * 0.5f + 0.5f));
      output[offset + 2] =
          static_cast<std::byte>(toByte(normal.z * 0.5f + 0.5f));
      output[offset + 3] =
          static_cast<std::byte>(toByte(outputFloats[offset + 3]));
    }
    return Result<std::vector<std::byte>, std::string>::makeResult(
        std::move(output));
  }
  const auto *src = reinterpret_cast<const unsigned char *>(source.data());
  auto *dst = reinterpret_cast<unsigned char *>(output.data());
  const auto resized =
      options.srgb
          ? stbir_resize_uint8_srgb(src, width, height, 0, dst, dstWidth,
                                    dstHeight, 0, STBIR_RGBA)
          : stbir_resize_uint8_linear(src, width, height, 0, dst, dstWidth,
                                      dstHeight, 0, STBIR_RGBA);
  if (!resized) {
    return Result<std::vector<std::byte>, std::string>::makeError(
        "failed to resize texture mip");
  }
  if (options.mipSemantic == TextureMipSemantic::AlphaCoverage) {
    const float cutoff = alphaCutoff(options.alphaCoverageCutoff);
    preserveAlphaCoverage(output, cutoff, alphaCoverage(source, cutoff));
  } else if (options.mipSemantic == TextureMipSemantic::RoughnessG ||
             options.mipSemantic == TextureMipSemantic::RoughnessA) {
    const uint32_t channel =
        options.mipSemantic == TextureMipSemantic::RoughnessA ? 3u : 1u;
    for (uint32_t y = 0; y < dstHeight; ++y) {
      for (uint32_t x = 0; x < dstWidth; ++x) {
        float sum = 0.0f;
        uint32_t count = 0;
        for (uint32_t dy = 0; dy < 2 && y * 2 + dy < height; ++dy) {
          for (uint32_t dx = 0; dx < 2 && x * 2 + dx < width; ++dx) {
            const size_t offset =
                (size_t{y * 2 + dy} * width + x * 2 + dx) * 4u + channel;
            const float roughness = toUnit(source[offset]);
            sum += roughness * roughness;
            ++count;
          }
        }
        output[(size_t{y} * dstWidth + x) * 4u + channel] =
            static_cast<std::byte>(toByte(std::sqrt(sum / count)));
      }
    }
  }
  return Result<std::vector<std::byte>, std::string>::makeResult(
      std::move(output));
}

Result<std::vector<std::byte>, std::string> generateSemanticRgba8MipChain(
    std::span<const std::byte> baseData, uint32_t width, uint32_t height,
    uint32_t mipLevels, const TextureLoadOptions &options) {
  const size_t baseSize = size_t{width} * height * 4u;
  if (baseData.size() < baseSize) {
    return Result<std::vector<std::byte>, std::string>::makeError(
        "base texture mip is truncated");
  }
  size_t totalSize = 0;
  for (uint32_t mip = 0; mip < mipLevels; ++mip) {
    totalSize += size_t{textureMipDimension(width, mip)} *
                 textureMipDimension(height, mip) * 4u;
  }
  std::vector<std::byte> chain;
  chain.reserve(totalSize);
  std::vector<std::byte> current(baseData.begin(), baseData.begin() + baseSize);
  for (uint32_t mip = 0; mip < mipLevels; ++mip) {
    chain.insert(chain.end(), current.begin(), current.end());
    if (mip + 1 == mipLevels) {
      break;
    }
    auto next = generateRgba8Mip(current, width, height, options);
    if (next.hasError()) {
      return next;
    }
    current = std::move(next.value());
    width = std::max(1u, width >> 1u);
    height = std::max(1u, height >> 1u);
  }
  return Result<std::vector<std::byte>, std::string>::makeResult(
      std::move(chain));
}

bool shouldGenerateSemanticRgba8MipChain(const TextureLoadOptions &options,
                                         uint32_t mipLevels) noexcept {
  return options.generateMipmaps && mipLevels > 1u &&
         options.mipSemantic != TextureMipSemantic::Generic;
}

} // namespace nuri
