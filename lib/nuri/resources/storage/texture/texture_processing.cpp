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

struct NormalMoment {
  glm::vec3 meanNormal{0.0f, 0.0f, 1.0f};
  glm::vec2 meanSlope{0.0f};
  float meanSlopeLength2 = 0.0f;
  float cleanValidWeight = 1.0f;
};

[[nodiscard]] glm::vec3 normalizedOrZ(glm::vec3 value) noexcept {
  const float length2 = glm::dot(value, value);
  // The exact zero vector is not representable in RGB8 signed-normal
  // encoding: (128, 128, 128) decodes to three 1/255 components. Treat that
  // quantization cell as degenerate so malformed/empty normals use +Z.
  return std::isfinite(length2) && length2 > 1.0e-4f
             ? value * glm::inversesqrt(length2)
             : glm::vec3(0.0f, 0.0f, 1.0f);
}

void appendEncodedNormalMoment(std::vector<std::byte> &chain,
                               const NormalMoment &moment, bool baseLevel,
                               NormalVarianceBuildStats *stats) {
  const glm::vec3 normal = normalizedOrZ(moment.meanNormal);
  float variance = 0.0f;
  bool clean = true;
  if (!baseLevel) {
    const float slopeLength2 = glm::dot(moment.meanSlope, moment.meanSlope);
    clean = moment.cleanValidWeight >= 1.0f - kCleanValidWeightEpsilon &&
            std::isfinite(moment.meanSlopeLength2) &&
            std::isfinite(slopeLength2);
    if (clean) {
      variance = 0.5f * std::max(moment.meanSlopeLength2 - slopeLength2, 0.0f);
    } else {
      const float resultant =
          std::clamp(glm::length(moment.meanNormal), 0.0f, 1.0f);
      variance =
          std::max((1.0f - resultant) / std::max(resultant, 1.0e-4f), 0.0f);
    }
  }
  if (!std::isfinite(variance)) {
    clean = false;
    variance = 0.0f;
  }
  variance = std::clamp(variance, 0.0f, kEncodedSlopeVarianceMax);
  if (stats != nullptr) {
    clean ? ++stats->cleanTexels : ++stats->toksvigFallbackTexels;
  }
  chain.push_back(static_cast<std::byte>(toByte(normal.x * 0.5f + 0.5f)));
  chain.push_back(static_cast<std::byte>(toByte(normal.y * 0.5f + 0.5f)));
  chain.push_back(static_cast<std::byte>(toByte(normal.z * 0.5f + 0.5f)));
  chain.push_back(
      static_cast<std::byte>(toByte(variance / kEncodedSlopeVarianceMax)));
}

[[nodiscard]] std::vector<NormalMoment>
reduceNormalMoments(std::span<const NormalMoment> source, uint32_t sourceWidth,
                    uint32_t sourceHeight, uint32_t destinationWidth,
                    uint32_t destinationHeight) {
  std::vector<NormalMoment> output(size_t{destinationWidth} *
                                   destinationHeight);
  for (uint32_t y = 0u; y < destinationHeight; ++y) {
    const double y0 = static_cast<double>(y) * sourceHeight /
                      static_cast<double>(destinationHeight);
    const double y1 = static_cast<double>(y + 1u) * sourceHeight /
                      static_cast<double>(destinationHeight);
    const uint32_t firstY = static_cast<uint32_t>(std::floor(y0));
    const uint32_t lastY =
        std::min(sourceHeight, static_cast<uint32_t>(std::ceil(y1)));
    for (uint32_t x = 0u; x < destinationWidth; ++x) {
      const double x0 = static_cast<double>(x) * sourceWidth /
                        static_cast<double>(destinationWidth);
      const double x1 = static_cast<double>(x + 1u) * sourceWidth /
                        static_cast<double>(destinationWidth);
      const uint32_t firstX = static_cast<uint32_t>(std::floor(x0));
      const uint32_t lastX =
          std::min(sourceWidth, static_cast<uint32_t>(std::ceil(x1)));
      glm::dvec3 meanNormal(0.0);
      glm::dvec2 meanSlope(0.0);
      double meanSlopeLength2 = 0.0;
      double cleanValidWeight = 0.0;
      double totalWeight = 0.0;
      for (uint32_t sourceY = firstY; sourceY < lastY; ++sourceY) {
        const double yWeight =
            std::max(0.0, std::min(y1, static_cast<double>(sourceY + 1u)) -
                              std::max(y0, static_cast<double>(sourceY)));
        for (uint32_t sourceX = firstX; sourceX < lastX; ++sourceX) {
          const double xWeight =
              std::max(0.0, std::min(x1, static_cast<double>(sourceX + 1u)) -
                                std::max(x0, static_cast<double>(sourceX)));
          const double weight = xWeight * yWeight;
          const NormalMoment &sample =
              source[size_t{sourceY} * sourceWidth + sourceX];
          meanNormal += glm::dvec3(sample.meanNormal) * weight;
          meanSlope += glm::dvec2(sample.meanSlope) * weight;
          meanSlopeLength2 += sample.meanSlopeLength2 * weight;
          cleanValidWeight += sample.cleanValidWeight * weight;
          totalWeight += weight;
        }
      }
      const double inverseWeight = totalWeight > 0.0 ? 1.0 / totalWeight : 0.0;
      output[size_t{y} * destinationWidth + x] = NormalMoment{
          .meanNormal = glm::vec3(meanNormal * inverseWeight),
          .meanSlope = glm::vec2(meanSlope * inverseWeight),
          .meanSlopeLength2 =
              static_cast<float>(meanSlopeLength2 * inverseWeight),
          .cleanValidWeight =
              static_cast<float>(cleanValidWeight * inverseWeight),
      };
    }
  }
  return output;
}

[[nodiscard]] Result<std::vector<std::byte>, std::string>
generateCleanNormalMipChain(std::span<const std::byte> baseData, uint32_t width,
                            uint32_t height, uint32_t mipLevels,
                            NormalVarianceBuildStats *stats) {
  const size_t baseTexelCount = size_t{width} * height;
  if (baseData.size() < baseTexelCount * 4u) {
    return Result<std::vector<std::byte>, std::string>::makeError(
        "base texture mip is truncated");
  }
  std::vector<NormalMoment> current(baseTexelCount);
  for (size_t i = 0u; i < baseTexelCount; ++i) {
    const size_t offset = i * 4u;
    const glm::vec3 normal =
        normalizedOrZ(glm::vec3(toUnit(baseData[offset]) * 2.0f - 1.0f,
                                toUnit(baseData[offset + 1u]) * 2.0f - 1.0f,
                                toUnit(baseData[offset + 2u]) * 2.0f - 1.0f));
    const bool clean = normal.z >= kCleanNormalZFloor;
    const glm::vec2 slope =
        clean ? glm::vec2(normal) / normal.z : glm::vec2(0.0f);
    current[i] = NormalMoment{
        .meanNormal = normal,
        .meanSlope = slope,
        .meanSlopeLength2 = clean ? glm::dot(slope, slope) : 0.0f,
        .cleanValidWeight = clean ? 1.0f : 0.0f,
    };
  }
  size_t byteCount = 0u;
  for (uint32_t mip = 0u; mip < mipLevels; ++mip) {
    byteCount += size_t{textureMipDimension(width, mip)} *
                 textureMipDimension(height, mip) * 4u;
  }
  std::vector<std::byte> chain;
  chain.reserve(byteCount);
  uint32_t currentWidth = width;
  uint32_t currentHeight = height;
  for (uint32_t mip = 0u; mip < mipLevels; ++mip) {
    for (const NormalMoment &moment : current) {
      appendEncodedNormalMoment(chain, moment, mip == 0u, stats);
    }
    if (mip + 1u == mipLevels) {
      break;
    }
    const uint32_t nextWidth = std::max(1u, currentWidth / 2u);
    const uint32_t nextHeight = std::max(1u, currentHeight / 2u);
    current = reduceNormalMoments(current, currentWidth, currentHeight,
                                  nextWidth, nextHeight);
    currentWidth = nextWidth;
    currentHeight = nextHeight;
  }
  return Result<std::vector<std::byte>, std::string>::makeResult(
      std::move(chain));
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
    uint32_t mipLevels, const TextureLoadOptions &options,
    TextureContentContract contentContract,
    NormalVarianceBuildStats *varianceStats) {
  if (contentContract == TextureContentContract::NormalRgbCleanVarianceA) {
    if (options.mipSemantic != TextureMipSemantic::NormalMap || options.srgb) {
      return Result<std::vector<std::byte>, std::string>::makeError(
          "normal variance content requires a linear normal-map semantic");
    }
    return generateCleanNormalMipChain(baseData, width, height, mipLevels,
                                       varianceStats);
  }
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
