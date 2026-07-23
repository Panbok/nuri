#include "nuri/tools/snapshot/snapshot_image.h"

#include "nuri/tools/core/sha256.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>

#include <OpenImageIO/imageio.h>
#include <yyjson.h>

namespace nuri::tools::snapshot {
namespace {

[[nodiscard]] uint32_t mipDimension(uint32_t dimension, uint32_t mip) noexcept {
  return mip >= 32u ? 1u : std::max(dimension >> mip, 1u);
}

[[nodiscard]] uint32_t channelCountForFormat(Format format) noexcept {
  switch (format) {
  case Format::R8_UNORM:
  case Format::R16_UNORM:
  case Format::R32_UINT:
  case Format::R32_FLOAT:
  case Format::D16_UNORM:
  case Format::D32_FLOAT:
    return 1u;
  case Format::RG16_FLOAT:
  case Format::RG32_FLOAT:
    return 2u;
  case Format::RGBA8_UNORM:
  case Format::RGBA8_SRGB:
  case Format::RGBA8_UINT:
  case Format::RGBA16_FLOAT:
  case Format::RGBA32_FLOAT:
    return 4u;
  case Format::BC7_RGBA_UNORM:
  case Format::BC7_RGBA_SRGB:
  case Format::ETC2_RGB8_UNORM:
  case Format::ETC2_RGB8_SRGB:
  case Format::Count:
    break;
  }
  return 0u;
}

[[nodiscard]] float halfToFloat(uint16_t value) noexcept {
  const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16u;
  uint32_t exponent = (value >> 10u) & 0x1fu;
  uint32_t mantissa = value & 0x03ffu;
  uint32_t bits = 0u;
  if (exponent == 0u) {
    if (mantissa == 0u) {
      bits = sign;
    } else {
      exponent = 1u;
      while ((mantissa & 0x0400u) == 0u) {
        mantissa <<= 1u;
        --exponent;
      }
      mantissa &= 0x03ffu;
      bits = sign | ((exponent + 112u) << 23u) | (mantissa << 13u);
    }
  } else if (exponent == 31u) {
    bits = sign | 0x7f800000u | (mantissa << 13u);
  } else {
    bits = sign | ((exponent + 112u) << 23u) | (mantissa << 13u);
  }
  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

template <typename T>
[[nodiscard]] T loadScalar(const std::byte *bytes) noexcept {
  T out{};
  std::memcpy(&out, bytes, sizeof(T));
  return out;
}

[[nodiscard]] float decodeComponent(Format format, const std::byte *bytes,
                                    uint32_t channel) noexcept {
  switch (format) {
  case Format::R8_UNORM:
    return static_cast<float>(static_cast<uint8_t>(bytes[0])) / 255.0f;
  case Format::R16_UNORM:
    return static_cast<float>(loadScalar<uint16_t>(bytes)) / 65535.0f;
  case Format::RGBA8_UNORM:
  case Format::RGBA8_SRGB:
    return static_cast<float>(static_cast<uint8_t>(bytes[channel])) / 255.0f;
  case Format::RGBA8_UINT:
    return static_cast<float>(static_cast<uint8_t>(bytes[channel]));
  case Format::R32_UINT:
    return static_cast<float>(loadScalar<uint32_t>(bytes));
  case Format::R32_FLOAT:
  case Format::D32_FLOAT:
    return loadScalar<float>(bytes);
  case Format::RG32_FLOAT:
    return loadScalar<float>(bytes + channel * sizeof(float));
  case Format::RGBA32_FLOAT:
    return loadScalar<float>(bytes + channel * sizeof(float));
  case Format::D16_UNORM:
    return static_cast<float>(loadScalar<uint16_t>(bytes)) / 65535.0f;
  case Format::RG16_FLOAT:
    return halfToFloat(
        loadScalar<uint16_t>(bytes + channel * sizeof(uint16_t)));
  case Format::RGBA16_FLOAT:
    return halfToFloat(
        loadScalar<uint16_t>(bytes + channel * sizeof(uint16_t)));
  case Format::BC7_RGBA_UNORM:
  case Format::BC7_RGBA_SRGB:
  case Format::ETC2_RGB8_UNORM:
  case Format::ETC2_RGB8_SRGB:
  case Format::Count:
    break;
  }
  return 0.0f;
}

[[nodiscard]] uint8_t toU8(float value) noexcept {
  const float clamped = std::clamp(value, 0.0f, 1.0f);
  return static_cast<uint8_t>(std::lround(clamped * 255.0f));
}

[[nodiscard]] std::array<float, 3u> hsvToRgb(float hue, float saturation,
                                             float value) noexcept {
  hue = hue - std::floor(hue);
  const float scaled = hue * 6.0f;
  const int32_t sector = static_cast<int32_t>(std::floor(scaled));
  const float fraction = scaled - static_cast<float>(sector);
  const float p = value * (1.0f - saturation);
  const float q = value * (1.0f - saturation * fraction);
  const float t = value * (1.0f - saturation * (1.0f - fraction));
  switch (sector % 6) {
  case 0:
    return {value, t, p};
  case 1:
    return {q, value, p};
  case 2:
    return {p, value, t};
  case 3:
    return {p, q, value};
  case 4:
    return {t, p, value};
  default:
    return {value, p, q};
  }
}

struct SnapshotPreviewScale {
  float minValue = std::numeric_limits<float>::max();
  float maxValue = std::numeric_limits<float>::lowest();
  float nonClearMinValue = std::numeric_limits<float>::max();
  float nonClearMaxValue = std::numeric_limits<float>::lowest();
  uint32_t nonClearValueCount = 0u;
  float vectorMaxAbs = 0.0f;
  float hdrMaxComponent = 0.0f;
};

void extendPreviewScale(const SnapshotImage &image,
                        SnapshotPreviewScale &scale) {
  if (image.channelCount == 1u) {
    for (const float value : image.values) {
      if (std::isfinite(value)) {
        scale.minValue = std::min(scale.minValue, value);
        scale.maxValue = std::max(scale.maxValue, value);
        if (value < 0.9999f) {
          scale.nonClearMinValue = std::min(scale.nonClearMinValue, value);
          scale.nonClearMaxValue = std::max(scale.nonClearMaxValue, value);
          ++scale.nonClearValueCount;
        }
      }
    }
  }
  if (image.channelCount == 2u) {
    for (const float value : image.values) {
      if (std::isfinite(value)) {
        scale.vectorMaxAbs = std::max(scale.vectorMaxAbs, std::abs(value));
      }
    }
  }
  if (image.channelCount >= 3u) {
    const size_t pixelCount = static_cast<size_t>(image.width) * image.height;
    for (size_t pixel = 0u; pixel < pixelCount; ++pixel) {
      const size_t offset = pixel * image.channelCount;
      for (uint32_t channel = 0u; channel < 3u; ++channel) {
        const float value = image.values[offset + channel];
        if (std::isfinite(value)) {
          scale.hdrMaxComponent =
              std::max(scale.hdrMaxComponent, std::max(0.0f, value));
        }
      }
    }
  }
}

[[nodiscard]] SnapshotPreviewScale
makePreviewScale(const SnapshotImage &actual,
                 const SnapshotImage *expected = nullptr) {
  SnapshotPreviewScale scale{};
  extendPreviewScale(actual, scale);
  if (expected != nullptr) {
    extendPreviewScale(*expected, scale);
  }
  return scale;
}

[[nodiscard]] std::vector<uint8_t>
makePreviewRgba(const SnapshotImage &image, const SnapshotPreviewScale &scale,
                std::string_view profile) {
  std::vector<uint8_t> rgba(static_cast<size_t>(image.width) * image.height *
                            4u);
  if (image.values.empty() || image.channelCount == 0u) {
    return rgba;
  }
  const bool useNonClearRange = scale.nonClearValueCount > 0u &&
                                scale.nonClearMaxValue > scale.nonClearMinValue;
  const float scalarMin =
      useNonClearRange ? scale.nonClearMinValue : scale.minValue;
  const float scalarMax =
      useNonClearRange ? scale.nonClearMaxValue : scale.maxValue;
  const float range = scalarMax > scalarMin ? scalarMax - scalarMin : 1.0f;
  const float vectorScale =
      scale.vectorMaxAbs > 1.0e-6f ? 0.5f / scale.vectorMaxAbs : 1.0f;
  const float hdrDenominator = std::log1p(scale.hdrMaxComponent);
  const size_t pixelCount = static_cast<size_t>(image.width) * image.height;
  for (size_t i = 0u; i < pixelCount; ++i) {
    const size_t src = i * image.channelCount;
    const size_t dst = i * 4u;
    if (image.channelCount == 1u) {
      float v = 1.0f;
      if (std::isfinite(image.values[src])) {
        v = scalarMax > scalarMin ? (image.values[src] - scalarMin) / range
                                  : image.values[src];
      }
      rgba[dst + 0u] = toU8(v);
      rgba[dst + 1u] = toU8(v);
      rgba[dst + 2u] = toU8(v);
      rgba[dst + 3u] = 255u;
    } else if (image.channelCount == 2u) {
      const float x =
          std::isfinite(image.values[src + 0u]) ? image.values[src + 0u] : 0.0f;
      const float y =
          std::isfinite(image.values[src + 1u]) ? image.values[src + 1u] : 0.0f;
      const float magnitude = std::sqrt(x * x + y * y);
      if (magnitude <= 1.0e-8f) {
        rgba[dst + 0u] = 0u;
        rgba[dst + 1u] = 0u;
        rgba[dst + 2u] = 0u;
      } else {
        constexpr float kInvTwoPi = 0.15915494309189535f;
        const float hue = std::atan2(y, x) * kInvTwoPi + 1.0f;
        const float value =
            std::clamp(magnitude * vectorScale * 2.0f, 0.18f, 1.0f);
        const std::array<float, 3u> rgb = hsvToRgb(hue, 0.9f, value);
        rgba[dst + 0u] = toU8(rgb[0u]);
        rgba[dst + 1u] = toU8(rgb[1u]);
        rgba[dst + 2u] = toU8(rgb[2u]);
      }
      rgba[dst + 3u] = 255u;
    } else {
      float r = image.values[src + 0u];
      float g = image.values[src + std::min(1u, image.channelCount - 1u)];
      float b = image.values[src + std::min(2u, image.channelCount - 1u)];
      if (profile == "normal") {
        r = r * 0.5f + 0.5f;
        g = g * 0.5f + 0.5f;
        b = b * 0.5f + 0.5f;
      } else if (profile == "hdr_color" && hdrDenominator > 0.0f) {
        r = std::log1p(std::max(0.0f, r)) / hdrDenominator;
        g = std::log1p(std::max(0.0f, g)) / hdrDenominator;
        b = std::log1p(std::max(0.0f, b)) / hdrDenominator;
      }
      rgba[dst + 0u] = toU8(std::isfinite(r) ? r : 1.0f);
      rgba[dst + 1u] = toU8(std::isfinite(g) ? g : 0.0f);
      rgba[dst + 2u] = toU8(std::isfinite(b) ? b : 1.0f);
      rgba[dst + 3u] =
          image.channelCount > 3u ? toU8(image.values[src + 3u]) : 255u;
    }
  }
  return rgba;
}

enum class SnapshotArtifactPayloadKind {
  Png,
  Exr,
  Raw,
};

[[nodiscard]] SnapshotArtifactPayloadKind
payloadKindForFormat(Format format) noexcept {
  switch (format) {
  case Format::R8_UNORM:
  case Format::RGBA8_UNORM:
  case Format::RGBA8_SRGB:
  case Format::RGBA8_UINT:
    return SnapshotArtifactPayloadKind::Png;
  case Format::R16_UNORM:
  case Format::D16_UNORM:
  case Format::R32_FLOAT:
  case Format::D32_FLOAT:
  case Format::RG16_FLOAT:
  case Format::RG32_FLOAT:
  case Format::RGBA16_FLOAT:
  case Format::RGBA32_FLOAT:
    return SnapshotArtifactPayloadKind::Exr;
  case Format::R32_UINT:
  case Format::BC7_RGBA_UNORM:
  case Format::BC7_RGBA_SRGB:
  case Format::ETC2_RGB8_UNORM:
  case Format::ETC2_RGB8_SRGB:
  case Format::Count:
    break;
  }
  return SnapshotArtifactPayloadKind::Raw;
}

[[nodiscard]] std::string_view
payloadExtension(SnapshotArtifactPayloadKind kind) noexcept {
  switch (kind) {
  case SnapshotArtifactPayloadKind::Png:
    return ".png";
  case SnapshotArtifactPayloadKind::Exr:
    return ".exr";
  case SnapshotArtifactPayloadKind::Raw:
    return ".nuri_tex";
  }
  return ".nuri_tex";
}

[[nodiscard]] Result<bool, std::string>
writeOiioFloatImage(const SnapshotImage &image,
                    const std::filesystem::path &path,
                    std::string_view colorSpace) {
  OIIO::ImageSpec spec(static_cast<int>(image.width),
                       static_cast<int>(image.height),
                       static_cast<int>(image.channelCount), OIIO::TypeFloat);
  if (!colorSpace.empty()) {
    spec.set_colorspace(std::string(colorSpace));
  }
  OIIO::ImageOutput::unique_ptr output =
      OIIO::ImageOutput::create(path.string());
  if (!output) {
    return Result<bool, std::string>::makeError(
        "writeOiioFloatImage: failed to create " + path.string() + ": " +
        OIIO::geterror());
  }
  if (!output->open(path.string(), spec)) {
    return Result<bool, std::string>::makeError(
        "writeOiioFloatImage: failed to open " + path.string() + ": " +
        output->geterror());
  }
  if (!output->write_image(OIIO::TypeFloat, image.values.data())) {
    return Result<bool, std::string>::makeError(
        "writeOiioFloatImage: failed to write " + path.string() + ": " +
        output->geterror());
  }
  output->close();
  return Result<bool, std::string>::makeResult(true);
}

[[nodiscard]] Result<bool, std::string>
writeOiioU8Image(const SnapshotReadbackImage &image,
                 const std::filesystem::path &path) {
  const uint32_t channels = channelCountForFormat(image.point.format);
  if (channels == 0u) {
    return Result<bool, std::string>::makeError(
        "writeOiioU8Image: unsupported channel count");
  }
  OIIO::ImageSpec spec(static_cast<int>(mipDimension(
                           image.point.dimensions.width, image.point.mip)),
                       static_cast<int>(mipDimension(
                           image.point.dimensions.height, image.point.mip)),
                       static_cast<int>(channels), OIIO::TypeUInt8);
  if (!image.point.colorSpace.empty()) {
    spec.set_colorspace(std::string(image.point.colorSpace));
  }
  OIIO::ImageOutput::unique_ptr output =
      OIIO::ImageOutput::create(path.string());
  if (!output) {
    return Result<bool, std::string>::makeError(
        "writeOiioU8Image: failed to create " + path.string() + ": " +
        OIIO::geterror());
  }
  if (!output->open(path.string(), spec)) {
    return Result<bool, std::string>::makeError(
        "writeOiioU8Image: failed to open " + path.string() + ": " +
        output->geterror());
  }
  if (!output->write_image(OIIO::TypeUInt8, image.bytes.data())) {
    return Result<bool, std::string>::makeError(
        "writeOiioU8Image: failed to write " + path.string() + ": " +
        output->geterror());
  }
  output->close();
  return Result<bool, std::string>::makeResult(true);
}

} // namespace

size_t snapshotFormatBytesPerPixel(Format format) noexcept {
  switch (format) {
  case Format::R8_UNORM:
    return 1u;
  case Format::R16_UNORM:
  case Format::D16_UNORM:
    return 2u;
  case Format::R32_UINT:
  case Format::R32_FLOAT:
  case Format::D32_FLOAT:
  case Format::RG16_FLOAT:
  case Format::RGBA8_UNORM:
  case Format::RGBA8_SRGB:
  case Format::RGBA8_UINT:
    return 4u;
  case Format::RG32_FLOAT:
  case Format::RGBA16_FLOAT:
    return 8u;
  case Format::RGBA32_FLOAT:
    return 16u;
  case Format::BC7_RGBA_UNORM:
  case Format::BC7_RGBA_SRGB:
  case Format::ETC2_RGB8_UNORM:
  case Format::ETC2_RGB8_SRGB:
  case Format::Count:
    break;
  }
  return 0u;
}

Result<SnapshotReadbackImage, std::string>
readSnapshotCapture(GPUDevice &gpu, const RenderCapturePoint &point) {
  const size_t bpp = snapshotFormatBytesPerPixel(point.format);
  if (bpp == 0u) {
    return Result<SnapshotReadbackImage, std::string>::makeError(
        "unsupported capture format");
  }
  const uint32_t width = mipDimension(point.dimensions.width, point.mip);
  const uint32_t height = mipDimension(point.dimensions.height, point.mip);
  const size_t rowStride = static_cast<size_t>(width) * bpp;
  std::vector<std::byte> bytes(rowStride * static_cast<size_t>(height));
  const TextureReadbackRegion region{
      .x = 0u,
      .y = 0u,
      .width = width,
      .height = height,
      .mipLevel = point.mip,
      .layer = point.layer,
  };
  auto readResult = gpu.readTexture(point.texture, region, bytes);
  if (readResult.hasError()) {
    return Result<SnapshotReadbackImage, std::string>::makeError(
        readResult.error());
  }
  SnapshotReadbackImage out{};
  out.point = point;
  out.bytes = std::move(bytes);
  out.rowStride = rowStride;
  out.hash = snapshotHashBytes(out.bytes);
  return Result<SnapshotReadbackImage, std::string>::makeResult(std::move(out));
}

Result<SnapshotImage, std::string>
decodeSnapshotImage(const SnapshotReadbackImage &image) {
  const uint32_t channels = channelCountForFormat(image.point.format);
  const size_t bpp = snapshotFormatBytesPerPixel(image.point.format);
  if (channels == 0u || bpp == 0u) {
    return Result<SnapshotImage, std::string>::makeError(
        "decodeSnapshotImage: unsupported format");
  }
  const uint32_t width =
      mipDimension(image.point.dimensions.width, image.point.mip);
  const uint32_t height =
      mipDimension(image.point.dimensions.height, image.point.mip);
  const size_t minimumRowStride = static_cast<size_t>(width) * bpp;
  const size_t requiredBytes = image.rowStride * static_cast<size_t>(height);
  if (image.rowStride < minimumRowStride ||
      requiredBytes > image.bytes.size()) {
    return Result<SnapshotImage, std::string>::makeError(
        "decodeSnapshotImage: readback payload is smaller than mip dimensions");
  }
  SnapshotImage out{};
  out.width = width;
  out.height = height;
  out.channelCount = channels;
  out.values.resize(static_cast<size_t>(width) * height * channels);
  for (uint32_t y = 0u; y < height; ++y) {
    for (uint32_t x = 0u; x < width; ++x) {
      const std::byte *pixel = image.bytes.data() +
                               static_cast<size_t>(y) * image.rowStride +
                               static_cast<size_t>(x) * bpp;
      const size_t dst =
          (static_cast<size_t>(y) * width + static_cast<size_t>(x)) * channels;
      for (uint32_t c = 0u; c < channels; ++c) {
        out.values[dst + c] = decodeComponent(image.point.format, pixel, c);
      }
    }
  }
  return Result<SnapshotImage, std::string>::makeResult(std::move(out));
}

std::string snapshotHashBytes(std::span<const std::byte> bytes) {
  return "sha256:" + nuri::tools::core::sha256Hex(bytes);
}

Result<bool, std::string>
writeSnapshotArtifacts(const SnapshotReadbackImage &image,
                       const std::filesystem::path &artifactStem,
                       SnapshotArtifactPaths &outPaths,
                       std::string_view compareProfile) {
  if (!artifactStem.parent_path().empty()) {
    std::filesystem::create_directories(artifactStem.parent_path());
  }
  const SnapshotArtifactPayloadKind payloadKind =
      payloadKindForFormat(image.point.format);
  outPaths.raw = artifactStem;
  outPaths.raw += payloadExtension(payloadKind);
  outPaths.metadata = artifactStem;
  outPaths.metadata += ".json";
  outPaths.preview = artifactStem;
  outPaths.preview += "_preview.png";

  if (payloadKind == SnapshotArtifactPayloadKind::Png) {
    auto written = writeOiioU8Image(image, outPaths.raw);
    if (written.hasError()) {
      return written;
    }
  } else if (payloadKind == SnapshotArtifactPayloadKind::Exr) {
    auto decoded = decodeSnapshotImage(image);
    if (decoded.hasError()) {
      return Result<bool, std::string>::makeError(decoded.error());
    }
    auto written = writeOiioFloatImage(decoded.value(), outPaths.raw,
                                       image.point.colorSpace);
    if (written.hasError()) {
      return written;
    }
  } else {
    std::ofstream file(outPaths.raw, std::ios::binary);
    if (!file) {
      return Result<bool, std::string>::makeError(
          "writeSnapshotArtifacts: failed to open raw fallback artifact");
    }
    file.write(reinterpret_cast<const char *>(image.bytes.data()),
               static_cast<std::streamsize>(image.bytes.size()));
    file.close();
    if (!file) {
      return Result<bool, std::string>::makeError(
          "writeSnapshotArtifacts: failed to write raw fallback artifact");
    }
  }
  {
    std::ofstream file(outPaths.metadata, std::ios::binary);
    if (!file) {
      return Result<bool, std::string>::makeError(
          "writeSnapshotArtifacts: failed to open metadata artifact");
    }
    file << "{\n"
         << "  \"target\": \"" << image.point.name << "\",\n"
         << "  \"capturePointVersion\": " << image.point.version << ",\n"
         << "  \"kind\": \"" << renderCaptureValueKindName(image.point.kind)
         << "\",\n"
         << "  \"format\": " << static_cast<uint32_t>(image.point.format)
         << ",\n"
         << "  \"width\": "
         << mipDimension(image.point.dimensions.width, image.point.mip) << ",\n"
         << "  \"height\": "
         << mipDimension(image.point.dimensions.height, image.point.mip)
         << ",\n"
         << "  \"mip\": " << image.point.mip << ",\n"
         << "  \"layer\": " << image.point.layer << ",\n"
         << "  \"rowStride\": " << image.rowStride << ",\n"
         << "  \"origin\": \"top_left\",\n"
         << "  \"colorSpace\": \"" << image.point.colorSpace << "\",\n"
         << "  \"profile\": \""
         << (compareProfile.empty() ? image.point.defaultCompareProfile
                                    : compareProfile)
         << "\",\n"
         << "  \"payload\": \"" << outPaths.raw.filename().generic_string()
         << "\",\n"
         << "  \"hash\": \"" << image.hash << "\",\n"
         << "  \"ddgi\": {\n"
         << "    \"valid\": " << image.point.ddgiMetadata.valid << ",\n"
         << "    \"effectiveKind\": " << image.point.ddgiMetadata.effectiveKind
         << ",\n"
         << "    \"effectiveKeyHash\": "
         << image.point.ddgiMetadata.effectiveKeyHash << ",\n"
         << "    \"cascadeIndex\": " << image.point.ddgiMetadata.cascadeIndex
         << ",\n"
         << "    \"coverageGeneration\": "
         << image.point.ddgiMetadata.coverageGeneration << ",\n"
         << "    \"layoutGeneration\": "
         << image.point.ddgiMetadata.layoutGeneration << ",\n"
         << "    \"resourceGeneration\": "
         << image.point.ddgiMetadata.resourceGeneration << ",\n"
         << "    \"sceneBoundsGeneration\": "
         << image.point.ddgiMetadata.sceneBoundsGeneration << ",\n"
         << "    \"ringOrigin\": [" << image.point.ddgiMetadata.ringOrigin.x
         << ", " << image.point.ddgiMetadata.ringOrigin.y << ", "
         << image.point.ddgiMetadata.ringOrigin.z << "],\n"
         << "    \"cameraCell\": [" << image.point.ddgiMetadata.cameraCell.x
         << ", " << image.point.ddgiMetadata.cameraCell.y << ", "
         << image.point.ddgiMetadata.cameraCell.z << "],\n"
         << "    \"requestedHalfExtents\": ["
         << image.point.ddgiMetadata.requestedHalfExtents.x << ", "
         << image.point.ddgiMetadata.requestedHalfExtents.y << ", "
         << image.point.ddgiMetadata.requestedHalfExtents.z << "],\n"
         << "    \"achievedHalfExtents\": ["
         << image.point.ddgiMetadata.achievedHalfExtents.x << ", "
         << image.point.ddgiMetadata.achievedHalfExtents.y << ", "
         << image.point.ddgiMetadata.achievedHalfExtents.z << "],\n"
         << "    \"fadeStartHalfExtents\": ["
         << image.point.ddgiMetadata.fadeStartHalfExtents.x << ", "
         << image.point.ddgiMetadata.fadeStartHalfExtents.y << ", "
         << image.point.ddgiMetadata.fadeStartHalfExtents.z << "],\n"
         << "    \"fadeEndHalfExtents\": ["
         << image.point.ddgiMetadata.fadeEndHalfExtents.x << ", "
         << image.point.ddgiMetadata.fadeEndHalfExtents.y << ", "
         << image.point.ddgiMetadata.fadeEndHalfExtents.z << "],\n"
         << "    \"transitionCells\": "
         << image.point.ddgiMetadata.transitionCells << "\n"
         << "  }\n"
         << "}\n";
    file.close();
    if (!file) {
      return Result<bool, std::string>::makeError(
          "writeSnapshotArtifacts: failed to write metadata artifact");
    }
  }
  auto decoded = decodeSnapshotImage(image);
  if (decoded.hasError()) {
    return Result<bool, std::string>::makeError(decoded.error());
  }
  return writeSnapshotPreviewPng(decoded.value(), outPaths.preview);
}

Result<SnapshotArtifactMetadata, std::string>
readSnapshotArtifactMetadata(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return Result<SnapshotArtifactMetadata, std::string>::makeError(
        "readSnapshotArtifactMetadata: failed to open " + path.string());
  }
  std::string json((std::istreambuf_iterator<char>(file)),
                   std::istreambuf_iterator<char>());
  using JsonDoc = std::unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)>;
  yyjson_read_err error{};
  JsonDoc doc(yyjson_read_opts(json.data(), json.size(), 0, nullptr, &error),
              &yyjson_doc_free);
  if (!doc || !yyjson_is_obj(yyjson_doc_get_root(doc.get()))) {
    return Result<SnapshotArtifactMetadata, std::string>::makeError(
        "readSnapshotArtifactMetadata: invalid JSON in " + path.string());
  }
  yyjson_val *root = yyjson_doc_get_root(doc.get());
  const auto stringValue = [&](const char *key) {
    yyjson_val *value = yyjson_obj_get(root, key);
    return yyjson_is_str(value)
               ? std::string(yyjson_get_str(value), yyjson_get_len(value))
               : std::string{};
  };
  const auto u32Value = [&](const char *key) {
    yyjson_val *value = yyjson_obj_get(root, key);
    return yyjson_is_uint(value) && yyjson_get_uint(value) <= UINT32_MAX
               ? static_cast<uint32_t>(yyjson_get_uint(value))
               : 0u;
  };
  SnapshotArtifactMetadata metadata{};
  metadata.target = stringValue("target");
  metadata.capturePointVersion = u32Value("capturePointVersion");
  metadata.kind = stringValue("kind");
  yyjson_val *format = yyjson_obj_get(root, "format");
  if (yyjson_is_uint(format) && yyjson_get_uint(format) <= UINT32_MAX) {
    metadata.format = std::to_string(yyjson_get_uint(format));
  }
  metadata.width = u32Value("width");
  metadata.height = u32Value("height");
  metadata.mip = u32Value("mip");
  metadata.layer = u32Value("layer");
  metadata.colorSpace = stringValue("colorSpace");
  metadata.origin = stringValue("origin");
  metadata.profile = stringValue("profile");
  metadata.payload = stringValue("payload");
  metadata.hash = stringValue("hash");
  if (yyjson_val *ddgi = yyjson_obj_get(root, "ddgi"); yyjson_is_obj(ddgi)) {
    const auto ddgiU32 = [ddgi](const char *key) {
      yyjson_val *value = yyjson_obj_get(ddgi, key);
      return yyjson_is_uint(value) && yyjson_get_uint(value) <= UINT32_MAX
                 ? static_cast<uint32_t>(yyjson_get_uint(value))
                 : 0u;
    };
    const auto ddgiU64 = [ddgi](const char *key) {
      yyjson_val *value = yyjson_obj_get(ddgi, key);
      return yyjson_is_uint(value) ? yyjson_get_uint(value) : 0u;
    };
    const auto readUvec3 = [ddgi](const char *key) {
      glm::uvec3 result{0u};
      yyjson_val *array = yyjson_obj_get(ddgi, key);
      if (!yyjson_is_arr(array) || yyjson_arr_size(array) != 3u) {
        return result;
      }
      for (size_t index = 0u; index < 3u; ++index) {
        yyjson_val *value = yyjson_arr_get(array, index);
        if (yyjson_is_uint(value) && yyjson_get_uint(value) <= UINT32_MAX) {
          result[index] = static_cast<uint32_t>(yyjson_get_uint(value));
        }
      }
      return result;
    };
    const auto readIvec3 = [ddgi](const char *key) {
      glm::ivec3 result{0};
      yyjson_val *array = yyjson_obj_get(ddgi, key);
      if (!yyjson_is_arr(array) || yyjson_arr_size(array) != 3u) {
        return result;
      }
      for (size_t index = 0u; index < 3u; ++index) {
        yyjson_val *value = yyjson_arr_get(array, index);
        if (yyjson_is_int(value) && yyjson_get_sint(value) >= INT32_MIN &&
            yyjson_get_sint(value) <= INT32_MAX) {
          result[index] = static_cast<int32_t>(yyjson_get_sint(value));
        }
      }
      return result;
    };
    const auto readVec3 = [ddgi](const char *key) {
      glm::vec3 result{0.0f};
      yyjson_val *array = yyjson_obj_get(ddgi, key);
      if (!yyjson_is_arr(array) || yyjson_arr_size(array) != 3u) {
        return result;
      }
      for (size_t index = 0u; index < 3u; ++index) {
        yyjson_val *value = yyjson_arr_get(array, index);
        if (yyjson_is_num(value)) {
          result[index] = static_cast<float>(yyjson_get_num(value));
        }
      }
      return result;
    };
    metadata.ddgiMetadata = DDGICaptureMetadata{
        .effectiveKeyHash = ddgiU64("effectiveKeyHash"),
        .coverageGeneration = ddgiU64("coverageGeneration"),
        .layoutGeneration = ddgiU64("layoutGeneration"),
        .resourceGeneration = ddgiU64("resourceGeneration"),
        .sceneBoundsGeneration = ddgiU64("sceneBoundsGeneration"),
        .effectiveKind = ddgiU32("effectiveKind"),
        .cascadeIndex = ddgiU32("cascadeIndex"),
        .ringOrigin = readUvec3("ringOrigin"),
        .cameraCell = readIvec3("cameraCell"),
        .requestedHalfExtents = readVec3("requestedHalfExtents"),
        .achievedHalfExtents = readVec3("achievedHalfExtents"),
        .fadeStartHalfExtents = readVec3("fadeStartHalfExtents"),
        .fadeEndHalfExtents = readVec3("fadeEndHalfExtents"),
        .transitionCells = ddgiU32("transitionCells"),
        .valid = ddgiU32("valid"),
    };
  }
  if (metadata.target.empty() || metadata.capturePointVersion == 0u ||
      metadata.kind.empty() || metadata.format.empty() ||
      metadata.width == 0u || metadata.height == 0u ||
      metadata.origin.empty() || metadata.profile.empty() ||
      metadata.payload.empty() || metadata.hash.empty()) {
    return Result<SnapshotArtifactMetadata, std::string>::makeError(
        "readSnapshotArtifactMetadata: incomplete descriptor in " +
        path.string());
  }
  return Result<SnapshotArtifactMetadata, std::string>::makeResult(
      std::move(metadata));
}

Result<SnapshotImage, std::string>
readSnapshotImageFile(const std::filesystem::path &path) {
  OIIO::ImageInput::unique_ptr input = OIIO::ImageInput::open(path.string());
  if (!input) {
    return Result<SnapshotImage, std::string>::makeError(
        "readSnapshotImageFile: failed to open " + path.string() + ": " +
        OIIO::geterror());
  }
  const OIIO::ImageSpec spec = input->spec(0, 0);
  if (spec.width <= 0 || spec.height <= 0 || spec.nchannels <= 0) {
    return Result<SnapshotImage, std::string>::makeError(
        "readSnapshotImageFile: invalid image dimensions for " + path.string());
  }
  SnapshotImage out{};
  out.width = static_cast<uint32_t>(spec.width);
  out.height = static_cast<uint32_t>(spec.height);
  out.channelCount = static_cast<uint32_t>(spec.nchannels);
  out.values.resize(static_cast<size_t>(out.width) * out.height *
                    out.channelCount);
  const bool ok = input->read_image(0, 0, 0, spec.nchannels, OIIO::TypeFloat,
                                    out.values.data());
  if (!ok) {
    return Result<SnapshotImage, std::string>::makeError(
        "readSnapshotImageFile: failed to read " + path.string() + ": " +
        input->geterror());
  }
  input->close();
  return Result<SnapshotImage, std::string>::makeResult(std::move(out));
}

Result<bool, std::string> writePreviewPng(const SnapshotImage &image,
                                          const SnapshotPreviewScale &scale,
                                          std::string_view profile,
                                          const std::filesystem::path &path) {
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  const std::vector<uint8_t> rgba = makePreviewRgba(image, scale, profile);
  OIIO::ImageSpec spec(static_cast<int>(image.width),
                       static_cast<int>(image.height), 4, OIIO::TypeUInt8);
  spec.set_colorspace("sRGB");
  OIIO::ImageOutput::unique_ptr output =
      OIIO::ImageOutput::create(path.string());
  if (!output) {
    return Result<bool, std::string>::makeError(
        "writeSnapshotPreviewPng: failed to create " + path.string() + ": " +
        OIIO::geterror());
  }
  if (!output->open(path.string(), spec)) {
    return Result<bool, std::string>::makeError(
        "writeSnapshotPreviewPng: failed to open " + path.string() + ": " +
        output->geterror());
  }
  const bool ok = output->write_image(OIIO::TypeUInt8, rgba.data());
  if (!ok) {
    return Result<bool, std::string>::makeError(
        "writeSnapshotPreviewPng: failed to write " + path.string() + ": " +
        output->geterror());
  }
  output->close();
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
writeSnapshotPreviewPng(const SnapshotImage &image,
                        const std::filesystem::path &path) {
  return writePreviewPng(image, makePreviewScale(image), {}, path);
}

Result<bool, std::string> writeSnapshotComparisonPreviews(
    const SnapshotImage &actual, const SnapshotImage &expected,
    std::string_view profile, const std::filesystem::path &actualPath,
    const std::filesystem::path &expectedPath) {
  if (actual.width != expected.width || actual.height != expected.height ||
      actual.channelCount != expected.channelCount) {
    return Result<bool, std::string>::makeError(
        "writeSnapshotComparisonPreviews: incompatible images");
  }
  const SnapshotPreviewScale scale = makePreviewScale(actual, &expected);
  auto written = writePreviewPng(actual, scale, profile, actualPath);
  if (written.hasError()) {
    return written;
  }
  written = writePreviewPng(expected, scale, profile, expectedPath);
  if (written.hasError()) {
    return written;
  }
  return Result<bool, std::string>::makeResult(true);
}

} // namespace nuri::tools::snapshot
