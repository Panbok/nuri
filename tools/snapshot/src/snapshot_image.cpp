#include "nuri/tools/snapshot/snapshot_image.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

#include <OpenImageIO/imageio.h>

namespace nuri::tools::snapshot {
namespace {

[[nodiscard]] uint32_t channelCountForFormat(Format format) noexcept {
  switch (format) {
  case Format::R8_UNORM:
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

[[nodiscard]] std::vector<uint8_t> makePreviewRgba(const SnapshotImage &image) {
  std::vector<uint8_t> rgba(static_cast<size_t>(image.width) * image.height *
                            4u);
  if (image.values.empty() || image.channelCount == 0u) {
    return rgba;
  }
  float minValue = std::numeric_limits<float>::max();
  float maxValue = std::numeric_limits<float>::lowest();
  float nonClearMinValue = std::numeric_limits<float>::max();
  float nonClearMaxValue = std::numeric_limits<float>::lowest();
  uint32_t nonClearValueCount = 0u;
  if (image.channelCount == 1u) {
    for (const float value : image.values) {
      if (std::isfinite(value)) {
        minValue = std::min(minValue, value);
        maxValue = std::max(maxValue, value);
        if (value < 0.9999f) {
          nonClearMinValue = std::min(nonClearMinValue, value);
          nonClearMaxValue = std::max(nonClearMaxValue, value);
          ++nonClearValueCount;
        }
      }
    }
  }
  const bool useNonClearRange =
      nonClearValueCount > 0u && nonClearMaxValue > nonClearMinValue;
  const float scalarMin = useNonClearRange ? nonClearMinValue : minValue;
  const float scalarMax = useNonClearRange ? nonClearMaxValue : maxValue;
  const float range = scalarMax > scalarMin ? scalarMax - scalarMin : 1.0f;
  float vectorMaxAbs = 0.0f;
  if (image.channelCount == 2u) {
    for (const float value : image.values) {
      if (std::isfinite(value)) {
        vectorMaxAbs = std::max(vectorMaxAbs, std::abs(value));
      }
    }
  }
  const float vectorScale = vectorMaxAbs > 1.0e-6f ? 0.5f / vectorMaxAbs : 1.0f;
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
      const float r = image.values[src + 0u];
      const float g = image.values[src + std::min(1u, image.channelCount - 1u)];
      const float b = image.values[src + std::min(2u, image.channelCount - 1u)];
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
  OIIO::ImageSpec spec(static_cast<int>(image.point.dimensions.width),
                       static_cast<int>(image.point.dimensions.height),
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
  const uint32_t width = std::max(point.dimensions.width >> point.mip, 1u);
  const uint32_t height = std::max(point.dimensions.height >> point.mip, 1u);
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
  const uint32_t width = image.point.dimensions.width;
  const uint32_t height = image.point.dimensions.height;
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
  uint64_t hash = 14695981039346656037ull;
  for (const std::byte byte : bytes) {
    hash ^= static_cast<uint8_t>(byte);
    hash *= 1099511628211ull;
  }
  std::ostringstream out;
  out << "fnv1a64:" << std::hex << std::setw(16) << std::setfill('0') << hash;
  return out.str();
}

Result<bool, std::string>
writeSnapshotArtifacts(const SnapshotReadbackImage &image,
                       const std::filesystem::path &artifactStem,
                       SnapshotArtifactPaths &outPaths) {
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
  }
  {
    std::ofstream file(outPaths.metadata, std::ios::binary);
    if (!file) {
      return Result<bool, std::string>::makeError(
          "writeSnapshotArtifacts: failed to open metadata artifact");
    }
    file << "{\n"
         << "  \"target\": \"" << image.point.name << "\",\n"
         << "  \"format\": " << static_cast<uint32_t>(image.point.format)
         << ",\n"
         << "  \"width\": " << image.point.dimensions.width << ",\n"
         << "  \"height\": " << image.point.dimensions.height << ",\n"
         << "  \"rowStride\": " << image.rowStride << ",\n"
         << "  \"origin\": \"top_left\",\n"
         << "  \"colorSpace\": \"" << image.point.colorSpace << "\",\n"
         << "  \"payload\": \"" << outPaths.raw.filename().generic_string()
         << "\",\n"
         << "  \"hash\": \"" << image.hash << "\"\n"
         << "}\n";
  }
  auto decoded = decodeSnapshotImage(image);
  if (decoded.hasError()) {
    return Result<bool, std::string>::makeError(decoded.error());
  }
  return writeSnapshotPreviewPng(decoded.value(), outPaths.preview);
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

Result<bool, std::string>
writeSnapshotPreviewPng(const SnapshotImage &image,
                        const std::filesystem::path &path) {
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  const std::vector<uint8_t> rgba = makePreviewRgba(image);
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

} // namespace nuri::tools::snapshot
