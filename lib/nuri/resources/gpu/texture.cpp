#include "nuri/resources/gpu/texture.h"

#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
#include "nuri/resources/cpu/bitmap.h"

#include <bit>
#include <cstring>
#include <vector>
#include <stb_image.h>

namespace nuri {
namespace {

[[nodiscard]] uint16_t floatToHalf(float value) {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  const uint16_t sign = static_cast<uint16_t>((bits >> 16u) & 0x8000u);
  const uint32_t exponent = (bits >> 23u) & 0xffu;
  const uint32_t mantissa = bits & 0x007fffffu;

  if (exponent == 0xffu) {
    if (mantissa == 0u) {
      return static_cast<uint16_t>(sign | 0x7c00u);
    }
    return static_cast<uint16_t>(sign | 0x7e00u);
  }

  int32_t halfExponent = static_cast<int32_t>(exponent) - 127 + 15;
  if (halfExponent >= 31) {
    return static_cast<uint16_t>(sign | 0x7c00u);
  }

  if (halfExponent <= 0) {
    if (halfExponent < -10) {
      return sign;
    }

    const uint32_t denormMantissa = mantissa | 0x00800000u;
    const uint32_t shift = static_cast<uint32_t>(14 - halfExponent);
    uint16_t halfMantissa = static_cast<uint16_t>(denormMantissa >> shift);
    if (((denormMantissa >> (shift - 1u)) & 1u) != 0u) {
      halfMantissa = static_cast<uint16_t>(halfMantissa + 1u);
    }
    return static_cast<uint16_t>(sign | halfMantissa);
  }

  uint16_t halfMantissa = static_cast<uint16_t>(mantissa >> 13u);
  uint16_t halfExponentBits =
      static_cast<uint16_t>(static_cast<uint32_t>(halfExponent) << 10u);

  if ((mantissa & 0x00001000u) != 0u) {
    halfMantissa = static_cast<uint16_t>(halfMantissa + 1u);
    if (halfMantissa == 0x0400u) {
      halfMantissa = 0;
      halfExponentBits = static_cast<uint16_t>(halfExponentBits + 0x0400u);
      if (halfExponentBits >= 0x7c00u) {
        return static_cast<uint16_t>(sign | 0x7c00u);
      }
    }
  }

  return static_cast<uint16_t>(sign | halfExponentBits | halfMantissa);
}

[[nodiscard]] std::vector<std::byte>
convertFloatBitmapToHalfBytes(std::span<const uint8_t> srcBytes) {
  if ((srcBytes.size() % sizeof(float)) != 0u) {
    return {};
  }

  const size_t floatCount = srcBytes.size() / sizeof(float);
  std::vector<std::byte> dstBytes(floatCount * sizeof(uint16_t));

  for (size_t i = 0; i < floatCount; ++i) {
    float value = 0.0f;
    std::memcpy(&value, srcBytes.data() + (i * sizeof(float)), sizeof(float));
    const uint16_t half = floatToHalf(value);
    std::memcpy(dstBytes.data() + (i * sizeof(uint16_t)), &half,
                sizeof(uint16_t));
  }

  return dstBytes;
}

} // namespace

Result<std::unique_ptr<Texture>, std::string>
Texture::create(GPUDevice &gpu, const TextureDesc &desc,
                std::string_view debugName) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  auto result = gpu.createTexture(desc, debugName);
  if (result.hasError()) {
    return Result<std::unique_ptr<Texture>, std::string>::makeError(
        result.error());
  }

  return Result<std::unique_ptr<Texture>, std::string>::makeResult(
      std::unique_ptr<Texture>(
          new Texture(result.value(), desc, std::string(debugName))));
}

Result<std::unique_ptr<Texture>, std::string>
Texture::loadTexture(GPUDevice &gpu, const std::string &filePath,
                     std::string_view debugName) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  int32_t width = 0;
  int32_t height = 0;
  int32_t channels = 0;
  void *pixels = stbi_load(filePath.c_str(), &width, &height, &channels, 4);
  if (!pixels) {
    NURI_LOG_WARNING("Texture::loadTexture: Failed to load texture '%s': %s",
                     filePath.c_str(), stbi_failure_reason());
    return Result<std::unique_ptr<Texture>, std::string>::makeError(
        "Failed to load texture from file: " + filePath + " " +
        stbi_failure_reason());
  }

  const size_t dataSize =
      static_cast<size_t>(width) * static_cast<size_t>(height) * 4u;
  const std::span<const std::byte> initialData{
      static_cast<const std::byte *>(pixels), dataSize};

  TextureDesc desc{
      .type = TextureType::Texture2D,
      .format = Format::RGBA8_UNORM,
      .dimensions = {static_cast<uint32_t>(width),
                     static_cast<uint32_t>(height), 1},
      .usage = TextureUsage::Sampled,
      .storage = Storage::Device,
      .numLayers = 1,
      .numSamples = 1,
      .numMipLevels = 1,
      .data = initialData,
      .dataNumMipLevels = 1,
      .generateMipmaps = false,
  };
  auto result = gpu.createTexture(desc, debugName);
  if (result.hasError()) {
    NURI_LOG_WARNING("Texture::loadTexture: Failed to create texture '%s': %s",
                     filePath.c_str(), result.error().c_str());
    stbi_image_free(pixels);
    return Result<std::unique_ptr<Texture>, std::string>::makeError(
        result.error());
  }

  stbi_image_free(pixels);

  NURI_LOG_DEBUG("Texture::loadTexture: Created texture from file '%s'",
                 filePath.c_str());

  return Result<std::unique_ptr<Texture>, std::string>::makeResult(
      std::unique_ptr<Texture>(
          new Texture(result.value(), desc, std::string(debugName))));
}

Result<std::unique_ptr<Texture>, std::string>
Texture::loadCubemapFromEquirectangularHDR(GPUDevice &gpu,
                                           std::string_view filePath,
                                           std::string_view debugName) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  const std::string filePathStr(filePath);
  int32_t width = 0;
  int32_t height = 0;
  int32_t channels = 0;
  float *pixels = stbi_loadf(filePathStr.c_str(), &width, &height, &channels, 4);
  if (!pixels) {
    const char *reason = stbi_failure_reason();
    NURI_LOG_WARNING(
        "Texture::loadCubemapFromEquirectangularHDR: Failed to load '%s': %s",
        filePathStr.c_str(), reason ? reason : "unknown error");
    return Result<std::unique_ptr<Texture>, std::string>::makeError(
        "Failed to load HDR texture from file: " + filePathStr + " " +
        (reason ? std::string(reason) : std::string("unknown error")));
  }

  const Bitmap equirectangular(width, height, 4, BitmapFormat::F32, pixels);
  stbi_image_free(pixels);

  const Bitmap cubemapFaces =
      equirectangular.convertEquirectangularMapToCubeMapFaces();
  if (cubemapFaces.empty()) {
    NURI_LOG_WARNING("Texture::loadCubemapFromEquirectangularHDR: Failed to "
                     "convert equirectangular HDR to cubemap faces '%s'",
                     filePathStr.c_str());
    return Result<std::unique_ptr<Texture>, std::string>::makeError(
        "Failed to convert HDR texture to cubemap faces: " + filePathStr);
  }

  std::vector<std::byte> halfBytes =
      convertFloatBitmapToHalfBytes(cubemapFaces.data());
  if (halfBytes.empty()) {
    NURI_LOG_WARNING("Texture::loadCubemapFromEquirectangularHDR: Failed to "
                     "convert cubemap data to RGBA16F '%s'",
                     filePathStr.c_str());
    return Result<std::unique_ptr<Texture>, std::string>::makeError(
        "Failed to convert cubemap face data to RGBA16F: " + filePathStr);
  }

  const std::span<const std::byte> initialData(halfBytes.data(),
                                                halfBytes.size());

  TextureDesc desc{
      .type = TextureType::TextureCube,
      .format = Format::RGBA16_FLOAT,
      .dimensions = {static_cast<uint32_t>(cubemapFaces.width()),
                     static_cast<uint32_t>(cubemapFaces.height()), 1},
      .usage = TextureUsage::Sampled,
      .storage = Storage::Device,
      .numLayers = 1,
      .numSamples = 1,
      .numMipLevels = 1,
      .data = initialData,
      .dataNumMipLevels = 1,
      .generateMipmaps = false,
  };
  const std::string resolvedDebugName =
      debugName.empty() ? filePathStr : std::string(debugName);
  auto result = gpu.createTexture(desc, resolvedDebugName);
  if (result.hasError()) {
    NURI_LOG_WARNING("Texture::loadCubemapFromEquirectangularHDR: Failed to "
                     "create cubemap texture '%s': %s",
                     filePathStr.c_str(), result.error().c_str());
    return Result<std::unique_ptr<Texture>, std::string>::makeError(
        result.error());
  }

  NURI_LOG_DEBUG("Texture::loadCubemapFromEquirectangularHDR: Created cubemap "
                 "from file '%s'",
                 filePathStr.c_str());

  return Result<std::unique_ptr<Texture>, std::string>::makeResult(
      std::unique_ptr<Texture>(
          new Texture(result.value(), desc, resolvedDebugName)));
}

} // namespace nuri
