#include "nuri/resources/gpu/texture.h"

#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
#include "nuri/resources/cpu/bitmap.h"

#include "nuri/pch.h"

#include <stb_image.h>

namespace nuri {
namespace {

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
    const uint16_t half = static_cast<uint16_t>(glm::packHalf1x16(value));
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
Texture::loadTexture(GPUDevice &gpu, std::string_view filePath,
                     std::string_view debugName) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  const std::string filePathStr(filePath);
  int32_t width = 0;
  int32_t height = 0;
  int32_t channels = 0;
  void *pixels = stbi_load(filePathStr.c_str(), &width, &height, &channels, 4);
  if (!pixels) {
    NURI_LOG_WARNING("Texture::loadTexture: Failed to load texture '%s': %s",
                     filePathStr.c_str(), stbi_failure_reason());
    return Result<std::unique_ptr<Texture>, std::string>::makeError(
        "Failed to load texture from file: " + filePathStr + " " +
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
                     filePathStr.c_str(), result.error().c_str());
    stbi_image_free(pixels);
    return Result<std::unique_ptr<Texture>, std::string>::makeError(
        result.error());
  }

  stbi_image_free(pixels);

  NURI_LOG_DEBUG("Texture::loadTexture: Created texture from file '%s'",
                 filePathStr.c_str());

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
  float *pixels =
      stbi_loadf(filePathStr.c_str(), &width, &height, &channels, 4);
  if (!pixels) {
    const char *reason = stbi_failure_reason();
    NURI_LOG_WARNING(
        "Texture::loadCubemapFromEquirectangularHDR: Failed to load '%s': %s",
        filePathStr.c_str(), reason ? reason : "unknown error");
    return Result<std::unique_ptr<Texture>, std::string>::makeError(
        "Failed to load HDR texture from file: " + filePathStr + " " +
        (reason ? std::string(reason) : std::string("unknown error")));
  }

  // Bitmap copies pixel data via memcpy; safe to free stbi buffer after
  // construction.
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
