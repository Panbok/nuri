#include "nuri/resources/gpu/texture.h"

#include "nuri/core/log.h"
#include "nuri/core/profiling.h"

#include <stb_image.h>

namespace nuri {

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

  NURI_LOG_DEBUG("Texture::loadTexture: Created texture from file '%.*s'",
                 static_cast<int>(filePath.size()), filePath.c_str());

  return Result<std::unique_ptr<Texture>, std::string>::makeResult(
      std::unique_ptr<Texture>(
          new Texture(result.value(), desc, std::string(debugName))));
}

} // namespace nuri
