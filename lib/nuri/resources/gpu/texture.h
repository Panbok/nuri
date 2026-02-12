#pragma once

#include "nuri/core/result.h"
#include "nuri/gfx/gpu_device.h"

#include <string_view>

namespace nuri {

class NURI_API Texture final {
public:
  ~Texture() = default;

  Texture(const Texture &) = delete;
  Texture &operator=(const Texture &) = delete;
  Texture(Texture &&) = delete;
  Texture &operator=(Texture &&) = delete;

  [[nodiscard]] static Result<std::unique_ptr<Texture>, std::string>
  create(GPUDevice &gpu, const TextureDesc &desc,
         std::string_view debugName = {});

  [[nodiscard]] static Result<std::unique_ptr<Texture>, std::string>
  loadTexture(GPUDevice &gpu, std::string_view filePath,
              std::string_view debugName = {});

  [[nodiscard]] static Result<std::unique_ptr<Texture>, std::string>
  loadCubemapFromEquirectangularHDR(GPUDevice &gpu,
                                    std::string_view filePath,
                                    std::string_view debugName = {});

  [[nodiscard]] TextureHandle handle() const { return handle_; }
  [[nodiscard]] TextureType type() const { return type_; }
  [[nodiscard]] Format format() const { return format_; }
  [[nodiscard]] TextureUsage usage() const { return usage_; }
  [[nodiscard]] TextureDimensions dimensions() const { return dimensions_; }
  [[nodiscard]] Storage storage() const { return storage_; }
  [[nodiscard]] uint32_t numLayers() const { return numLayers_; }
  [[nodiscard]] uint32_t numSamples() const { return numSamples_; }
  [[nodiscard]] uint32_t numMipLevels() const { return numMipLevels_; }
  [[nodiscard]] bool generateMipmaps() const { return generateMipmaps_; }
  [[nodiscard]] std::string_view debugName() const noexcept {
    return debugName_;
  }
  [[nodiscard]] bool valid() const noexcept { return nuri::isValid(handle_); }

private:
  Texture(TextureHandle handle, const TextureDesc &desc, std::string debugName)
      : handle_(handle), type_(desc.type), format_(desc.format),
        usage_(desc.usage), dimensions_(desc.dimensions),
        storage_(desc.storage), numLayers_(desc.numLayers),
        numSamples_(desc.numSamples), numMipLevels_(desc.numMipLevels),
        generateMipmaps_(desc.generateMipmaps),
        debugName_(std::move(debugName)) {}

  TextureHandle handle_;
  TextureType type_ = TextureType::Texture2D;
  Format format_ = Format::RGBA8_UNORM;
  TextureUsage usage_ = TextureUsage::Sampled;
  TextureDimensions dimensions_{};
  Storage storage_ = Storage::Device;
  uint32_t numLayers_ = 1;
  uint32_t numSamples_ = 1;
  uint32_t numMipLevels_ = 1;
  bool generateMipmaps_ = false;
  std::string debugName_;
};

} // namespace nuri
