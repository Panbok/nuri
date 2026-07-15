#pragma once

#include "nuri/core/result.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/owned_gpu_resource.h"

#include <cstdint>
#include <string_view>

namespace nuri {

enum class TextureMipSemantic : uint8_t {
  Generic = 0,
  AlphaCoverage = 1,
  NormalMap = 2,
  RoughnessG = 3,
  RoughnessA = 4,
};

struct TextureLoadOptions {
  bool srgb = false;
  bool generateMipmaps = false;
  TextureMipSemantic mipSemantic = TextureMipSemantic::Generic;
  float alphaCoverageCutoff = 0.5f;
};

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
  loadTexture(GPUDevice &gpu, std::string_view filePath,
              const TextureLoadOptions &options,
              std::string_view debugName = {});

  [[nodiscard]] static Result<std::unique_ptr<Texture>, std::string>
  loadCubemapFromEquirectangularHDR(GPUDevice &gpu, std::string_view filePath,
                                    std::string_view debugName = {});

  [[nodiscard]] static Result<std::unique_ptr<Texture>, std::string>
  loadTextureKtx2(GPUDevice &gpu, std::string_view filePath,
                  std::string_view debugName = {});

  // `loadPortableTextureKtx2` accepts portable KTX2 payloads such as
  // UASTC/ETC1S with universal supercompression and runtime transcoding
  // controlled by `TextureLoadOptions`; prefer it over `loadTextureKtx2` for
  // device-portable assets, and use `loadTextureKtx2` for native KTX2 data
  // whose sRGB and mipmap behavior is already fixed by the file contents.
  [[nodiscard]] static Result<std::unique_ptr<Texture>, std::string>
  loadPortableTextureKtx2(GPUDevice &gpu, std::string_view filePath,
                          const TextureLoadOptions &options = {},
                          std::string_view debugName = {});

  [[nodiscard]] static Result<std::unique_ptr<Texture>, std::string>
  loadCubemapKtx2(GPUDevice &gpu, std::string_view filePath,
                  std::string_view debugName = {});

  [[nodiscard]] TextureHandle handle() const { return resource_.get(); }
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
  [[nodiscard]] bool valid() const noexcept { return resource_.valid(); }

  // Explicitly transfers native ownership to another move-only owner. The
  // Texture metadata wrapper is invalid after this call.
  [[nodiscard]] TextureHandle release() noexcept { return resource_.release(); }

private:
  Texture(GPUDevice &gpu, TextureHandle handle, const TextureDesc &desc,
          std::string debugName)
      : resource_(gpu, handle), type_(desc.type), format_(desc.format),
        usage_(desc.usage), dimensions_(desc.dimensions),
        storage_(desc.storage), numLayers_(desc.numLayers),
        numSamples_(desc.numSamples), numMipLevels_(desc.numMipLevels),
        generateMipmaps_(desc.generateMipmaps),
        debugName_(std::move(debugName)) {}

  OwnedTextureHandle resource_;
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
