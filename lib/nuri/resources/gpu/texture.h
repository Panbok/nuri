#pragma once
#include "nuri/core/result.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/owned_gpu_resource.h"
#include "nuri/resources/storage/texture/texture_processing.h"
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
namespace nuri {

struct TextureCacheTelemetry {
  uint64_t nativeHits = 0u;
  uint64_t nativeMisses = 0u;
  uint64_t nativeStale = 0u;
  uint64_t nativeCorrupt = 0u;
  uint64_t nativeWrites = 0u;
  uint64_t nativeWriteFailures = 0u;
  uint64_t artifactBuilds = 0u;
  uint64_t authoredSourceBytesRead = 0u;
  uint64_t nativeArtifactBytesRead = 0u;
  uint64_t ddsSourceBytesRead = 0u;
  uint64_t artifactBuildTimeNs = 0u;
  uint64_t ddsReadTimeNs = 0u;
};

struct NURI_API PreparedTextureData {
  TextureDesc createDesc{};
  std::vector<std::byte> bytes{};
  std::string debugName{};
  [[nodiscard]] TextureDesc descriptor() const noexcept {
    TextureDesc desc = createDesc;
    desc.data = std::span<const std::byte>(bytes.data(), bytes.size());
    return desc;
  }
  [[nodiscard]] uint64_t uploadBytes() const noexcept {
    return static_cast<uint64_t>(bytes.size());
  }
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
  createPrepared(GPUDevice &gpu, PreparedTextureData data);
  [[nodiscard]] static std::unique_ptr<Texture>
  adoptPrepared(GPUDevice &gpu, TextureHandle handle, const TextureDesc &desc,
                std::string debugName);
  [[nodiscard]] static Result<PreparedTextureData, std::string>
  prepareTexture(std::string_view filePath,
                 const TextureLoadOptions &options = {},
                 std::string_view debugName = {});
  [[nodiscard]] static Result<PreparedTextureData, std::string>
  prepareDdsTexture(std::span<const std::byte> fileBytes,
                    std::string_view sourceName,
                    std::string_view debugName = {});
  [[nodiscard]] static Result<PreparedTextureData, std::string>
  prepareCubemapFromEquirectangularHDR(std::string_view filePath,
                                       std::string_view debugName = {});
  [[nodiscard]] static Result<PreparedTextureData, std::string>
  prepareTextureKtx2(std::string_view filePath,
                     std::string_view debugName = {});
  [[nodiscard]] static Result<PreparedTextureData, std::string>
  prepareCubemapKtx2(std::string_view filePath,
                     std::string_view debugName = {});
  [[nodiscard]] static Result<std::unique_ptr<Texture>, std::string>
  loadTexture(GPUDevice &gpu, std::string_view filePath,
              std::string_view debugName = {});
  [[nodiscard]] static Result<std::unique_ptr<Texture>, std::string>
  loadTexture(GPUDevice &gpu, std::string_view filePath,
              const TextureLoadOptions &options,
              std::string_view debugName = {});
  [[nodiscard]] static Result<std::unique_ptr<Texture>, std::string>
  loadDdsTexture(GPUDevice &gpu, std::span<const std::byte> fileBytes,
                 std::string_view sourceName, std::string_view debugName = {});
  [[nodiscard]] static Result<std::unique_ptr<Texture>, std::string>
  loadCubemapFromEquirectangularHDR(GPUDevice &gpu, std::string_view filePath,
                                    std::string_view debugName = {});
  [[nodiscard]] static Result<std::unique_ptr<Texture>, std::string>
  loadTextureKtx2(GPUDevice &gpu, std::string_view filePath,
                  std::string_view debugName = {});
  [[nodiscard]] static Result<std::unique_ptr<Texture>, std::string>
  loadCubemapKtx2(GPUDevice &gpu, std::string_view filePath,
                  std::string_view debugName = {});
  [[nodiscard]] static TextureCacheTelemetry cacheTelemetry() noexcept;
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
