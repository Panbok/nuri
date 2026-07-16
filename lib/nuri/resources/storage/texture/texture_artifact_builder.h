#pragma once

#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_types.h"
#include "nuri/resources/cpu/material_data.h"
#include "nuri/resources/storage/texture/texture_cache_utils.h"
#include "nuri/resources/storage/texture/texture_processing.h"

#include <cstdint>
#include <filesystem>
#include <memory>

namespace nuri {

enum class TextureArtifactEncoding : uint8_t { Etc1s, Uastc };

struct TextureArtifactBuildOptions {
  TextureLoadOptions loadOptions{
      .srgb = false,
      .generateMipmaps = true,
  };
  TextureArtifactEncoding encoding = TextureArtifactEncoding::Etc1s;
};

struct TextureArtifactBuildResult {
  std::filesystem::path artifactPath{};
  Format targetFormat = Format::RGBA8_UNORM;
  NativeTextureCacheProbeStatus previousStatus =
      NativeTextureCacheProbeStatus::Missing;
  bool built = false;
  uint64_t artifactSizeBytes = 0u;
};

struct TextureArtifactCacheTelemetry {
  uint64_t nativeHits = 0u;
  uint64_t nativeMisses = 0u;
  uint64_t nativeStale = 0u;
  uint64_t nativeCorrupt = 0u;
  uint64_t nativeWrites = 0u;
  uint64_t nativeWriteFailures = 0u;
  uint64_t artifactBuilds = 0u;
  uint64_t authoredSourceBytesRead = 0u;
  uint64_t nativeArtifactBytesRead = 0u;
  uint64_t artifactBuildTimeNs = 0u;
};

class NURI_API SceneTextureArtifactBuilder final {
public:
  ~SceneTextureArtifactBuilder();

  SceneTextureArtifactBuilder(const SceneTextureArtifactBuilder &) = delete;
  SceneTextureArtifactBuilder &
  operator=(const SceneTextureArtifactBuilder &) = delete;
  SceneTextureArtifactBuilder(SceneTextureArtifactBuilder &&) noexcept;
  SceneTextureArtifactBuilder &
  operator=(SceneTextureArtifactBuilder &&) noexcept;

  [[nodiscard]] static Result<SceneTextureArtifactBuilder, std::string>
  create(const std::filesystem::path &sceneSourcePath);

  [[nodiscard]] Result<TextureArtifactBuildResult, std::string>
  ensure(const MaterialTextureSlotData &source, uint64_t sourceIdentityHash,
         Format targetFormat, const TextureArtifactBuildOptions &options,
         bool forceRebuild = false);

private:
  struct Impl;

  explicit SceneTextureArtifactBuilder(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] NURI_API Result<TextureArtifactBuildResult, std::string>
ensureTextureArtifactFromFile(
    const std::filesystem::path &sourcePath, uint64_t sourceIdentityHash,
    Format targetFormat, const TextureArtifactBuildOptions &options,
    bool forceRebuild = false);

[[nodiscard]] NURI_API uint32_t textureArtifactProcessingTag(
    const TextureArtifactBuildOptions &options) noexcept;

[[nodiscard]] NURI_API TextureArtifactCacheTelemetry
textureArtifactCacheTelemetry() noexcept;

} // namespace nuri
