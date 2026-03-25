#pragma once

#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_types.h"
#include "nuri/resources/cpu/material_data.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

namespace nuri {

constexpr uint32_t kSceneTextureBakeSettingsVersion = 13u;

[[nodiscard]] NURI_API uint64_t
hashSceneTextureSourceIdentity(std::string_view sceneCanonicalPath,
                               const MaterialTextureSlotData &slotData,
                               bool srgb, uint32_t bakeSettingsTag = 0u);

[[nodiscard]] NURI_API Result<std::filesystem::path, std::string>
buildPortableTextureCachePath(const std::filesystem::path &sceneSourcePath,
                              uint64_t sourceIdentityHash);

[[nodiscard]] NURI_API std::filesystem::path
buildNativeTextureCachePath(const std::filesystem::path &portableTexturePath,
                            Format targetFormat);

[[nodiscard]] NURI_API bool
isTextureCacheUpToDate(const std::filesystem::path &cachePath,
                       const std::filesystem::path &sourcePath) noexcept;

[[nodiscard]] NURI_API std::string_view
textureFormatCacheSuffix(Format format) noexcept;

} // namespace nuri
