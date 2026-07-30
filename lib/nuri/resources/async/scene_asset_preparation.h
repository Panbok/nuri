#pragma once
#include "nuri/core/result.h"
#include "nuri/gfx/gpu_types.h"
#include "nuri/resources/gpu/material.h"
#include "nuri/resources/gpu/resource_keys.h"
#include "nuri/resources/gpu/resource_manager.h"
#include "nuri/resources/scene_importer.h"
#include <array>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>
namespace nuri {

struct PreparedImportedMaterial {
  MaterialDesc desc{};
  std::array<std::optional<TextureRequest>, kMaterialTextureSlotCount>
      textures{};
  std::string debugName{};
  std::string sourceIdentity{};
  std::vector<std::string> optionalTextureErrors{};
};

[[nodiscard]] Result<PreparedImportedMaterial, std::string>
prepareImportedMaterial(
    const MaterialData &material, std::string_view scenePath,
    uint32_t sourceMaterialIndex, const TextureCompressionCaps &compressionCaps,
    std::span<const EmbeddedSceneTextureData> embeddedTextures = {},
    std::string_view debugNamePrefix = "scene");

} // namespace nuri
