#pragma once

#include "nuri/defines.h"
#include <array>
#include <cstdint>
#include <memory_resource>
#include <span>
#include <string>
#include <string_view>

#include "nuri/core/result.h"
#include "nuri/resources/cpu/material_data.h"
#include "nuri/resources/cpu/mesh_data.h"

namespace nuri {

struct MeshImportOptions {
  static constexpr uint32_t kMaxLodCount = 4;

  bool triangulate = true;
  bool genNormals = true;
  bool genTangents = true;
  bool flipUVs = false;
  bool joinIdenticalVertices = true;
  bool genUVCoords = true;
  bool removeRedundantMaterials = true;
  bool limitBoneWeights = true;
  bool optimize = true;
  bool generateLods = true;
  uint32_t lodCount = kMaxLodCount;
  std::array<float, kMaxLodCount - 1> lodTriangleRatios{0.60f, 0.35f, 0.20f};
  float lodTargetError = 1e-2f;
};

using ImportedMaterialAlphaMode = MaterialAlphaMode;
using ImportedMaterialTexture = MaterialTextureSlotData;
using ImportedMaterialInfo = MaterialData;
using ImportedMaterialSet = MaterialDataSet;

class NURI_API MeshImporter {
public:
  [[nodiscard]] static nuri::Result<MeshData, std::string> loadFromFile(
      std::string_view path, const MeshImportOptions &options = {},
      std::pmr::memory_resource *mem = std::pmr::get_default_resource());
  [[nodiscard]] static nuri::Result<MeshData, std::string>
  loadSceneMeshFromFile(
      std::string_view path, uint32_t sceneMeshIndex,
      const MeshImportOptions &options = {},
      std::pmr::memory_resource *mem = std::pmr::get_default_resource());
  [[nodiscard]] static nuri::Result<std::pmr::vector<MeshData>, std::string>
  loadSceneMeshesFromFile(
      std::string_view path, std::span<const uint32_t> sceneMeshIndices,
      const MeshImportOptions &options = {},
      std::pmr::memory_resource *mem = std::pmr::get_default_resource());

  // Extracts material factors and texture slots from source files via Assimp.
  // Texture paths are resolved to normalized absolute paths when external.
  // Embedded scene textures use the identifier carried by
  // `ImportedMaterialTexture` (`MaterialTextureSlotData`): the
  // `sourceKind`/`embeddedIndex` fields form a `sourceKind:embeddedIndex`
  // pair, for example `EmbeddedSceneTexture:3`. See `ImportedMaterialTexture`
  // for the field definitions and expected consumer-side population.
  [[nodiscard]] static nuri::Result<ImportedMaterialSet, std::string>
  loadMaterialInfoFromFile(std::string_view path);
};

} // namespace nuri
