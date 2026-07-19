#pragma once
#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/resources/cpu/material_data.h"
#include "nuri/resources/cpu/mesh_data.h"
#include <array>
#include <cstdint>
#include <memory_resource>
#include <span>
#include <string>
#include <string_view>
namespace nuri {

struct MeshImportOptions {
  static constexpr uint32_t kMaxLodCount = 4;
  bool triangulate = true;
  bool genNormals = true;
  bool calcTangents = true;
  bool flipUVs = false;
  bool joinIdenticalVertices = true;
  bool genUVCoords = true;
  bool removeRedundantMaterials = true;
  bool limitBoneWeights = true;
  bool optimize = true;
  bool generateLods = true;
  bool generateMeshlets = false;
  uint32_t lodCount = kMaxLodCount;
  uint32_t meshletMaxVertices = 64u;
  uint32_t meshletMaxPrimitives = 124u;
  float meshletConeWeight = 0.0f;
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
  [[nodiscard]] static nuri::Result<ImportedMaterialSet, std::string>
  loadMaterialInfoFromFile(std::string_view path);
};

} // namespace nuri
