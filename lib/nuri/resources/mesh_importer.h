#pragma once

#include "nuri/defines.h"
#include <array>
#include <cstdint>
#include <memory_resource>
#include <string>
#include <string_view>

#include "nuri/core/result.h"
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

class NURI_API MeshImporter {
public:
  [[nodiscard]] static nuri::Result<MeshData, std::string> loadFromFile(
      std::string_view path, const MeshImportOptions &options = {},
      std::pmr::memory_resource *mem = std::pmr::get_default_resource());
};

} // namespace nuri
