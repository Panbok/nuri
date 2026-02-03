#pragma once

#include "nuri/core/result.h"
#include "nuri/resources/cpu/mesh_data.h"

namespace nuri {

struct MeshImportOptions {
  bool triangulate = true;
  bool genNormals = true;
  bool genTangents = true;
  bool flipUVs = false;
  bool joinIdenticalVertices = true;
  bool optimize = true;
};

class MeshImporter {
public:
  [[nodiscard]] static nuri::Result<MeshData, std::string> loadFromFile(
      std::string_view path, const MeshImportOptions &options = {},
      std::pmr::memory_resource *mem = std::pmr::get_default_resource());
};

} // namespace nuri
