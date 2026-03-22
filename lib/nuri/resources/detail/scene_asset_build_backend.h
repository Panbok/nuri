#pragma once

#include "nuri/core/result.h"
#include "nuri/resources/mesh_importer.h"

#include <cstdint>
#include <memory_resource>
#include <span>
#include <string>
#include <string_view>

namespace nuri::detail {

[[nodiscard]] Result<MeshData, std::string>
loadSceneMeshFromSourceIndex(std::string_view path, uint32_t sceneMeshIndex,
                             const MeshImportOptions &options,
                             std::pmr::memory_resource *mem);
[[nodiscard]] Result<std::pmr::vector<MeshData>, std::string>
loadSceneMeshesFromSourceIndices(std::string_view path,
                                 std::span<const uint32_t> sceneMeshIndices,
                                 const MeshImportOptions &options,
                                 std::pmr::memory_resource *mem);
[[nodiscard]] Result<ImportedMaterialSet, std::string>
loadMaterialInfoFromSourceFile(std::string_view path);

} // namespace nuri::detail
