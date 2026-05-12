#pragma once

#include "nuri/core/result.h"

#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <glm/glm.hpp>
#include <yyjson.h>

namespace nuri::detail {

using YyJsonDocPtr = std::unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)>;
using YyJsonDocResult = Result<YyJsonDocPtr, std::string>;

/// CPU-side lookup table for glTF material indices referenced by meshes and
/// primitives.
///
/// `meshCount` is the number of processed entries in `meshes`, capped to
/// `uint32_t::max()`. `singlePrimitiveMeshMaterialIndices` has one entry per
/// processed mesh and stores the material index when that mesh has exactly one
/// primitive; otherwise it stores `uint32_t::max()`. `primitiveMaterialIndices`
/// stores material indices in mesh order, then primitive order within each
/// mesh, and may contain more entries than `meshCount`. All material indices
/// refer to
/// `[0, materialCount)`, with `uint32_t::max()` used when a primitive omits
/// `primitive.material` or the index cannot be read.
struct GltfPrimitiveMaterialMapping {
  /// Number of entries in the glTF top-level `materials` array, capped to
  /// `uint32_t::max()`.
  uint32_t materialCount = 0;
  /// Number of processed entries in the glTF top-level `meshes` array.
  uint32_t meshCount = 0;
  /// True when scene mesh indices are expected to address the flattened
  /// primitive stream directly rather than nested mesh indices. This is used
  /// for single-mesh assets whose primitives become individual scene entries.
  bool sceneMeshIndicesAreFlatPrimitiveOrder = false;
  /// Per-mesh fast path for meshes with exactly one primitive. Entries are
  /// material indices or `uint32_t::max()` for multi-primitive/no-material
  /// meshes.
  std::vector<uint32_t> singlePrimitiveMeshMaterialIndices{};
  /// Per-primitive material indices flattened by mesh order. Multi-primitive
  /// meshes use this vector for primitive-level material assignment.
  std::vector<uint32_t> primitiveMaterialIndices{};
};

[[nodiscard]] bool
hasExtensionCaseInsensitive(const std::filesystem::path &path,
                            std::string_view extension);
[[nodiscard]] bool isGltfJsonAssetPath(const std::filesystem::path &path);
[[nodiscard]] bool isGltfJsonAssetPath(std::string_view path);

[[nodiscard]] std::string_view readJsonStringView(yyjson_val *value);
[[nodiscard]] std::string_view readJsonStringView(yyjson_val *object,
                                                  const char *key);

[[nodiscard]] bool tryReadJsonFloat(yyjson_val *value, float &out);
[[nodiscard]] bool tryReadJsonUint32(yyjson_val *value, uint32_t &out);
[[nodiscard]] bool tryReadJsonVec2(yyjson_val *value, glm::vec2 &out);
[[nodiscard]] bool tryReadJsonVec3(yyjson_val *value, glm::vec3 &out);
[[nodiscard]] bool tryReadJsonVec4(yyjson_val *value, glm::vec4 &out);
[[nodiscard]] bool tryReadJsonMat4(yyjson_val *value, glm::mat4 &out);
[[nodiscard]] bool tryReadJsonBool(yyjson_val *value, bool &out);

[[nodiscard]] YyJsonDocResult
loadGltfJsonDocument(const std::filesystem::path &path,
                     std::string_view sourceLabel);
[[nodiscard]] YyJsonDocResult
loadGltfJsonDocument(const std::filesystem::path &path,
                     std::span<const std::byte> fileBytes,
                     std::string_view sourceLabel);

/// Reads material assignment metadata from a yyjson value representing the
/// glTF JSON root object.
///
/// On success, returns a populated `GltfPrimitiveMaterialMapping`. The current
/// implementation treats missing `materials`, missing `meshes`, missing
/// `primitives`, missing `primitive.material`, and material type mismatches as
/// absent data represented by zero counts or `uint32_t::max()` sentinels. It
/// returns an error string when `root` is not the expected JSON object
/// ("glTF root is not a JSON object"). Callers should pass the document root,
/// not an individual primitive, and interpret indices according to the glTF
/// mesh primitive `material` field plus Nuri's material remapping convention.
[[nodiscard]] Result<GltfPrimitiveMaterialMapping, std::string>
readGltfPrimitiveMaterialMapping(yyjson_val *root);

} // namespace nuri::detail
