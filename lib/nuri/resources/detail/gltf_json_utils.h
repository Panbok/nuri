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

struct GltfPrimitiveMaterialMapping {
  uint32_t materialCount = 0;
  uint32_t meshCount = 0;
  bool sceneMeshIndicesAreFlatPrimitiveOrder = false;
  std::vector<uint32_t> singlePrimitiveMeshMaterialIndices{};
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

[[nodiscard]] Result<GltfPrimitiveMaterialMapping, std::string>
readGltfPrimitiveMaterialMapping(yyjson_val *root);

} // namespace nuri::detail
