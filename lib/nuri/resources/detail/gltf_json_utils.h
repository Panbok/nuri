#pragma once

#include "nuri/core/result.h"

#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>

#include <glm/glm.hpp>
#include <yyjson.h>

namespace nuri::detail {

using YyJsonDocPtr = std::unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)>;
using YyJsonDocResult = Result<YyJsonDocPtr, std::string>;

[[nodiscard]] bool
hasExtensionCaseInsensitive(const std::filesystem::path &path,
                            std::string_view extension);
[[nodiscard]] bool isGltfJsonAssetPath(const std::filesystem::path &path);
[[nodiscard]] bool isGltfJsonAssetPath(std::string_view path);

[[nodiscard]] std::string_view readJsonStringView(yyjson_val *value);
[[nodiscard]] std::string_view readJsonStringView(yyjson_val *object,
                                                  const char *key);

bool tryReadJsonFloat(yyjson_val *value, float &out);
bool tryReadJsonUint32(yyjson_val *value, uint32_t &out);
bool tryReadJsonVec2(yyjson_val *value, glm::vec2 &out);
bool tryReadJsonVec3(yyjson_val *value, glm::vec3 &out);
bool tryReadJsonVec4(yyjson_val *value, glm::vec4 &out);
bool tryReadJsonMat4(yyjson_val *value, glm::mat4 &out);
bool tryReadJsonBool(yyjson_val *value, bool &out);

[[nodiscard]] YyJsonDocResult
loadGltfJsonDocument(const std::filesystem::path &path,
                     std::string_view sourceLabel);

} // namespace nuri::detail
