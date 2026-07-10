#pragma once

#include "nuri/core/result.h"

#include <span>
#include <string>
#include <string_view>

struct yyjson_val;

namespace nuri::tools::core {

enum class JsonFieldType {
  Any,
  Object,
  Array,
  String,
  Boolean,
  Unsigned,
  Number,
  NullOrNumber,
  NullOrBoolean,
};

struct JsonFieldContract {
  std::string_view name{};
  JsonFieldType type = JsonFieldType::Any;
  bool required = true;
};

[[nodiscard]] Result<void, std::string>
rejectDuplicateJsonFieldsRecursively(yyjson_val *value,
                                     std::string_view path = "$");

[[nodiscard]] Result<void, std::string>
validateJsonObject(yyjson_val *object,
                   std::span<const JsonFieldContract> fields,
                   std::string_view path);

// Detailed report artifact fields can describe an external source, so absolute
// paths remain readable for v1 compatibility. Traversal components are never
// accepted because callers may resolve these fields under a run root.
[[nodiscard]] Result<void, std::string>
validateJsonArtifactPath(yyjson_val *object, std::string_view field,
                         std::string_view path, bool allowEmpty = true);

[[nodiscard]] Result<void, std::string>
validateJsonArtifactPath(std::string_view value, std::string_view path,
                         bool allowEmpty = true);

} // namespace nuri::tools::core
