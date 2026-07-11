#include "nuri/tools/core/json_contract.h"

#include <algorithm>
#include <filesystem>
#include <unordered_set>

#include <yyjson.h>

namespace nuri::tools::core {
namespace {

[[nodiscard]] bool matchesType(yyjson_val *value, JsonFieldType type) {
  switch (type) {
  case JsonFieldType::Any:
    return value != nullptr;
  case JsonFieldType::Object:
    return yyjson_is_obj(value);
  case JsonFieldType::Array:
    return yyjson_is_arr(value);
  case JsonFieldType::String:
    return yyjson_is_str(value);
  case JsonFieldType::Boolean:
    return yyjson_is_bool(value);
  case JsonFieldType::Unsigned:
    return yyjson_is_uint(value);
  case JsonFieldType::Number:
    return yyjson_is_num(value);
  case JsonFieldType::NullOrNumber:
    return yyjson_is_null(value) || yyjson_is_num(value);
  case JsonFieldType::NullOrBoolean:
    return yyjson_is_null(value) || yyjson_is_bool(value);
  }
  return false;
}

[[nodiscard]] std::string_view typeName(JsonFieldType type) {
  switch (type) {
  case JsonFieldType::Any:
    return "a value";
  case JsonFieldType::Object:
    return "an object";
  case JsonFieldType::Array:
    return "an array";
  case JsonFieldType::String:
    return "a string";
  case JsonFieldType::Boolean:
    return "a boolean";
  case JsonFieldType::Unsigned:
    return "an unsigned integer";
  case JsonFieldType::Number:
    return "a number";
  case JsonFieldType::NullOrNumber:
    return "null or a number";
  case JsonFieldType::NullOrBoolean:
    return "null or a boolean";
  }
  return "the expected type";
}

} // namespace

Result<void, std::string>
rejectDuplicateJsonFieldsRecursively(yyjson_val *value, std::string_view path) {
  if (yyjson_is_obj(value)) {
    std::unordered_set<std::string> seen;
    yyjson_obj_iter iterator{};
    yyjson_obj_iter_init(value, &iterator);
    yyjson_val *key = nullptr;
    while ((key = yyjson_obj_iter_next(&iterator)) != nullptr) {
      const std::string_view name(yyjson_get_str(key), yyjson_get_len(key));
      if (!seen.emplace(name).second) {
        return Result<void, std::string>::makeError(
            std::string(path) + " contains duplicate field '" +
            std::string(name) + "'");
      }
      auto child = rejectDuplicateJsonFieldsRecursively(
          yyjson_obj_iter_get_val(key),
          std::string(path) + "." + std::string(name));
      if (child.hasError()) {
        return child;
      }
    }
  } else if (yyjson_is_arr(value)) {
    yyjson_arr_iter iterator{};
    yyjson_arr_iter_init(value, &iterator);
    yyjson_val *entry = nullptr;
    size_t index = 0u;
    while ((entry = yyjson_arr_iter_next(&iterator)) != nullptr) {
      auto child = rejectDuplicateJsonFieldsRecursively(
          entry, std::string(path) + "[" + std::to_string(index++) + "]");
      if (child.hasError()) {
        return child;
      }
    }
  }
  return Result<void, std::string>::makeResult();
}

Result<void, std::string>
validateJsonObject(yyjson_val *object,
                   std::span<const JsonFieldContract> fields,
                   std::string_view path) {
  if (!yyjson_is_obj(object)) {
    return Result<void, std::string>::makeError(std::string(path) +
                                                " must be an object");
  }

  yyjson_obj_iter iterator{};
  yyjson_obj_iter_init(object, &iterator);
  yyjson_val *key = nullptr;
  while ((key = yyjson_obj_iter_next(&iterator)) != nullptr) {
    const std::string_view name(yyjson_get_str(key), yyjson_get_len(key));
    const auto field = std::find_if(fields.begin(), fields.end(),
                                    [name](const JsonFieldContract &candidate) {
                                      return candidate.name == name;
                                    });
    if (field == fields.end()) {
      return Result<void, std::string>::makeError(std::string(path) +
                                                  " contains unknown field '" +
                                                  std::string(name) + "'");
    }
  }

  for (const JsonFieldContract &field : fields) {
    yyjson_val *value =
        yyjson_obj_getn(object, field.name.data(), field.name.size());
    if (value == nullptr) {
      if (field.required) {
        return Result<void, std::string>::makeError(
            std::string(path) + "." + std::string(field.name) + " is required");
      }
      continue;
    }
    if (!matchesType(value, field.type)) {
      return Result<void, std::string>::makeError(
          std::string(path) + "." + std::string(field.name) + " must be " +
          std::string(typeName(field.type)));
    }
  }
  return Result<void, std::string>::makeResult();
}

Result<void, std::string> validateJsonArtifactPath(yyjson_val *object,
                                                   std::string_view field,
                                                   std::string_view path,
                                                   bool allowEmpty) {
  yyjson_val *value = yyjson_obj_getn(object, field.data(), field.size());
  if (!yyjson_is_str(value)) {
    return Result<void, std::string>::makeError(
        std::string(path) + "." + std::string(field) + " must be a string");
  }
  return validateJsonArtifactPath(
      std::string_view(yyjson_get_str(value), yyjson_get_len(value)),
      std::string(path) + "." + std::string(field), allowEmpty);
}

Result<void, std::string> validateJsonArtifactPath(std::string_view value,
                                                   std::string_view path,
                                                   bool allowEmpty) {
  if (value.empty()) {
    if (allowEmpty) {
      return Result<void, std::string>::makeResult();
    }
    return Result<void, std::string>::makeError(std::string(path) +
                                                " must not be empty");
  }
  const std::u8string utf8(reinterpret_cast<const char8_t *>(value.data()),
                           value.size());
  const std::filesystem::path candidate(utf8);
  for (const auto &component : candidate) {
    if (component == "..") {
      return Result<void, std::string>::makeError(
          std::string(path) + " must not contain traversal components");
    }
  }
  return Result<void, std::string>::makeResult();
}

} // namespace nuri::tools::core
