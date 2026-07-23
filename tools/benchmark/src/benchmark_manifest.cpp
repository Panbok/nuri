#include "nuri/tools/benchmark/benchmark_manifest.h"

#include "nuri/core/runtime_config.h"
#include "nuri/tools/benchmark/benchmark_environment.h"
#include "nuri/tools/benchmark/benchmark_metric_registry.h"
#include "nuri/tools/core/case_catalog.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <climits>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <set>
#include <span>
#include <string_view>

#include <yyjson.h>

namespace nuri::tools::benchmark {
namespace {

using JsonDocPtr = std::unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)>;

[[nodiscard]] bool isLowerAsciiLetter(char value) {
  return value >= 'a' && value <= 'z';
}

[[nodiscard]] bool isAsciiDigit(char value) {
  return value >= '0' && value <= '9';
}

[[nodiscard]] bool isFiniteVec3(const glm::vec3 &value) {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
         std::isfinite(value.z);
}

[[nodiscard]] bool isSafeIdentifierSegment(std::string_view segment) {
  if (segment.empty() || segment.size() > 64u ||
      (!isLowerAsciiLetter(segment.front()) &&
       !isAsciiDigit(segment.front()))) {
    return false;
  }
  for (const char value : segment) {
    if (!isLowerAsciiLetter(value) && !isAsciiDigit(value) && value != '_' &&
        value != '-') {
      return false;
    }
  }
  static constexpr std::array reserved{
      std::string_view("con"),  std::string_view("prn"),
      std::string_view("aux"),  std::string_view("nul"),
      std::string_view("com1"), std::string_view("com2"),
      std::string_view("com3"), std::string_view("com4"),
      std::string_view("com5"), std::string_view("com6"),
      std::string_view("com7"), std::string_view("com8"),
      std::string_view("com9"), std::string_view("lpt1"),
      std::string_view("lpt2"), std::string_view("lpt3"),
      std::string_view("lpt4"), std::string_view("lpt5"),
      std::string_view("lpt6"), std::string_view("lpt7"),
      std::string_view("lpt8"), std::string_view("lpt9")};
  return std::find(reserved.begin(), reserved.end(), segment) == reserved.end();
}

[[nodiscard]] bool isSafeDottedIdentifier(std::string_view identifier) {
  size_t begin = 0u;
  while (begin < identifier.size()) {
    const size_t end = identifier.find('.', begin);
    const std::string_view segment = identifier.substr(
        begin, end == std::string_view::npos ? identifier.size() - begin
                                             : end - begin);
    if (!isSafeIdentifierSegment(segment)) {
      return false;
    }
    if (end == std::string_view::npos) {
      return true;
    }
    begin = end + 1u;
  }
  return false;
}

[[nodiscard]] std::string jsonPath(std::string_view path,
                                   std::string_view key) {
  std::string out(path);
  if (!out.empty()) {
    out += ".";
  }
  out += key;
  return out;
}

[[nodiscard]] bool hasKey(std::span<const std::string_view> keys,
                          std::string_view key) {
  return std::find(keys.begin(), keys.end(), key) != keys.end();
}

[[nodiscard]] Result<bool, std::string>
rejectUnknownKeys(yyjson_val *object, std::span<const std::string_view> keys,
                  std::string_view path) {
  if (!yyjson_is_obj(object)) {
    return Result<bool, std::string>::makeError(std::string(path) +
                                                " must be an object");
  }
  yyjson_obj_iter iter;
  yyjson_obj_iter_init(object, &iter);
  yyjson_val *key = nullptr;
  while ((key = yyjson_obj_iter_next(&iter)) != nullptr) {
    const std::string_view name(yyjson_get_str(key), yyjson_get_len(key));
    if (!hasKey(keys, name)) {
      return Result<bool, std::string>::makeError(
          std::string(path) + ": unknown key '" + std::string(name) + "'");
    }
  }
  return Result<bool, std::string>::makeResult(true);
}

[[nodiscard]] yyjson_val *optionalObject(yyjson_val *object,
                                         std::string_view key) {
  return yyjson_obj_getn(object, key.data(), key.size());
}

[[nodiscard]] Result<yyjson_val *, std::string>
requiredValue(yyjson_val *object, std::string_view key, std::string_view path) {
  yyjson_val *value = optionalObject(object, key);
  if (value == nullptr) {
    return Result<yyjson_val *, std::string>::makeError(jsonPath(path, key) +
                                                        " is required");
  }
  return Result<yyjson_val *, std::string>::makeResult(value);
}

[[nodiscard]] Result<std::string, std::string>
readString(yyjson_val *object, std::string_view key, std::string_view path,
           bool required = false, std::string defaultValue = {}) {
  yyjson_val *value = optionalObject(object, key);
  if (value == nullptr) {
    if (required) {
      return Result<std::string, std::string>::makeError(jsonPath(path, key) +
                                                         " is required");
    }
    return Result<std::string, std::string>::makeResult(
        std::move(defaultValue));
  }
  if (!yyjson_is_str(value)) {
    return Result<std::string, std::string>::makeError(jsonPath(path, key) +
                                                       " must be a string");
  }
  return Result<std::string, std::string>::makeResult(
      std::string(yyjson_get_str(value), yyjson_get_len(value)));
}

[[nodiscard]] Result<bool, std::string> readBool(yyjson_val *object,
                                                 std::string_view key,
                                                 std::string_view path,
                                                 bool defaultValue = false) {
  yyjson_val *value = optionalObject(object, key);
  if (value == nullptr) {
    return Result<bool, std::string>::makeResult(defaultValue);
  }
  if (!yyjson_is_bool(value)) {
    return Result<bool, std::string>::makeError(jsonPath(path, key) +
                                                " must be a bool");
  }
  return Result<bool, std::string>::makeResult(yyjson_get_bool(value));
}

[[nodiscard]] Result<uint32_t, std::string>
readU32(yyjson_val *object, std::string_view key, std::string_view path,
        uint32_t defaultValue = 0u) {
  yyjson_val *value = optionalObject(object, key);
  if (value == nullptr) {
    return Result<uint32_t, std::string>::makeResult(defaultValue);
  }
  if (!yyjson_is_uint(value) || yyjson_get_uint(value) > UINT32_MAX) {
    return Result<uint32_t, std::string>::makeError(jsonPath(path, key) +
                                                    " must be a uint32");
  }
  return Result<uint32_t, std::string>::makeResult(
      static_cast<uint32_t>(yyjson_get_uint(value)));
}

[[nodiscard]] Result<double, std::string>
readDouble(yyjson_val *object, std::string_view key, std::string_view path,
           double defaultValue = 0.0) {
  yyjson_val *value = optionalObject(object, key);
  if (value == nullptr) {
    return Result<double, std::string>::makeResult(defaultValue);
  }
  if (!yyjson_is_num(value)) {
    return Result<double, std::string>::makeError(jsonPath(path, key) +
                                                  " must be a number");
  }
  return Result<double, std::string>::makeResult(yyjson_get_num(value));
}

[[nodiscard]] Result<std::vector<std::string>, std::string>
readStringArray(yyjson_val *object, std::string_view key,
                std::string_view path) {
  std::vector<std::string> out;
  yyjson_val *value = optionalObject(object, key);
  if (value == nullptr) {
    return Result<std::vector<std::string>, std::string>::makeResult(
        std::move(out));
  }
  if (!yyjson_is_arr(value)) {
    return Result<std::vector<std::string>, std::string>::makeError(
        jsonPath(path, key) + " must be an array");
  }
  out.reserve(yyjson_arr_size(value));
  yyjson_arr_iter iter;
  yyjson_arr_iter_init(value, &iter);
  yyjson_val *item = nullptr;
  while ((item = yyjson_arr_iter_next(&iter)) != nullptr) {
    if (!yyjson_is_str(item)) {
      return Result<std::vector<std::string>, std::string>::makeError(
          jsonPath(path, key) + " entries must be strings");
    }
    out.emplace_back(yyjson_get_str(item), yyjson_get_len(item));
  }
  return Result<std::vector<std::string>, std::string>::makeResult(
      std::move(out));
}

[[nodiscard]] Result<glm::vec3, std::string> readVec3(yyjson_val *object,
                                                      std::string_view key,
                                                      std::string_view path,
                                                      glm::vec3 defaultValue) {
  yyjson_val *value = optionalObject(object, key);
  if (value == nullptr) {
    return Result<glm::vec3, std::string>::makeResult(defaultValue);
  }
  if (!yyjson_is_arr(value) || yyjson_arr_size(value) != 3u) {
    return Result<glm::vec3, std::string>::makeError(jsonPath(path, key) +
                                                     " must be a vec3 array");
  }
  glm::vec3 out{};
  for (uint32_t i = 0u; i < 3u; ++i) {
    yyjson_val *entry = yyjson_arr_get(value, i);
    if (!yyjson_is_num(entry)) {
      return Result<glm::vec3, std::string>::makeError(
          jsonPath(path, key) + " entries must be numbers");
    }
    out[i] = static_cast<float>(yyjson_get_num(entry));
  }
  return Result<glm::vec3, std::string>::makeResult(out);
}

[[nodiscard]] Result<glm::uvec3, std::string>
readUVec3(yyjson_val *object, std::string_view key, std::string_view path,
          glm::uvec3 defaultValue) {
  yyjson_val *value = optionalObject(object, key);
  if (value == nullptr) {
    return Result<glm::uvec3, std::string>::makeResult(defaultValue);
  }
  if (!yyjson_is_arr(value) || yyjson_arr_size(value) != 3u) {
    return Result<glm::uvec3, std::string>::makeError(jsonPath(path, key) +
                                                      " must be a uvec3 array");
  }
  glm::uvec3 out{};
  for (uint32_t i = 0u; i < 3u; ++i) {
    yyjson_val *entry = yyjson_arr_get(value, i);
    if (!yyjson_is_uint(entry) || yyjson_get_uint(entry) > UINT32_MAX) {
      return Result<glm::uvec3, std::string>::makeError(
          jsonPath(path, key) + " entries must be uint32 values");
    }
    out[i] = static_cast<uint32_t>(yyjson_get_uint(entry));
  }
  return Result<glm::uvec3, std::string>::makeResult(out);
}

template <typename Enum>
[[nodiscard]] Result<Enum, std::string>
parseEnum(std::string_view text,
          std::initializer_list<std::pair<std::string_view, Enum>> values,
          std::string_view path) {
  for (const auto &[name, value] : values) {
    if (text == name) {
      return Result<Enum, std::string>::makeResult(value);
    }
  }
  return Result<Enum, std::string>::makeError(
      std::string(path) + ": invalid enum value '" + std::string(text) + "'");
}

template <typename Enum>
[[nodiscard]] Result<bool, std::string>
readEnumField(yyjson_val *object, std::string_view key, std::string_view path,
              Enum &out,
              std::initializer_list<std::pair<std::string_view, Enum>> values) {
  yyjson_val *value = optionalObject(object, key);
  if (value == nullptr) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (!yyjson_is_str(value)) {
    return Result<bool, std::string>::makeError(jsonPath(path, key) +
                                                " must be a string");
  }
  auto parsed = parseEnum<Enum>(
      std::string_view(yyjson_get_str(value), yyjson_get_len(value)), values,
      jsonPath(path, key));
  if (parsed.hasError()) {
    return Result<bool, std::string>::makeError(parsed.error());
  }
  out = parsed.value();
  return Result<bool, std::string>::makeResult(true);
}

[[nodiscard]] Result<bool, std::string>
parseDDGICoverageSettings(yyjson_val *object, DDGICoverageSettings &settings,
                          std::string_view path) {
  static constexpr std::array keys{
      std::string_view("schemaVersion"),
      std::string_view("mode"),
      std::string_view("constraintPolicy"),
      std::string_view("sceneBoundsSource"),
      std::string_view("authoredBounds"),
      std::string_view("cascadeCount"),
      std::string_view("cascadeProbeCounts"),
      std::string_view("requestedNearSpacing"),
      std::string_view("spacingRatio"),
      std::string_view("requestedCoverageHalfExtents"),
      std::string_view("scenePaddingCells"),
      std::string_view("transitionCells"),
      std::string_view("includeAuthoredVolumes")};
  auto result = rejectUnknownKeys(object, keys, path);
  if (result.hasError()) {
    return result;
  }
  auto schema = readU32(object, "schemaVersion", path);
  if (schema.hasError()) {
    return Result<bool, std::string>::makeError(schema.error());
  }
  if (schema.value() != kDDGICoverageSchemaVersion) {
    return Result<bool, std::string>::makeError(
        jsonPath(path, "schemaVersion") + " must be 1");
  }
  result = readEnumField(object, "mode", path, settings.mode,
                         {{"Manual", DDGICoverageMode::Manual},
                          {"SceneFit", DDGICoverageMode::SceneFit},
                          {"CameraClipmaps", DDGICoverageMode::CameraClipmaps},
                          {"Hybrid", DDGICoverageMode::Hybrid}});
  if (result.hasError()) {
    return result;
  }
  result = readEnumField(
      object, "constraintPolicy", path, settings.constraintPolicy,
      {{"PreserveCoverage", DDGICoverageConstraintPolicy::PreserveCoverage},
       {"PreserveNearSpacing",
        DDGICoverageConstraintPolicy::PreserveNearSpacing},
       {"RejectUnsatisfied", DDGICoverageConstraintPolicy::RejectUnsatisfied}});
  if (result.hasError()) {
    return result;
  }
  result = readEnumField(
      object, "sceneBoundsSource", path, settings.sceneBoundsSource,
      {{"ActivationSnapshot", DDGISceneBoundsSource::ActivationSnapshot},
       {"StaticRayTracingGeometry",
        DDGISceneBoundsSource::StaticRayTracingGeometry},
       {"Authored", DDGISceneBoundsSource::Authored}});
  if (result.hasError()) {
    return result;
  }
  if (yyjson_val *bounds = optionalObject(object, "authoredBounds")) {
    static constexpr std::array boundsKeys{std::string_view("min"),
                                           std::string_view("max")};
    const std::string boundsPath = jsonPath(path, "authoredBounds");
    result = rejectUnknownKeys(bounds, boundsKeys, boundsPath);
    if (result.hasError()) {
      return result;
    }
    if (optionalObject(bounds, "min") == nullptr ||
        optionalObject(bounds, "max") == nullptr) {
      return Result<bool, std::string>::makeError(boundsPath +
                                                  " must contain min and max");
    }
    auto minimum = readVec3(bounds, "min", boundsPath, glm::vec3(0.0f));
    if (minimum.hasError()) {
      return Result<bool, std::string>::makeError(minimum.error());
    }
    auto maximum = readVec3(bounds, "max", boundsPath, glm::vec3(0.0f));
    if (maximum.hasError()) {
      return Result<bool, std::string>::makeError(maximum.error());
    }
    for (uint32_t axis = 0u; axis < 3u; ++axis) {
      if (!std::isfinite(minimum.value()[axis]) ||
          !std::isfinite(maximum.value()[axis]) ||
          minimum.value()[axis] > maximum.value()[axis]) {
        return Result<bool, std::string>::makeError(
            boundsPath +
            " must contain finite min values not greater than max values");
      }
    }
    settings.authoredBounds = DDGISceneCoverageBounds{
        .bounds = BoundingBox(minimum.value(), maximum.value()),
        .valid = true,
        .complete = true,
    };
  }
  auto count = readU32(object, "cascadeCount", path, settings.cascadeCount);
  if (count.hasError()) {
    return Result<bool, std::string>::makeError(count.error());
  }
  settings.cascadeCount = count.value();
  auto probeCounts = readUVec3(object, "cascadeProbeCounts", path,
                               settings.cascadeProbeCounts);
  if (probeCounts.hasError()) {
    return Result<bool, std::string>::makeError(probeCounts.error());
  }
  settings.cascadeProbeCounts = probeCounts.value();
  auto vector = readVec3(object, "requestedNearSpacing", path,
                         settings.requestedNearSpacing);
  if (vector.hasError()) {
    return Result<bool, std::string>::makeError(vector.error());
  }
  settings.requestedNearSpacing = vector.value();
  vector = readVec3(object, "requestedCoverageHalfExtents", path,
                    settings.requestedCoverageHalfExtents);
  if (vector.hasError()) {
    return Result<bool, std::string>::makeError(vector.error());
  }
  settings.requestedCoverageHalfExtents = vector.value();
  auto real = readDouble(object, "spacingRatio", path, settings.spacingRatio);
  if (real.hasError()) {
    return Result<bool, std::string>::makeError(real.error());
  }
  settings.spacingRatio = static_cast<float>(real.value());
  count =
      readU32(object, "scenePaddingCells", path, settings.scenePaddingCells);
  if (count.hasError()) {
    return Result<bool, std::string>::makeError(count.error());
  }
  settings.scenePaddingCells = count.value();
  count = readU32(object, "transitionCells", path, settings.transitionCells);
  if (count.hasError()) {
    return Result<bool, std::string>::makeError(count.error());
  }
  settings.transitionCells = count.value();
  auto boolean = readBool(object, "includeAuthoredVolumes", path,
                          settings.includeAuthoredVolumes);
  if (boolean.hasError()) {
    return Result<bool, std::string>::makeError(boolean.error());
  }
  settings.includeAuthoredVolumes = boolean.value();
  return Result<bool, std::string>::makeResult(true);
}

[[nodiscard]] Result<bool, std::string>
parseOpaqueSettings(yyjson_val *object, RenderSettings &settings,
                    std::string_view path) {
  static constexpr std::array keys{
      std::string_view("enabled"),
      std::string_view("enableDepthPrepass"),
      std::string_view("enableDepthPyramid"),
      std::string_view("enableInstanceCompute"),
      std::string_view("enableIndirectDraw"),
      std::string_view("enableInstancedDraw"),
      std::string_view("enableMeshLod"),
      std::string_view("meshLodTargetPixelError"),
      std::string_view("meshLodHysteresisRatio"),
      std::string_view("enableCpuFrustumCulling"),
      std::string_view("meshletMode"),
      std::string_view("enableMeshletFrustumCulling"),
      std::string_view("enableMeshletConeCulling"),
      std::string_view("hybridClassicMaxMeshlets"),
      std::string_view("enableInstanceAnimation"),
      std::string_view("enableTessellation"),
      std::string_view("forcedMeshLod"),
  };
  auto keysResult = rejectUnknownKeys(object, keys, path);
  if (keysResult.hasError()) {
    return keysResult;
  }
  auto b = readBool(object, "enabled", path, settings.opaque.enabled);
  if (b.hasError()) {
    return Result<bool, std::string>::makeError(b.error());
  }
  settings.opaque.enabled = b.value();
  b = readBool(object, "enableDepthPrepass", path,
               settings.opaque.enableDepthPrepass);
  if (b.hasError()) {
    return Result<bool, std::string>::makeError(b.error());
  }
  settings.opaque.enableDepthPrepass = b.value();
  b = readBool(object, "enableDepthPyramid", path,
               settings.opaque.enableDepthPyramid);
  if (b.hasError()) {
    return Result<bool, std::string>::makeError(b.error());
  }
  settings.opaque.enableDepthPyramid = b.value();
  b = readBool(object, "enableInstanceCompute", path,
               settings.opaque.enableInstanceCompute);
  if (b.hasError()) {
    return Result<bool, std::string>::makeError(b.error());
  }
  settings.opaque.enableInstanceCompute = b.value();
  b = readBool(object, "enableIndirectDraw", path,
               settings.opaque.enableIndirectDraw);
  if (b.hasError()) {
    return Result<bool, std::string>::makeError(b.error());
  }
  settings.opaque.enableIndirectDraw = b.value();
  b = readBool(object, "enableInstancedDraw", path,
               settings.opaque.enableInstancedDraw);
  if (b.hasError()) {
    return Result<bool, std::string>::makeError(b.error());
  }
  settings.opaque.enableInstancedDraw = b.value();
  b = readBool(object, "enableMeshLod", path, settings.opaque.enableMeshLod);
  if (b.hasError()) {
    return Result<bool, std::string>::makeError(b.error());
  }
  settings.opaque.enableMeshLod = b.value();
  auto lodNumber = readDouble(object, "meshLodTargetPixelError", path,
                              settings.opaque.meshLodTargetPixelError);
  if (lodNumber.hasError()) {
    return Result<bool, std::string>::makeError(lodNumber.error());
  }
  settings.opaque.meshLodTargetPixelError =
      static_cast<float>(lodNumber.value());
  lodNumber = readDouble(object, "meshLodHysteresisRatio", path,
                         settings.opaque.meshLodHysteresisRatio);
  if (lodNumber.hasError()) {
    return Result<bool, std::string>::makeError(lodNumber.error());
  }
  settings.opaque.meshLodHysteresisRatio =
      static_cast<float>(lodNumber.value());
  b = readBool(object, "enableCpuFrustumCulling", path,
               settings.opaque.enableCpuFrustumCulling);
  if (b.hasError()) {
    return Result<bool, std::string>::makeError(b.error());
  }
  settings.opaque.enableCpuFrustumCulling = b.value();
  auto enumResult =
      readEnumField(object, "meshletMode", path, settings.opaque.meshletMode,
                    {{"Disabled", MeshletRenderMode::Disabled},
                     {"Opportunistic", MeshletRenderMode::Opportunistic},
                     {"Required", MeshletRenderMode::Required}});
  if (enumResult.hasError()) {
    return enumResult;
  }
  auto hybridClassicMaxMeshlets =
      readU32(object, "hybridClassicMaxMeshlets", path,
              settings.opaque.hybridClassicMaxMeshlets);
  if (hybridClassicMaxMeshlets.hasError()) {
    return Result<bool, std::string>::makeError(
        hybridClassicMaxMeshlets.error());
  }
  settings.opaque.hybridClassicMaxMeshlets = hybridClassicMaxMeshlets.value();
  b = readBool(object, "enableMeshletFrustumCulling", path,
               settings.opaque.enableMeshletFrustumCulling);
  if (b.hasError()) {
    return Result<bool, std::string>::makeError(b.error());
  }
  settings.opaque.enableMeshletFrustumCulling = b.value();
  b = readBool(object, "enableMeshletConeCulling", path,
               settings.opaque.enableMeshletConeCulling);
  if (b.hasError()) {
    return Result<bool, std::string>::makeError(b.error());
  }
  settings.opaque.enableMeshletConeCulling = b.value();
  b = readBool(object, "enableInstanceAnimation", path,
               settings.opaque.enableInstanceAnimation);
  if (b.hasError()) {
    return Result<bool, std::string>::makeError(b.error());
  }
  settings.opaque.enableInstanceAnimation = b.value();
  b = readBool(object, "enableTessellation", path,
               settings.opaque.enableTessellation);
  if (b.hasError()) {
    return Result<bool, std::string>::makeError(b.error());
  }
  settings.opaque.enableTessellation = b.value();
  yyjson_val *forced = optionalObject(object, "forcedMeshLod");
  if (forced != nullptr) {
    if (yyjson_is_uint(forced) &&
        yyjson_get_uint(forced) <= static_cast<uint64_t>(INT32_MAX)) {
      settings.opaque.forcedMeshLod =
          static_cast<int32_t>(yyjson_get_uint(forced));
    } else if (!yyjson_is_sint(forced) || yyjson_get_sint(forced) < INT32_MIN ||
               yyjson_get_sint(forced) > INT32_MAX) {
      return Result<bool, std::string>::makeError(
          jsonPath(path, "forcedMeshLod") + " must be an int32");
    } else {
      settings.opaque.forcedMeshLod =
          static_cast<int32_t>(yyjson_get_sint(forced));
    }
  }
  return Result<bool, std::string>::makeResult(true);
}

[[nodiscard]] Result<bool, std::string>
parseAntiAliasingSettings(yyjson_val *object, RenderSettings &settings,
                          std::string_view path) {
  static constexpr std::array keys{std::string_view("mode"),
                                   std::string_view("temporalProvider"),
                                   std::string_view("qualityPreset"),
                                   std::string_view("spatialPostMsaaCleanup")};
  auto keysResult = rejectUnknownKeys(object, keys, path);
  if (keysResult.hasError()) {
    return keysResult;
  }
  auto result =
      readEnumField(object, "mode", path, settings.antiAliasing.mode,
                    {{"None", AntiAliasingMode::None},
                     {"TAA", AntiAliasingMode::TAA},
                     {"SpatialFallback", AntiAliasingMode::SpatialFallback},
                     {"MSAA4x", AntiAliasingMode::MSAA4x},
                     {"MSAA8x", AntiAliasingMode::MSAA8x}});
  if (result.hasError()) {
    return result;
  }
  result = readEnumField(
      object, "temporalProvider", path, settings.antiAliasing.temporalProvider,
      {{"Legacy", TemporalReconstructionProvider::Legacy},
       {"Reference", TemporalReconstructionProvider::Reference},
       {"External", TemporalReconstructionProvider::External}});
  if (result.hasError()) {
    return result;
  }
  result = readEnumField(object, "qualityPreset", path,
                         settings.antiAliasing.qualityPreset,
                         {{"Performance", TemporalAAQualityPreset::Performance},
                          {"Balanced", TemporalAAQualityPreset::Balanced},
                          {"Quality", TemporalAAQualityPreset::Quality},
                          {"Ultra", TemporalAAQualityPreset::Ultra},
                          {"Custom", TemporalAAQualityPreset::Custom}});
  if (result.hasError()) {
    return result;
  }
  auto cleanup = readBool(object, "spatialPostMsaaCleanup", path,
                          settings.antiAliasing.debug.spatialPostMsaaCleanup);
  if (cleanup.hasError()) {
    return Result<bool, std::string>::makeError(cleanup.error());
  }
  settings.antiAliasing.debug.spatialPostMsaaCleanup = cleanup.value();
  return Result<bool, std::string>::makeResult(true);
}

[[nodiscard]] Result<bool, std::string>
parseAmbientOcclusionSettings(yyjson_val *object, RenderSettings &settings,
                              std::string_view path) {
  static constexpr std::array keys{std::string_view("mode"),
                                   std::string_view("preset")};
  auto keysResult = rejectUnknownKeys(object, keys, path);
  if (keysResult.hasError()) {
    return keysResult;
  }
  auto result =
      readEnumField(object, "mode", path, settings.ambientOcclusion.mode,
                    {{"Disabled", AmbientOcclusionMode::Disabled},
                     {"GTAO", AmbientOcclusionMode::GTAO}});
  if (result.hasError()) {
    return result;
  }
  return readEnumField(object, "preset", path, settings.ambientOcclusion.preset,
                       {{"Low", AmbientOcclusionPreset::Low},
                        {"Balanced", AmbientOcclusionPreset::Balanced},
                        {"High", AmbientOcclusionPreset::High},
                        {"Ultra", AmbientOcclusionPreset::Ultra},
                        {"Custom", AmbientOcclusionPreset::Custom}});
}

[[nodiscard]] Result<bool, std::string>
parseShadowSettings(yyjson_val *object, RenderSettings &settings,
                    std::string_view path) {
  static constexpr std::array keys{
      std::string_view("enabled"),
      std::string_view("qualityPreset"),
      std::string_view("depthFormat"),
      std::string_view("maxDistance"),
      std::string_view("maxDistanceFadeFraction"),
      std::string_view("splitLambda"),
      std::string_view("cascadeBlendFraction"),
      std::string_view("pcfSampleCount"),
      std::string_view("sdsmTemporalBlend"),
      std::string_view("enableCascadeCasterCulling")};
  auto keysResult = rejectUnknownKeys(object, keys, path);
  if (keysResult.hasError()) {
    return keysResult;
  }
  auto enabled = readBool(object, "enabled", path, settings.shadow.enabled);
  if (enabled.hasError()) {
    return Result<bool, std::string>::makeError(enabled.error());
  }
  settings.shadow.enabled = enabled.value();
  ShadowQualityPreset preset = settings.shadow.qualityPreset;
  auto presetResult = readEnumField(object, "qualityPreset", path, preset,
                                    {{"Custom", ShadowQualityPreset::Custom},
                                     {"Low", ShadowQualityPreset::Low},
                                     {"Medium", ShadowQualityPreset::Medium},
                                     {"High", ShadowQualityPreset::High},
                                     {"Ultra", ShadowQualityPreset::Ultra}});
  if (presetResult.hasError()) {
    return presetResult;
  }
  if (optionalObject(object, "qualityPreset") != nullptr) {
    applyShadowQualityPreset(settings.shadow, preset);
  }
  auto enumResult = readEnumField(
      object, "depthFormat", path, settings.shadow.depthFormat,
      {{"D16_UNORM", Format::D16_UNORM}, {"D32_FLOAT", Format::D32_FLOAT}});
  if (enumResult.hasError()) {
    return enumResult;
  }
  auto number =
      readDouble(object, "maxDistance", path, settings.shadow.maxDistance);
  if (number.hasError()) {
    return Result<bool, std::string>::makeError(number.error());
  }
  settings.shadow.maxDistance = static_cast<float>(number.value());
  number = readDouble(object, "maxDistanceFadeFraction", path,
                      settings.shadow.maxDistanceFadeFraction);
  if (number.hasError()) {
    return Result<bool, std::string>::makeError(number.error());
  }
  settings.shadow.maxDistanceFadeFraction = static_cast<float>(number.value());
  number = readDouble(object, "splitLambda", path, settings.shadow.splitLambda);
  if (number.hasError()) {
    return Result<bool, std::string>::makeError(number.error());
  }
  settings.shadow.splitLambda = static_cast<float>(number.value());
  number = readDouble(object, "cascadeBlendFraction", path,
                      settings.shadow.cascadeBlendFraction);
  if (number.hasError()) {
    return Result<bool, std::string>::makeError(number.error());
  }
  settings.shadow.cascadeBlendFraction = static_cast<float>(number.value());
  auto count =
      readU32(object, "pcfSampleCount", path, settings.shadow.pcfSampleCount);
  if (count.hasError()) {
    return Result<bool, std::string>::makeError(count.error());
  }
  settings.shadow.pcfSampleCount = count.value();
  number = readDouble(object, "sdsmTemporalBlend", path,
                      settings.shadow.sdsmTemporalBlend);
  if (number.hasError()) {
    return Result<bool, std::string>::makeError(number.error());
  }
  settings.shadow.sdsmTemporalBlend = static_cast<float>(number.value());
  enabled = readBool(object, "enableCascadeCasterCulling", path,
                     settings.shadow.debug.enableCascadeCasterCulling);
  if (enabled.hasError()) {
    return Result<bool, std::string>::makeError(enabled.error());
  }
  settings.shadow.debug.enableCascadeCasterCulling = enabled.value();
  return Result<bool, std::string>::makeResult(true);
}

[[nodiscard]] Result<bool, std::string>
parseVisibilitySettings(yyjson_val *object, RenderSettings &settings,
                        std::string_view path) {
  static constexpr std::array keys{
      std::string_view("mainViewMode"),
      std::string_view("shadowMode"),
      std::string_view("occlusionMode"),
      std::string_view("enableMeshletFrustumCulling"),
      std::string_view("enableMeshletConeCulling"),
      std::string_view("enableGpuInstanceCulling"),
      std::string_view("enableGpuIndirectDraw"),
      std::string_view("enableIndirectMeshDispatch"),
      std::string_view("enableMeshletPreTaskCompaction"),
      std::string_view("visibleOnUncertain")};
  auto keysResult = rejectUnknownKeys(object, keys, path);
  if (keysResult.hasError()) {
    return keysResult;
  }
  auto result = readEnumField(
      object, "mainViewMode", path, settings.visibility.mainViewMode,
      {{"Disabled", VisibilityCullingMode::Disabled},
       {"CpuCoarse", VisibilityCullingMode::CpuCoarse},
       {"Hybrid", VisibilityCullingMode::Hybrid},
       {"GpuDriven", VisibilityCullingMode::GpuDriven}});
  if (result.hasError()) {
    return result;
  }
  result =
      readEnumField(object, "shadowMode", path, settings.visibility.shadowMode,
                    {{"Disabled", VisibilityCullingMode::Disabled},
                     {"CpuCoarse", VisibilityCullingMode::CpuCoarse},
                     {"Hybrid", VisibilityCullingMode::Hybrid},
                     {"GpuDriven", VisibilityCullingMode::GpuDriven}});
  if (result.hasError()) {
    return result;
  }
  result = readEnumField(
      object, "occlusionMode", path, settings.visibility.occlusionMode,
      {{"Disabled", VisibilityOcclusionMode::Disabled},
       {"PreviousFrameHiZ", VisibilityOcclusionMode::PreviousFrameHiZ},
       {"CurrentFrameHiZExperimental",
        VisibilityOcclusionMode::CurrentFrameHiZExperimental}});
  if (result.hasError()) {
    return result;
  }
  auto b = readBool(object, "enableMeshletFrustumCulling", path,
                    settings.visibility.enableMeshletFrustumCulling);
  if (b.hasError()) {
    return Result<bool, std::string>::makeError(b.error());
  }
  settings.visibility.enableMeshletFrustumCulling = b.value();
  b = readBool(object, "enableMeshletConeCulling", path,
               settings.visibility.enableMeshletConeCulling);
  if (b.hasError()) {
    return Result<bool, std::string>::makeError(b.error());
  }
  settings.visibility.enableMeshletConeCulling = b.value();
  b = readBool(object, "enableGpuInstanceCulling", path,
               settings.visibility.enableGpuInstanceCulling);
  if (b.hasError()) {
    return Result<bool, std::string>::makeError(b.error());
  }
  settings.visibility.enableGpuInstanceCulling = b.value();
  b = readBool(object, "enableGpuIndirectDraw", path,
               settings.visibility.enableGpuIndirectDraw);
  if (b.hasError()) {
    return Result<bool, std::string>::makeError(b.error());
  }
  settings.visibility.enableGpuIndirectDraw = b.value();
  b = readBool(object, "enableIndirectMeshDispatch", path,
               settings.visibility.enableIndirectMeshDispatch);
  if (b.hasError()) {
    return Result<bool, std::string>::makeError(b.error());
  }
  settings.visibility.enableIndirectMeshDispatch = b.value();
  b = readBool(object, "enableMeshletPreTaskCompaction", path,
               settings.visibility.enableMeshletPreTaskCompaction);
  if (b.hasError()) {
    return Result<bool, std::string>::makeError(b.error());
  }
  settings.visibility.enableMeshletPreTaskCompaction = b.value();
  b = readBool(object, "visibleOnUncertain", path,
               settings.visibility.visibleOnUncertain);
  if (b.hasError()) {
    return Result<bool, std::string>::makeError(b.error());
  }
  settings.visibility.visibleOnUncertain = b.value();
  return Result<bool, std::string>::makeResult(true);
}

[[nodiscard]] Result<bool, std::string>
parseHdrSettings(yyjson_val *object, RenderSettings &settings,
                 std::string_view path) {
  static constexpr std::array keys{std::string_view("bloomEnabled"),
                                   std::string_view("adaptationEnabled")};
  auto keysResult = rejectUnknownKeys(object, keys, path);
  if (keysResult.hasError()) {
    return keysResult;
  }
  auto b = readBool(object, "bloomEnabled", path,
                    settings.hdrPostProcess.bloomEnabled);
  if (b.hasError()) {
    return Result<bool, std::string>::makeError(b.error());
  }
  settings.hdrPostProcess.bloomEnabled = b.value();
  b = readBool(object, "adaptationEnabled", path,
               settings.hdrPostProcess.adaptationEnabled);
  if (b.hasError()) {
    return Result<bool, std::string>::makeError(b.error());
  }
  settings.hdrPostProcess.adaptationEnabled = b.value();
  return Result<bool, std::string>::makeResult(true);
}

[[nodiscard]] Result<bool, std::string>
parseTextureFilteringSettings(yyjson_val *object, RenderSettings &settings,
                              std::string_view path) {
  static constexpr std::array keys{std::string_view("mode"),
                                   std::string_view("anisotropy")};
  auto keysResult = rejectUnknownKeys(object, keys, path);
  if (keysResult.hasError()) {
    return keysResult;
  }
  auto modeResult =
      readEnumField(object, "mode", path, settings.textureFiltering.mode,
                    {{"Bilinear", TextureFilterMode::Bilinear},
                     {"Trilinear", TextureFilterMode::Trilinear},
                     {"Anisotropic", TextureFilterMode::Anisotropic}});
  if (modeResult.hasError()) {
    return modeResult;
  }
  auto anisotropy =
      readU32(object, "anisotropy", path, settings.textureFiltering.anisotropy);
  if (anisotropy.hasError()) {
    return Result<bool, std::string>::makeError(anisotropy.error());
  }
  settings.textureFiltering.anisotropy =
      static_cast<uint8_t>(std::min<uint32_t>(anisotropy.value(), 255u));
  return Result<bool, std::string>::makeResult(true);
}

[[nodiscard]] Result<bool, std::string>
parseDDGISettings(yyjson_val *object, RenderSettings &settings,
                  std::string_view path) {
  static constexpr std::array keys{
      std::string_view("enabled"),
      std::string_view("preset"),
      std::string_view("raysPerProbe"),
      std::string_view("maxProbeUpdatesPerFrame"),
      std::string_view("maxRayQueriesPerFrame"),
      std::string_view("maxLocalLightsPerHit"),
      std::string_view("maxCandidateIntersectionsPerRay"),
      std::string_view("irradianceHysteresis"),
      std::string_view("distanceHysteresis"),
      std::string_view("changeIrradianceHysteresisScale"),
      std::string_view("changeDistanceHysteresisScale"),
      std::string_view("selfShadowBias"),
      std::string_view("multiBounceLuminanceClamp"),
      std::string_view("relocation"),
      std::string_view("classification"),
      std::string_view("multiBounce"),
      std::string_view("freezeUpdates"),
      std::string_view("showVolumes"),
      std::string_view("showProbes"),
      std::string_view("showSelectedProbeRays"),
      std::string_view("debugView"),
      std::string_view("coverage")};
  auto result = rejectUnknownKeys(object, keys, path);
  if (result.hasError()) {
    return result;
  }
  auto enabled = readBool(object, "enabled", path, settings.ddgi.enabled);
  if (enabled.hasError()) {
    return Result<bool, std::string>::makeError(enabled.error());
  }
  settings.ddgi.enabled = enabled.value();
  result = readEnumField(object, "preset", path, settings.ddgi.preset,
                         {{"Low", DDGIQualityPreset::Low},
                          {"Balanced", DDGIQualityPreset::Balanced},
                          {"High", DDGIQualityPreset::High},
                          {"Custom", DDGIQualityPreset::Custom}});
  if (result.hasError()) {
    return result;
  }
  for (const auto [key, output] :
       {std::pair<std::string_view, uint32_t *>{"raysPerProbe",
                                                &settings.ddgi.raysPerProbe},
        {"maxProbeUpdatesPerFrame", &settings.ddgi.maxProbeUpdatesPerFrame},
        {"maxRayQueriesPerFrame", &settings.ddgi.maxRayQueriesPerFrame},
        {"maxLocalLightsPerHit", &settings.ddgi.maxLocalLightsPerHit},
        {"maxCandidateIntersectionsPerRay",
         &settings.ddgi.maxCandidateIntersectionsPerRay}}) {
    auto value = readU32(object, key, path, *output);
    if (value.hasError()) {
      return Result<bool, std::string>::makeError(value.error());
    }
    *output = value.value();
  }
  for (const auto [key, output] :
       {std::pair<std::string_view, float *>{
            "irradianceHysteresis", &settings.ddgi.irradianceHysteresis},
        {"distanceHysteresis", &settings.ddgi.distanceHysteresis},
        {"changeIrradianceHysteresisScale",
         &settings.ddgi.changeIrradianceHysteresisScale},
        {"changeDistanceHysteresisScale",
         &settings.ddgi.changeDistanceHysteresisScale},
        {"selfShadowBias", &settings.ddgi.selfShadowBias},
        {"multiBounceLuminanceClamp",
         &settings.ddgi.multiBounceLuminanceClamp}}) {
    auto value = readDouble(object, key, path, *output);
    if (value.hasError()) {
      return Result<bool, std::string>::makeError(value.error());
    }
    *output = static_cast<float>(value.value());
  }
  for (const auto [key, output] :
       {std::pair<std::string_view, bool *>{"relocation",
                                            &settings.ddgi.relocation},
        {"classification", &settings.ddgi.classification},
        {"multiBounce", &settings.ddgi.multiBounce},
        {"freezeUpdates", &settings.ddgi.freezeUpdates},
        {"showVolumes", &settings.ddgi.showVolumes},
        {"showProbes", &settings.ddgi.showProbes},
        {"showSelectedProbeRays", &settings.ddgi.showSelectedProbeRays}}) {
    auto value = readBool(object, key, path, *output);
    if (value.hasError()) {
      return Result<bool, std::string>::makeError(value.error());
    }
    *output = value.value();
  }
  result = readEnumField(object, "debugView", path, settings.ddgi.debugView,
                         {{"None", DDGIDebugView::None},
                          {"DiffuseIndirect", DDGIDebugView::DiffuseIndirect},
                          {"VolumeId", DDGIDebugView::VolumeId},
                          {"ProbeWeights", DDGIDebugView::ProbeWeights},
                          {"Confidence", DDGIDebugView::Confidence},
                          {"Visibility", DDGIDebugView::Visibility},
                          {"Irradiance", DDGIDebugView::Irradiance},
                          {"DistanceMean", DDGIDebugView::DistanceMean},
                          {"DistanceVariance", DDGIDebugView::DistanceVariance},
                          {"Classification", DDGIDebugView::Classification},
                          {"RelocationOffset", DDGIDebugView::RelocationOffset},
                          {"UpdateAge", DDGIDebugView::UpdateAge},
                          {"LeakRisk", DDGIDebugView::LeakRisk}});
  if (result.hasError()) {
    return result;
  }
  if (yyjson_val *coverage = optionalObject(object, "coverage")) {
    result = parseDDGICoverageSettings(coverage, settings.ddgi.coverage,
                                       jsonPath(path, "coverage"));
    if (result.hasError()) {
      return result;
    }
  }
  return Result<bool, std::string>::makeResult(true);
}

[[nodiscard]] Result<bool, std::string>
parseSettings(yyjson_val *object, RenderSettings &settings) {
  static constexpr std::array keys{
      std::string_view("opaque"),           std::string_view("antiAliasing"),
      std::string_view("ambientOcclusion"), std::string_view("shadow"),
      std::string_view("visibility"),       std::string_view("hdrPostProcess"),
      std::string_view("transmission"),     std::string_view("transparent"),
      std::string_view("textureFiltering"), std::string_view("ddgi"),
  };
  auto keysResult = rejectUnknownKeys(object, keys, "settings");
  if (keysResult.hasError()) {
    return keysResult;
  }
  struct Parser {
    std::string_view key;
    Result<bool, std::string> (*parse)(yyjson_val *, RenderSettings &,
                                       std::string_view);
  };
  static constexpr std::array parsers{
      Parser{"opaque", parseOpaqueSettings},
      Parser{"antiAliasing", parseAntiAliasingSettings},
      Parser{"ambientOcclusion", parseAmbientOcclusionSettings},
      Parser{"shadow", parseShadowSettings},
      Parser{"visibility", parseVisibilitySettings},
      Parser{"hdrPostProcess", parseHdrSettings},
      Parser{"textureFiltering", parseTextureFilteringSettings},
      Parser{"ddgi", parseDDGISettings},
  };
  for (const Parser &parser : parsers) {
    yyjson_val *sub = optionalObject(object, parser.key);
    if (sub == nullptr) {
      continue;
    }
    auto result = parser.parse(sub, settings, jsonPath("settings", parser.key));
    if (result.hasError()) {
      return result;
    }
  }
  yyjson_val *transmission = optionalObject(object, "transmission");
  if (transmission != nullptr) {
    static constexpr std::array transmissionKeys{std::string_view("enabled")};
    auto result = rejectUnknownKeys(transmission, transmissionKeys,
                                    "settings.transmission");
    if (result.hasError()) {
      return result;
    }
    auto enabled = readBool(transmission, "enabled", "settings.transmission",
                            settings.transmission.enabled);
    if (enabled.hasError()) {
      return Result<bool, std::string>::makeError(enabled.error());
    }
    settings.transmission.enabled = enabled.value();
  }
  yyjson_val *transparent = optionalObject(object, "transparent");
  if (transparent != nullptr) {
    static constexpr std::array transparentKeys{std::string_view("enabled")};
    auto result =
        rejectUnknownKeys(transparent, transparentKeys, "settings.transparent");
    if (result.hasError()) {
      return result;
    }
    auto enabled = readBool(transparent, "enabled", "settings.transparent",
                            settings.transparent.enabled);
    if (enabled.hasError()) {
      return Result<bool, std::string>::makeError(enabled.error());
    }
    settings.transparent.enabled = enabled.value();
  }
  sanitizeBenchmarkRenderSettings(settings);
  return Result<bool, std::string>::makeResult(true);
}

[[nodiscard]] Result<bool, std::string>
parseScene(yyjson_val *object, BenchmarkSceneConfig &scene) {
  static constexpr std::array sceneKeys{
      std::string_view("kind"),      std::string_view("pathBase"),
      std::string_view("path"),      std::string_view("importOptions"),
      std::string_view("baseModel"), std::string_view("generator"),
      std::string_view("seed"),      std::string_view("contentHash"),
  };
  auto result = rejectUnknownKeys(object, sceneKeys, "scene");
  if (result.hasError()) {
    return result;
  }
  auto text = readString(object, "kind", "scene", false, scene.kind);
  if (text.hasError()) {
    return Result<bool, std::string>::makeError(text.error());
  }
  scene.kind = std::move(text.value());
  if (scene.kind != "procedural" && scene.kind != "prefab") {
    return Result<bool, std::string>::makeError(
        "scene.kind must be procedural or prefab");
  }
  text = readString(object, "pathBase", "scene", false, scene.pathBase);
  if (text.hasError()) {
    return Result<bool, std::string>::makeError(text.error());
  }
  scene.pathBase = std::move(text.value());
  if (!scene.pathBase.empty() && scene.pathBase != "repoRoot" &&
      scene.pathBase != "modelsRoot" && scene.pathBase != "texturesRoot") {
    return Result<bool, std::string>::makeError(
        "scene.pathBase must be repoRoot, modelsRoot, or texturesRoot");
  }
  text = readString(object, "path", "scene", false, scene.path.string());
  if (text.hasError()) {
    return Result<bool, std::string>::makeError(text.error());
  }
  scene.path = text.value();
  text = readString(object, "generator", "scene", false, scene.generator);
  if (text.hasError()) {
    return Result<bool, std::string>::makeError(text.error());
  }
  scene.generator = std::move(text.value());
  text = readString(object, "contentHash", "scene", false, scene.contentHash);
  if (text.hasError()) {
    return Result<bool, std::string>::makeError(text.error());
  }
  scene.contentHash = std::move(text.value());
  auto seed = readU32(object, "seed", "scene", scene.seed);
  if (seed.hasError()) {
    return Result<bool, std::string>::makeError(seed.error());
  }
  scene.seed = seed.value();

  yyjson_val *importOptions = optionalObject(object, "importOptions");
  if (importOptions != nullptr) {
    static constexpr std::array importKeys{std::string_view("mesh")};
    result =
        rejectUnknownKeys(importOptions, importKeys, "scene.importOptions");
    if (result.hasError()) {
      return result;
    }
    yyjson_val *mesh = optionalObject(importOptions, "mesh");
    if (mesh != nullptr) {
      static constexpr std::array meshKeys{
          std::string_view("flipUVs"), std::string_view("generateMeshlets"),
          std::string_view("meshletMaxVertices"),
          std::string_view("meshletMaxPrimitives"),
          std::string_view("meshletConeWeight")};
      result = rejectUnknownKeys(mesh, meshKeys, "scene.importOptions.mesh");
      if (result.hasError()) {
        return result;
      }
      auto flip =
          readBool(mesh, "flipUVs", "scene.importOptions.mesh", scene.flipUVs);
      if (flip.hasError()) {
        return Result<bool, std::string>::makeError(flip.error());
      }
      scene.flipUVs = flip.value();
      auto generateMeshlets =
          readBool(mesh, "generateMeshlets", "scene.importOptions.mesh",
                   scene.generateMeshlets);
      if (generateMeshlets.hasError()) {
        return Result<bool, std::string>::makeError(generateMeshlets.error());
      }
      scene.generateMeshlets = generateMeshlets.value();
      auto u32 = readU32(mesh, "meshletMaxVertices", "scene.importOptions.mesh",
                         scene.meshletMaxVertices);
      if (u32.hasError()) {
        return Result<bool, std::string>::makeError(u32.error());
      }
      scene.meshletMaxVertices = u32.value();
      u32 = readU32(mesh, "meshletMaxPrimitives", "scene.importOptions.mesh",
                    scene.meshletMaxPrimitives);
      if (u32.hasError()) {
        return Result<bool, std::string>::makeError(u32.error());
      }
      scene.meshletMaxPrimitives = u32.value();
      auto real =
          readDouble(mesh, "meshletConeWeight", "scene.importOptions.mesh",
                     scene.meshletConeWeight);
      if (real.hasError()) {
        return Result<bool, std::string>::makeError(real.error());
      }
      scene.meshletConeWeight = static_cast<float>(real.value());
      if (scene.meshletMaxVertices == 0u || scene.meshletMaxPrimitives == 0u ||
          !std::isfinite(scene.meshletConeWeight) ||
          scene.meshletConeWeight < 0.0f || scene.meshletConeWeight > 1.0f) {
        return Result<bool, std::string>::makeError(
            "scene.importOptions.mesh has invalid meshlet ranges");
      }
    }
  }

  yyjson_val *baseModel = optionalObject(object, "baseModel");
  if (baseModel != nullptr) {
    static constexpr std::array baseKeys{
        std::string_view("kind"), std::string_view("targetRadius"),
        std::string_view("minScale"), std::string_view("maxScale")};
    result = rejectUnknownKeys(baseModel, baseKeys, "scene.baseModel");
    if (result.hasError()) {
      return result;
    }
    text = readString(baseModel, "kind", "scene.baseModel", false,
                      scene.baseModelKind);
    if (text.hasError()) {
      return Result<bool, std::string>::makeError(text.error());
    }
    scene.baseModelKind = std::move(text.value());
    if (!scene.baseModelKind.empty() && scene.baseModelKind != "fitRadius") {
      return Result<bool, std::string>::makeError(
          "scene.baseModel.kind must be fitRadius");
    }
    auto number = readDouble(baseModel, "targetRadius", "scene.baseModel",
                             scene.baseModelTargetRadius);
    if (number.hasError()) {
      return Result<bool, std::string>::makeError(number.error());
    }
    scene.baseModelTargetRadius = number.value();
    number = readDouble(baseModel, "minScale", "scene.baseModel",
                        scene.baseModelMinScale);
    if (number.hasError()) {
      return Result<bool, std::string>::makeError(number.error());
    }
    scene.baseModelMinScale = number.value();
    number = readDouble(baseModel, "maxScale", "scene.baseModel",
                        scene.baseModelMaxScale);
    if (number.hasError()) {
      return Result<bool, std::string>::makeError(number.error());
    }
    scene.baseModelMaxScale = number.value();
    if (!std::isfinite(scene.baseModelTargetRadius) ||
        !std::isfinite(scene.baseModelMinScale) ||
        !std::isfinite(scene.baseModelMaxScale) ||
        scene.baseModelTargetRadius <= 0.0 || scene.baseModelMinScale <= 0.0 ||
        scene.baseModelMinScale > scene.baseModelMaxScale) {
      return Result<bool, std::string>::makeError(
          "scene.baseModel has invalid scale ranges");
    }
  }
  if (scene.kind == "prefab" &&
      (scene.pathBase.empty() || scene.path.empty())) {
    return Result<bool, std::string>::makeError(
        "prefab scenes require scene.pathBase and scene.path");
  }
  return Result<bool, std::string>::makeResult(true);
}

[[nodiscard]] Result<bool, std::string>
parseCamera(yyjson_val *object, BenchmarkCameraConfig &camera) {
  static constexpr std::array keys{
      std::string_view("position"),  std::string_view("direction"),
      std::string_view("target"),    std::string_view("verticalFovDegrees"),
      std::string_view("nearPlane"), std::string_view("farPlane")};
  auto result = rejectUnknownKeys(object, keys, "camera");
  if (result.hasError()) {
    return result;
  }
  auto vec = readVec3(object, "position", "camera", camera.position);
  if (vec.hasError()) {
    return Result<bool, std::string>::makeError(vec.error());
  }
  camera.position = vec.value();
  if (!isFiniteVec3(camera.position)) {
    return Result<bool, std::string>::makeError(
        "camera.position entries must be finite");
  }
  if (optionalObject(object, "target") != nullptr) {
    vec = readVec3(object, "target", "camera",
                   camera.position + camera.direction);
    if (vec.hasError()) {
      return Result<bool, std::string>::makeError(vec.error());
    }
    camera.target = vec.value();
    if (!isFiniteVec3(camera.target) ||
        glm::length(camera.target - camera.position) <= 1.0e-6f) {
      return Result<bool, std::string>::makeError(
          "camera.target must be finite and differ from camera.position");
    }
    camera.hasTarget = true;
    camera.direction = glm::normalize(camera.target - camera.position);
  } else {
    vec = readVec3(object, "direction", "camera", camera.direction);
    if (vec.hasError()) {
      return Result<bool, std::string>::makeError(vec.error());
    }
    if (!isFiniteVec3(vec.value()) || glm::length(vec.value()) <= 1.0e-6f) {
      return Result<bool, std::string>::makeError(
          "camera.direction must be a finite non-zero vector");
    }
    camera.direction = glm::normalize(vec.value());
    camera.target = camera.position + camera.direction;
    camera.hasTarget = false;
  }
  auto number = readDouble(object, "verticalFovDegrees", "camera",
                           camera.verticalFovDegrees);
  if (number.hasError()) {
    return Result<bool, std::string>::makeError(number.error());
  }
  camera.verticalFovDegrees = static_cast<float>(number.value());
  number = readDouble(object, "nearPlane", "camera", camera.nearPlane);
  if (number.hasError()) {
    return Result<bool, std::string>::makeError(number.error());
  }
  camera.nearPlane = static_cast<float>(number.value());
  number = readDouble(object, "farPlane", "camera", camera.farPlane);
  if (number.hasError()) {
    return Result<bool, std::string>::makeError(number.error());
  }
  camera.farPlane = static_cast<float>(number.value());
  if (!std::isfinite(camera.verticalFovDegrees) ||
      camera.verticalFovDegrees <= 0.0f ||
      camera.verticalFovDegrees >= 180.0f || !std::isfinite(camera.nearPlane) ||
      camera.nearPlane <= 0.0f || !std::isfinite(camera.farPlane) ||
      camera.farPlane <= camera.nearPlane) {
    return Result<bool, std::string>::makeError(
        "camera requires 0 < verticalFovDegrees < 180 and 0 < nearPlane < "
        "farPlane");
  }
  return Result<bool, std::string>::makeResult(true);
}

[[nodiscard]] Result<BenchmarkTimeline, std::string>
parseTimeline(yyjson_val *root, const BenchmarkCameraConfig &baseCamera,
              uint32_t defaultEndFrame) {
  BenchmarkTimeline timeline{};
  yyjson_val *object = optionalObject(root, "timeline");
  if (object == nullptr) {
    return Result<BenchmarkTimeline, std::string>::makeResult(timeline);
  }
  static constexpr std::array keys{std::string_view("cameraPaths")};
  auto result = rejectUnknownKeys(object, keys, "timeline");
  if (result.hasError()) {
    return Result<BenchmarkTimeline, std::string>::makeError(result.error());
  }
  yyjson_val *paths = optionalObject(object, "cameraPaths");
  if (paths == nullptr) {
    return Result<BenchmarkTimeline, std::string>::makeResult(timeline);
  }
  if (!yyjson_is_arr(paths)) {
    return Result<BenchmarkTimeline, std::string>::makeError(
        "timeline.cameraPaths must be an array");
  }

  yyjson_arr_iter pathIter;
  yyjson_arr_iter_init(paths, &pathIter);
  yyjson_val *pathValue = nullptr;
  while ((pathValue = yyjson_arr_iter_next(&pathIter)) != nullptr) {
    if (!yyjson_is_obj(pathValue)) {
      return Result<BenchmarkTimeline, std::string>::makeError(
          "timeline.cameraPaths entries must be objects");
    }
    static constexpr std::array pathKeys{
        std::string_view("id"), std::string_view("startFrame"),
        std::string_view("endFrame"), std::string_view("interpolation"),
        std::string_view("keyframes")};
    result = rejectUnknownKeys(pathValue, pathKeys, "timeline.cameraPaths[]");
    if (result.hasError()) {
      return Result<BenchmarkTimeline, std::string>::makeError(result.error());
    }

    BenchmarkCameraPath path{};
    auto text = readString(pathValue, "id", "timeline.cameraPaths[]", true);
    if (text.hasError()) {
      return Result<BenchmarkTimeline, std::string>::makeError(text.error());
    }
    path.id = std::move(text.value());
    if (!isSafeDottedIdentifier(path.id)) {
      return Result<BenchmarkTimeline, std::string>::makeError(
          "timeline.cameraPaths[].id must be a safe lowercase identifier");
    }
    auto u32 = readU32(pathValue, "startFrame", "timeline.cameraPaths[]", 0u);
    if (u32.hasError()) {
      return Result<BenchmarkTimeline, std::string>::makeError(u32.error());
    }
    path.startFrame = u32.value();
    u32 = readU32(pathValue, "endFrame", "timeline.cameraPaths[]",
                  defaultEndFrame);
    if (u32.hasError()) {
      return Result<BenchmarkTimeline, std::string>::makeError(u32.error());
    }
    path.endFrame = u32.value();
    if (path.startFrame > path.endFrame) {
      return Result<BenchmarkTimeline, std::string>::makeError(
          "timeline.cameraPaths[] has invalid frame range");
    }
    text = readString(pathValue, "interpolation", "timeline.cameraPaths[]",
                      false, path.interpolation);
    if (text.hasError()) {
      return Result<BenchmarkTimeline, std::string>::makeError(text.error());
    }
    path.interpolation = std::move(text.value());
    if (path.interpolation != "linear" && path.interpolation != "smoothstep") {
      return Result<BenchmarkTimeline, std::string>::makeError(
          "timeline.cameraPaths[] interpolation must be linear or smoothstep");
    }

    yyjson_val *keyframes = optionalObject(pathValue, "keyframes");
    if (!yyjson_is_arr(keyframes) || yyjson_arr_size(keyframes) == 0u) {
      return Result<BenchmarkTimeline, std::string>::makeError(
          "timeline.cameraPaths[].keyframes must be a non-empty array");
    }
    yyjson_arr_iter keyframeIter;
    yyjson_arr_iter_init(keyframes, &keyframeIter);
    yyjson_val *keyframeValue = nullptr;
    while ((keyframeValue = yyjson_arr_iter_next(&keyframeIter)) != nullptr) {
      if (!yyjson_is_obj(keyframeValue)) {
        return Result<BenchmarkTimeline, std::string>::makeError(
            "timeline.cameraPaths[].keyframes entries must be objects");
      }
      static constexpr std::array keyframeKeys{std::string_view("frame"),
                                               std::string_view("position"),
                                               std::string_view("target")};
      result = rejectUnknownKeys(keyframeValue, keyframeKeys,
                                 "timeline.cameraPaths[].keyframes[]");
      if (result.hasError()) {
        return Result<BenchmarkTimeline, std::string>::makeError(
            result.error());
      }
      BenchmarkCameraKeyframe keyframe{};
      u32 = readU32(keyframeValue, "frame",
                    "timeline.cameraPaths[].keyframes[]", path.startFrame);
      if (u32.hasError()) {
        return Result<BenchmarkTimeline, std::string>::makeError(u32.error());
      }
      keyframe.frame = u32.value();
      if (keyframe.frame < path.startFrame || keyframe.frame > path.endFrame) {
        return Result<BenchmarkTimeline, std::string>::makeError(
            "camera keyframe is outside its path range");
      }
      if (optionalObject(keyframeValue, "position") == nullptr) {
        return Result<BenchmarkTimeline, std::string>::makeError(
            "timeline.cameraPaths[].keyframes[].position is required");
      }
      auto vec =
          readVec3(keyframeValue, "position",
                   "timeline.cameraPaths[].keyframes[]", baseCamera.position);
      if (vec.hasError()) {
        return Result<BenchmarkTimeline, std::string>::makeError(vec.error());
      }
      keyframe.position = vec.value();
      if (optionalObject(keyframeValue, "target") != nullptr) {
        vec = readVec3(keyframeValue, "target",
                       "timeline.cameraPaths[].keyframes[]", baseCamera.target);
        if (vec.hasError()) {
          return Result<BenchmarkTimeline, std::string>::makeError(vec.error());
        }
        keyframe.target = vec.value();
        keyframe.hasTarget = true;
      }
      path.keyframes.push_back(keyframe);
    }
    std::sort(path.keyframes.begin(), path.keyframes.end(),
              [](const BenchmarkCameraKeyframe &lhs,
                 const BenchmarkCameraKeyframe &rhs) {
                return lhs.frame < rhs.frame;
              });
    timeline.cameraPaths.push_back(std::move(path));
  }
  std::set<std::string> pathIds;
  for (const BenchmarkCameraPath &path : timeline.cameraPaths) {
    if (!pathIds.insert(path.id).second) {
      return Result<BenchmarkTimeline, std::string>::makeError(
          "duplicate timeline camera path id '" + path.id + "'");
    }
  }
  return Result<BenchmarkTimeline, std::string>::makeResult(
      std::move(timeline));
}

} // namespace

std::filesystem::path defaultBenchmarkCaseRoot() {
  return benchmarkRepoRoot() / "tools" / "cases" / "benchmarks";
}

Result<std::filesystem::path, std::string>
resolveBenchmarkPath(std::string_view base, const std::filesystem::path &path) {
  if (path.is_absolute()) {
    return Result<std::filesystem::path, std::string>::makeResult(path);
  }
  if (base == "repoRoot") {
    return Result<std::filesystem::path, std::string>::makeResult(
        benchmarkRepoRoot() / path);
  }
  auto configResult = loadRuntimeConfigFromEnvOrDefault();
  if (configResult.hasError()) {
    return Result<std::filesystem::path, std::string>::makeError(
        "resolveBenchmarkPath: " + configResult.error());
  }
  const RuntimeConfig &config = configResult.value();
  if (base == "modelsRoot") {
    return Result<std::filesystem::path, std::string>::makeResult(
        config.roots.models / path);
  }
  if (base == "texturesRoot") {
    return Result<std::filesystem::path, std::string>::makeResult(
        config.roots.textures / path);
  }
  return Result<std::filesystem::path, std::string>::makeError(
      "resolveBenchmarkPath: unsupported base '" + std::string(base) + "'");
}

Result<BenchmarkCase, std::string>
loadBenchmarkCaseManifest(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return Result<BenchmarkCase, std::string>::makeError(
        "loadBenchmarkCaseManifest: failed to open " + path.string());
  }
  std::string json((std::istreambuf_iterator<char>(file)),
                   std::istreambuf_iterator<char>());
  yyjson_read_err error{};
  JsonDocPtr doc(yyjson_read_opts(json.data(), json.size(), 0, nullptr, &error),
                 &yyjson_doc_free);
  if (!doc) {
    return Result<BenchmarkCase, std::string>::makeError(
        "loadBenchmarkCaseManifest: JSON parse failed at byte " +
        std::to_string(error.pos) + ": " + error.msg);
  }
  yyjson_val *root = yyjson_doc_get_root(doc.get());
  static constexpr std::array rootKeys{
      std::string_view("schemaVersion"),
      std::string_view("id"),
      std::string_view("suite"),
      std::string_view("comparisonGroup"),
      std::string_view("variant"),
      std::string_view("description"),
      std::string_view("scene"),
      std::string_view("backend"),
      std::string_view("resolution"),
      std::string_view("fixedDeltaSeconds"),
      std::string_view("warmupFrames"),
      std::string_view("measurementFrames"),
      std::string_view("cooldownFrames"),
      std::string_view("maxDrainFrames"),
      std::string_view("drainTimeoutMs"),
      std::string_view("samples"),
      std::string_view("authoritative"),
      std::string_view("presentMode"),
      std::string_view("renderGraph"),
      std::string_view("camera"),
      std::string_view("timeline"),
      std::string_view("settings"),
      std::string_view("requirements"),
      std::string_view("thresholds"),
      std::string_view("requiredMetrics"),
  };
  auto keysResult = rejectUnknownKeys(root, rootKeys, "$");
  if (keysResult.hasError()) {
    return Result<BenchmarkCase, std::string>::makeError(keysResult.error());
  }

  BenchmarkCase out{};
  out.manifestPath = path;
  auto schema = readU32(root, "schemaVersion", "$", 1u);
  if (schema.hasError()) {
    return Result<BenchmarkCase, std::string>::makeError(schema.error());
  }
  out.schemaVersion = schema.value();
  if (out.schemaVersion != 1u) {
    return Result<BenchmarkCase, std::string>::makeError(
        "schemaVersion must be 1");
  }
  auto text = readString(root, "id", "$", true);
  if (text.hasError()) {
    return Result<BenchmarkCase, std::string>::makeError(text.error());
  }
  out.id = std::move(text.value());
  if (!isSafeDottedIdentifier(out.id)) {
    return Result<BenchmarkCase, std::string>::makeError(
        "$.id must be a safe lowercase dotted identifier");
  }
  text = readString(root, "suite", "$", true);
  if (text.hasError()) {
    return Result<BenchmarkCase, std::string>::makeError(text.error());
  }
  out.suite = std::move(text.value());
  if (!isSafeDottedIdentifier(out.suite)) {
    return Result<BenchmarkCase, std::string>::makeError(
        "$.suite must be a safe lowercase dotted identifier");
  }
  text = readString(root, "comparisonGroup", "$", false);
  if (text.hasError()) {
    return Result<BenchmarkCase, std::string>::makeError(text.error());
  }
  out.comparisonGroup = std::move(text.value());
  text = readString(root, "variant", "$", false);
  if (text.hasError()) {
    return Result<BenchmarkCase, std::string>::makeError(text.error());
  }
  out.variant = std::move(text.value());
  if (out.comparisonGroup.empty() != out.variant.empty()) {
    return Result<BenchmarkCase, std::string>::makeError(
        "comparisonGroup and variant must be provided together");
  }
  if (!out.comparisonGroup.empty() &&
      (!isSafeDottedIdentifier(out.comparisonGroup) ||
       !isSafeDottedIdentifier(out.variant))) {
    return Result<BenchmarkCase, std::string>::makeError(
        "comparisonGroup and variant must be safe lowercase dotted "
        "identifiers");
  }
  text = readString(root, "description", "$", false, out.description);
  if (text.hasError()) {
    return Result<BenchmarkCase, std::string>::makeError(text.error());
  }
  out.description = std::move(text.value());
  text = readString(root, "backend", "$", false, out.backend);
  if (text.hasError()) {
    return Result<BenchmarkCase, std::string>::makeError(text.error());
  }
  out.backend = std::move(text.value());
  if (out.backend != "default" && out.backend != "nvrhi") {
    return Result<BenchmarkCase, std::string>::makeError(
        "$.backend must be default or nvrhi");
  }
  text = readString(root, "presentMode", "$", false, out.presentMode);
  if (text.hasError()) {
    return Result<BenchmarkCase, std::string>::makeError(text.error());
  }
  out.presentMode = std::move(text.value());
  if (out.presentMode != "default" && out.presentMode != "immediate" &&
      out.presentMode != "mailbox" && out.presentMode != "fifo") {
    return Result<BenchmarkCase, std::string>::makeError(
        "$.presentMode must be default, immediate, mailbox, or fifo");
  }

  yyjson_val *resolution = optionalObject(root, "resolution");
  if (resolution != nullptr) {
    if (!yyjson_is_arr(resolution) || yyjson_arr_size(resolution) != 2u) {
      return Result<BenchmarkCase, std::string>::makeError(
          "resolution must be [width, height]");
    }
    for (uint32_t i = 0u; i < 2u; ++i) {
      yyjson_val *entry = yyjson_arr_get(resolution, i);
      if (!yyjson_is_uint(entry) || yyjson_get_uint(entry) > UINT32_MAX) {
        return Result<BenchmarkCase, std::string>::makeError(
            "resolution entries must be uint32");
      }
      out.resolution[i] = static_cast<uint32_t>(yyjson_get_uint(entry));
      if (out.resolution[i] == 0u) {
        return Result<BenchmarkCase, std::string>::makeError(
            "resolution entries must be greater than zero");
      }
    }
  }

  auto number =
      readDouble(root, "fixedDeltaSeconds", "$", out.fixedDeltaSeconds);
  if (number.hasError()) {
    return Result<BenchmarkCase, std::string>::makeError(number.error());
  }
  out.fixedDeltaSeconds = number.value();
  if (!std::isfinite(out.fixedDeltaSeconds) || out.fixedDeltaSeconds <= 0.0) {
    return Result<BenchmarkCase, std::string>::makeError(
        "$.fixedDeltaSeconds must be finite and greater than zero");
  }
  auto u32 = readU32(root, "warmupFrames", "$", out.warmupFrames);
  if (u32.hasError()) {
    return Result<BenchmarkCase, std::string>::makeError(u32.error());
  }
  out.warmupFrames = u32.value();
  u32 = readU32(root, "measurementFrames", "$", out.measurementFrames);
  if (u32.hasError()) {
    return Result<BenchmarkCase, std::string>::makeError(u32.error());
  }
  out.measurementFrames = u32.value();
  if (out.measurementFrames == 0u) {
    return Result<BenchmarkCase, std::string>::makeError(
        "$.measurementFrames must be greater than zero");
  }
  u32 = readU32(root, "cooldownFrames", "$", out.cooldownFrames);
  if (u32.hasError()) {
    return Result<BenchmarkCase, std::string>::makeError(u32.error());
  }
  out.cooldownFrames = u32.value();
  u32 = readU32(root, "maxDrainFrames", "$", out.maxDrainFrames);
  if (u32.hasError()) {
    return Result<BenchmarkCase, std::string>::makeError(u32.error());
  }
  out.maxDrainFrames = u32.value();
  u32 = readU32(root, "drainTimeoutMs", "$", out.drainTimeoutMs);
  if (u32.hasError()) {
    return Result<BenchmarkCase, std::string>::makeError(u32.error());
  }
  out.drainTimeoutMs = u32.value();
  u32 = readU32(root, "samples", "$", out.samples);
  if (u32.hasError()) {
    return Result<BenchmarkCase, std::string>::makeError(u32.error());
  }
  if (u32.value() == 0u) {
    return Result<BenchmarkCase, std::string>::makeError(
        "$.samples must be greater than zero");
  }
  out.samples = u32.value();
  auto boolean = readBool(root, "authoritative", "$", out.authoritative);
  if (boolean.hasError()) {
    return Result<BenchmarkCase, std::string>::makeError(boolean.error());
  }
  out.authoritative = boolean.value();

  yyjson_val *scene = optionalObject(root, "scene");
  if (scene != nullptr) {
    auto parsed = parseScene(scene, out.scene);
    if (parsed.hasError()) {
      return Result<BenchmarkCase, std::string>::makeError(parsed.error());
    }
  }
  yyjson_val *camera = optionalObject(root, "camera");
  if (camera != nullptr) {
    auto parsed = parseCamera(camera, out.camera);
    if (parsed.hasError()) {
      return Result<BenchmarkCase, std::string>::makeError(parsed.error());
    }
  }
  const uint32_t defaultTimelineEndFrame =
      out.warmupFrames + out.measurementFrames + out.cooldownFrames > 0u
          ? out.warmupFrames + out.measurementFrames + out.cooldownFrames - 1u
          : 0u;
  auto timeline = parseTimeline(root, out.camera, defaultTimelineEndFrame);
  if (timeline.hasError()) {
    return Result<BenchmarkCase, std::string>::makeError(timeline.error());
  }
  out.timeline = std::move(timeline.value());

  yyjson_val *settings = optionalObject(root, "settings");
  if (settings != nullptr) {
    auto parsed = parseSettings(settings, out.settings);
    if (parsed.hasError()) {
      return Result<BenchmarkCase, std::string>::makeError(parsed.error());
    }
  }

  yyjson_val *renderGraph = optionalObject(root, "renderGraph");
  if (renderGraph != nullptr) {
    static constexpr std::array keys{std::string_view("workerCount"),
                                     std::string_view("parallelCompile"),
                                     std::string_view("parallelRecording")};
    auto parsed = rejectUnknownKeys(renderGraph, keys, "renderGraph");
    if (parsed.hasError()) {
      return Result<BenchmarkCase, std::string>::makeError(parsed.error());
    }
    u32 = readU32(renderGraph, "workerCount", "renderGraph",
                  out.renderGraph.workerCount);
    if (u32.hasError()) {
      return Result<BenchmarkCase, std::string>::makeError(u32.error());
    }
    if (u32.value() == 0u) {
      return Result<BenchmarkCase, std::string>::makeError(
          "renderGraph.workerCount must be greater than zero");
    }
    out.renderGraph.workerCount = u32.value();
    boolean = readBool(renderGraph, "parallelCompile", "renderGraph",
                       out.renderGraph.parallelCompile);
    if (boolean.hasError()) {
      return Result<BenchmarkCase, std::string>::makeError(boolean.error());
    }
    out.renderGraph.parallelCompile = boolean.value();
    boolean = readBool(renderGraph, "parallelRecording", "renderGraph",
                       out.renderGraph.parallelRecording);
    if (boolean.hasError()) {
      return Result<BenchmarkCase, std::string>::makeError(boolean.error());
    }
    out.renderGraph.parallelRecording = boolean.value();
  }

  yyjson_val *requirements = optionalObject(root, "requirements");
  if (requirements != nullptr) {
    static constexpr std::array keys{std::string_view("assets"),
                                     std::string_view("backends"),
                                     std::string_view("allowVisibleWindow"),
                                     std::string_view("msaa4x"),
                                     std::string_view("accelerationStructure"),
                                     std::string_view("rayQuery")};
    auto parsed = rejectUnknownKeys(requirements, keys, "requirements");
    if (parsed.hasError()) {
      return Result<BenchmarkCase, std::string>::makeError(parsed.error());
    }
    auto array = readStringArray(requirements, "assets", "requirements");
    if (array.hasError()) {
      return Result<BenchmarkCase, std::string>::makeError(array.error());
    }
    out.requirements.assets = std::move(array.value());
    array = readStringArray(requirements, "backends", "requirements");
    if (array.hasError()) {
      return Result<BenchmarkCase, std::string>::makeError(array.error());
    }
    out.requirements.backends = std::move(array.value());
    for (const std::string &backend : out.requirements.backends) {
      if (backend != "default" && backend != "nvrhi") {
        return Result<BenchmarkCase, std::string>::makeError(
            "requirements.backends contains unsupported backend '" + backend +
            "'");
      }
    }
    boolean = readBool(requirements, "allowVisibleWindow", "requirements",
                       out.requirements.allowVisibleWindow);
    if (boolean.hasError()) {
      return Result<BenchmarkCase, std::string>::makeError(boolean.error());
    }
    out.requirements.allowVisibleWindow = boolean.value();
    boolean = readBool(requirements, "msaa4x", "requirements",
                       out.requirements.msaa4x);
    if (boolean.hasError()) {
      return Result<BenchmarkCase, std::string>::makeError(boolean.error());
    }
    out.requirements.msaa4x = boolean.value();
    boolean = readBool(requirements, "accelerationStructure", "requirements",
                       out.requirements.accelerationStructure);
    if (boolean.hasError()) {
      return Result<BenchmarkCase, std::string>::makeError(boolean.error());
    }
    out.requirements.accelerationStructure = boolean.value();
    boolean = readBool(requirements, "rayQuery", "requirements",
                       out.requirements.rayQuery);
    if (boolean.hasError()) {
      return Result<BenchmarkCase, std::string>::makeError(boolean.error());
    }
    out.requirements.rayQuery = boolean.value();
  }

  yyjson_val *thresholds = optionalObject(root, "thresholds");
  if (thresholds != nullptr) {
    static constexpr std::array keys{
        std::string_view("failPercent"), std::string_view("failAbsoluteMs"),
        std::string_view("warnPercent"), std::string_view("warnAbsoluteMs")};
    auto parsed = rejectUnknownKeys(thresholds, keys, "thresholds");
    if (parsed.hasError()) {
      return Result<BenchmarkCase, std::string>::makeError(parsed.error());
    }
    number = readDouble(thresholds, "failPercent", "thresholds",
                        out.thresholds.failPercent);
    if (number.hasError()) {
      return Result<BenchmarkCase, std::string>::makeError(number.error());
    }
    out.thresholds.failPercent = number.value();
    number = readDouble(thresholds, "failAbsoluteMs", "thresholds",
                        out.thresholds.failAbsoluteMs);
    if (number.hasError()) {
      return Result<BenchmarkCase, std::string>::makeError(number.error());
    }
    out.thresholds.failAbsoluteMs = number.value();
    number = readDouble(thresholds, "warnPercent", "thresholds",
                        out.thresholds.warnPercent);
    if (number.hasError()) {
      return Result<BenchmarkCase, std::string>::makeError(number.error());
    }
    out.thresholds.warnPercent = number.value();
    number = readDouble(thresholds, "warnAbsoluteMs", "thresholds",
                        out.thresholds.warnAbsoluteMs);
    if (number.hasError()) {
      return Result<BenchmarkCase, std::string>::makeError(number.error());
    }
    out.thresholds.warnAbsoluteMs = number.value();
    if (!std::isfinite(out.thresholds.failPercent) ||
        !std::isfinite(out.thresholds.failAbsoluteMs) ||
        !std::isfinite(out.thresholds.warnPercent) ||
        !std::isfinite(out.thresholds.warnAbsoluteMs) ||
        out.thresholds.failPercent < 0.0 ||
        out.thresholds.failAbsoluteMs < 0.0 ||
        out.thresholds.warnPercent < 0.0 ||
        out.thresholds.warnAbsoluteMs < 0.0 ||
        out.thresholds.warnPercent > out.thresholds.failPercent ||
        out.thresholds.warnAbsoluteMs > out.thresholds.failAbsoluteMs) {
      return Result<BenchmarkCase, std::string>::makeError(
          "thresholds must be finite, non-negative, and warn thresholds must "
          "not exceed fail thresholds");
    }
  }

  auto metrics = readStringArray(root, "requiredMetrics", "$");
  if (metrics.hasError()) {
    return Result<BenchmarkCase, std::string>::makeError(metrics.error());
  }
  out.requiredMetrics = std::move(metrics.value());
  if (out.requiredMetrics.empty()) {
    out.requiredMetrics.push_back("cpu.render_submit_ms");
  }
  std::set<std::string> requiredMetricIds;
  for (const std::string &metricId : out.requiredMetrics) {
    if (!isSafeDottedIdentifier(metricId)) {
      return Result<BenchmarkCase, std::string>::makeError(
          "requiredMetrics contains invalid metric id '" + metricId + "'");
    }
    if (!requiredMetricIds.insert(metricId).second) {
      return Result<BenchmarkCase, std::string>::makeError(
          "duplicate required metric id '" + metricId + "'");
    }
    auto descriptor =
        requireBenchmarkMetricDescriptor(metricId, "requiredMetrics");
    if (descriptor.hasError()) {
      return Result<BenchmarkCase, std::string>::makeError(descriptor.error());
    }
  }
  return Result<BenchmarkCase, std::string>::makeResult(std::move(out));
}

Result<std::vector<BenchmarkCase>, std::string>
discoverBenchmarkCases(const BenchmarkManifestLoadOptions &options) {
  const std::filesystem::path root =
      options.caseRoot.empty() ? defaultBenchmarkCaseRoot() : options.caseRoot;
  std::vector<BenchmarkCase> cases;
  auto manifests = nuri::tools::core::discoverCaseManifestPaths(root);
  if (manifests.hasError()) {
    return Result<std::vector<BenchmarkCase>, std::string>::makeError(
        manifests.error());
  }
  cases.reserve(manifests.value().size());
  for (const std::filesystem::path &manifest : manifests.value()) {
    auto loaded = loadBenchmarkCaseManifest(manifest);
    if (loaded.hasError()) {
      return Result<std::vector<BenchmarkCase>, std::string>::makeError(
          loaded.error());
    }
    cases.push_back(std::move(loaded.value()));
  }
  std::sort(cases.begin(), cases.end(),
            [](const BenchmarkCase &lhs, const BenchmarkCase &rhs) {
              return lhs.id < rhs.id;
            });
  std::vector<nuri::tools::core::CaseCatalogEntry> entries;
  entries.reserve(cases.size());
  for (const BenchmarkCase &benchmarkCase : cases) {
    entries.push_back({.id = benchmarkCase.id,
                       .suite = benchmarkCase.suite,
                       .manifestPath = benchmarkCase.manifestPath});
  }
  auto validCatalog =
      nuri::tools::core::validateCaseCatalog(entries, "benchmark");
  if (validCatalog.hasError()) {
    return Result<std::vector<BenchmarkCase>, std::string>::makeError(
        validCatalog.error());
  }
  std::set<std::string> experimentVariants;
  for (const BenchmarkCase &benchmarkCase : cases) {
    if (benchmarkCase.comparisonGroup.empty()) {
      continue;
    }
    const std::string identity =
        benchmarkCase.comparisonGroup + "\n" + benchmarkCase.variant;
    if (!experimentVariants.insert(identity).second) {
      return Result<std::vector<BenchmarkCase>, std::string>::makeError(
          "duplicate benchmark experiment variant '" +
          benchmarkCase.comparisonGroup + "/" + benchmarkCase.variant + "'");
    }
  }
  return Result<std::vector<BenchmarkCase>, std::string>::makeResult(
      std::move(cases));
}

const BenchmarkCase *
findBenchmarkCaseById(const std::vector<BenchmarkCase> &cases,
                      std::string_view id) {
  for (const BenchmarkCase &benchmarkCase : cases) {
    if (benchmarkCase.id == id) {
      return &benchmarkCase;
    }
  }
  return nullptr;
}

std::vector<const BenchmarkCase *>
filterBenchmarkCasesBySuite(const std::vector<BenchmarkCase> &cases,
                            std::string_view suite) {
  std::vector<const BenchmarkCase *> out;
  std::vector<nuri::tools::core::CaseCatalogEntry> entries;
  entries.reserve(cases.size());
  for (const BenchmarkCase &benchmarkCase : cases) {
    entries.push_back({.id = benchmarkCase.id,
                       .suite = benchmarkCase.suite,
                       .manifestPath = benchmarkCase.manifestPath});
  }
  auto selected = nuri::tools::core::selectCaseCatalog(
      entries,
      nuri::tools::core::CaseCatalogSelector{.suite = std::string(suite)},
      nuri::tools::core::CaseCatalogZeroMatchPolicy::Allow, "benchmark");
  if (selected.hasError()) {
    return out;
  }
  out.reserve(selected.value().size());
  for (const size_t index : selected.value()) {
    out.push_back(&cases[index]);
  }
  return out;
}

} // namespace nuri::tools::benchmark
