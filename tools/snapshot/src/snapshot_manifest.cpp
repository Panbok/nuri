#include "nuri/tools/snapshot/snapshot_manifest.h"

#include "nuri/core/runtime_config.h"
#include "nuri/tools/core/case_catalog.h"
#include "nuri/tools/snapshot/snapshot_capture_point.h"
#include "nuri/tools/snapshot/snapshot_compare.h"
#include "nuri/tools/snapshot/snapshot_environment.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <cwctype>
#include <filesystem>
#include <fstream>
#include <memory>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <system_error>

#if defined(_WIN32)
#include <windows.h>
#endif

#include <yyjson.h>

namespace nuri::tools::snapshot {
namespace {

using JsonDocPtr = std::unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)>;

[[nodiscard]] bool isReservedWindowsSegment(std::string_view segment) {
  static constexpr std::array names{
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
  return std::find(names.begin(), names.end(), segment) != names.end();
}

[[nodiscard]] bool equalPathComponent(const std::filesystem::path &lhs,
                                      const std::filesystem::path &rhs) {
#if defined(_WIN32)
  std::wstring left = lhs.native();
  std::wstring right = rhs.native();
  for (wchar_t &c : left) {
    c = static_cast<wchar_t>(std::towlower(c));
  }
  for (wchar_t &c : right) {
    c = static_cast<wchar_t>(std::towlower(c));
  }
  return left == right;
#else
  return lhs == rhs;
#endif
}

[[nodiscard]] bool pathIsUnder(const std::filesystem::path &root,
                               const std::filesystem::path &candidate) {
  auto rootIt = root.begin();
  auto candidateIt = candidate.begin();
  for (; rootIt != root.end(); ++rootIt, ++candidateIt) {
    if (candidateIt == candidate.end() ||
        !equalPathComponent(*rootIt, *candidateIt)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool isReparsePoint(const std::filesystem::path &path) {
#if defined(_WIN32)
  const DWORD attributes = GetFileAttributesW(path.c_str());
  return attributes != INVALID_FILE_ATTRIBUTES &&
         (attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0u;
#else
  std::error_code ec;
  return std::filesystem::is_symlink(std::filesystem::symlink_status(path, ec));
#endif
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
parseSettings(yyjson_val *object, RenderSettings &settings) {
  static constexpr std::array keys{
      std::string_view("opaque"),           std::string_view("antiAliasing"),
      std::string_view("ambientOcclusion"), std::string_view("shadow"),
      std::string_view("visibility"),       std::string_view("hdrPostProcess"),
      std::string_view("transmission"),     std::string_view("transparent"),
      std::string_view("textureFiltering"), std::string_view("ddgi"),
  };
  auto result = rejectUnknownKeys(object, keys, "settings");
  if (result.hasError()) {
    return result;
  }

  if (yyjson_val *opaque = optionalObject(object, "opaque")) {
    static constexpr std::array opaqueKeys{
        std::string_view("enabled"),
        std::string_view("enableDepthPrepass"),
        std::string_view("enableDepthPyramid"),
        std::string_view("enableInstanceCompute"),
        std::string_view("enableInstanceAnimation"),
        std::string_view("enableIndirectDraw"),
        std::string_view("enableInstancedDraw"),
        std::string_view("enableMeshLod"),
        std::string_view("meshLodTargetPixelError"),
        std::string_view("meshLodHysteresisRatio"),
        std::string_view("enableCpuFrustumCulling"),
        std::string_view("enableTessellation"),
        std::string_view("forcedMeshLod"),
        std::string_view("meshletMode"),
        std::string_view("enableMeshletFrustumCulling"),
        std::string_view("enableMeshletConeCulling"),
        std::string_view("hybridClassicMaxMeshlets")};
    result = rejectUnknownKeys(opaque, opaqueKeys, "settings.opaque");
    if (result.hasError()) {
      return result;
    }
    auto boolean =
        readBool(opaque, "enabled", "settings.opaque", settings.opaque.enabled);
    if (boolean.hasError()) {
      return Result<bool, std::string>::makeError(boolean.error());
    }
    settings.opaque.enabled = boolean.value();
    boolean = readBool(opaque, "enableDepthPrepass", "settings.opaque",
                       settings.opaque.enableDepthPrepass);
    if (boolean.hasError()) {
      return Result<bool, std::string>::makeError(boolean.error());
    }
    settings.opaque.enableDepthPrepass = boolean.value();
    boolean = readBool(opaque, "enableDepthPyramid", "settings.opaque",
                       settings.opaque.enableDepthPyramid);
    if (boolean.hasError()) {
      return Result<bool, std::string>::makeError(boolean.error());
    }
    settings.opaque.enableDepthPyramid = boolean.value();
    boolean = readBool(opaque, "enableInstanceCompute", "settings.opaque",
                       settings.opaque.enableInstanceCompute);
    if (boolean.hasError()) {
      return Result<bool, std::string>::makeError(boolean.error());
    }
    settings.opaque.enableInstanceCompute = boolean.value();
    boolean = readBool(opaque, "enableInstanceAnimation", "settings.opaque",
                       settings.opaque.enableInstanceAnimation);
    if (boolean.hasError()) {
      return Result<bool, std::string>::makeError(boolean.error());
    }
    settings.opaque.enableInstanceAnimation = boolean.value();
    boolean = readBool(opaque, "enableIndirectDraw", "settings.opaque",
                       settings.opaque.enableIndirectDraw);
    if (boolean.hasError()) {
      return Result<bool, std::string>::makeError(boolean.error());
    }
    settings.opaque.enableIndirectDraw = boolean.value();
    boolean = readBool(opaque, "enableInstancedDraw", "settings.opaque",
                       settings.opaque.enableInstancedDraw);
    if (boolean.hasError()) {
      return Result<bool, std::string>::makeError(boolean.error());
    }
    settings.opaque.enableInstancedDraw = boolean.value();
    boolean = readBool(opaque, "enableMeshLod", "settings.opaque",
                       settings.opaque.enableMeshLod);
    if (boolean.hasError()) {
      return Result<bool, std::string>::makeError(boolean.error());
    }
    settings.opaque.enableMeshLod = boolean.value();
    auto lodNumber =
        readDouble(opaque, "meshLodTargetPixelError", "settings.opaque",
                   settings.opaque.meshLodTargetPixelError);
    if (lodNumber.hasError()) {
      return Result<bool, std::string>::makeError(lodNumber.error());
    }
    settings.opaque.meshLodTargetPixelError =
        static_cast<float>(lodNumber.value());
    lodNumber = readDouble(opaque, "meshLodHysteresisRatio", "settings.opaque",
                           settings.opaque.meshLodHysteresisRatio);
    if (lodNumber.hasError()) {
      return Result<bool, std::string>::makeError(lodNumber.error());
    }
    settings.opaque.meshLodHysteresisRatio =
        static_cast<float>(lodNumber.value());
    boolean = readBool(opaque, "enableCpuFrustumCulling", "settings.opaque",
                       settings.opaque.enableCpuFrustumCulling);
    if (boolean.hasError()) {
      return Result<bool, std::string>::makeError(boolean.error());
    }
    settings.opaque.enableCpuFrustumCulling = boolean.value();
    result = readEnumField(opaque, "meshletMode", "settings.opaque",
                           settings.opaque.meshletMode,
                           {{"Disabled", MeshletRenderMode::Disabled},
                            {"Opportunistic", MeshletRenderMode::Opportunistic},
                            {"Required", MeshletRenderMode::Required}});
    if (result.hasError()) {
      return result;
    }
    auto hybridClassicMaxMeshlets =
        readU32(opaque, "hybridClassicMaxMeshlets", "settings.opaque",
                settings.opaque.hybridClassicMaxMeshlets);
    if (hybridClassicMaxMeshlets.hasError()) {
      return Result<bool, std::string>::makeError(
          hybridClassicMaxMeshlets.error());
    }
    settings.opaque.hybridClassicMaxMeshlets = hybridClassicMaxMeshlets.value();
    boolean = readBool(opaque, "enableMeshletFrustumCulling", "settings.opaque",
                       settings.opaque.enableMeshletFrustumCulling);
    if (boolean.hasError()) {
      return Result<bool, std::string>::makeError(boolean.error());
    }
    settings.opaque.enableMeshletFrustumCulling = boolean.value();
    boolean = readBool(opaque, "enableMeshletConeCulling", "settings.opaque",
                       settings.opaque.enableMeshletConeCulling);
    if (boolean.hasError()) {
      return Result<bool, std::string>::makeError(boolean.error());
    }
    settings.opaque.enableMeshletConeCulling = boolean.value();
    boolean = readBool(opaque, "enableTessellation", "settings.opaque",
                       settings.opaque.enableTessellation);
    if (boolean.hasError()) {
      return Result<bool, std::string>::makeError(boolean.error());
    }
    settings.opaque.enableTessellation = boolean.value();
    yyjson_val *forced = optionalObject(opaque, "forcedMeshLod");
    if (forced != nullptr) {
      if (yyjson_is_sint(forced) && yyjson_get_sint(forced) >= INT32_MIN &&
          yyjson_get_sint(forced) <= INT32_MAX) {
        settings.opaque.forcedMeshLod =
            static_cast<int32_t>(yyjson_get_sint(forced));
      } else if (yyjson_is_uint(forced) &&
                 yyjson_get_uint(forced) <= INT32_MAX) {
        settings.opaque.forcedMeshLod =
            static_cast<int32_t>(yyjson_get_uint(forced));
      } else {
        return Result<bool, std::string>::makeError(
            "settings.opaque.forcedMeshLod must be an int32");
      }
    }
  }

  if (yyjson_val *ddgi = optionalObject(object, "ddgi")) {
    static constexpr std::array ddgiKeys{
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
        std::string_view("debugView")};
    result = rejectUnknownKeys(ddgi, ddgiKeys, "settings.ddgi");
    if (result.hasError()) {
      return result;
    }
    auto boolean =
        readBool(ddgi, "enabled", "settings.ddgi", settings.ddgi.enabled);
    if (boolean.hasError()) {
      return Result<bool, std::string>::makeError(boolean.error());
    }
    settings.ddgi.enabled = boolean.value();
    result =
        readEnumField(ddgi, "preset", "settings.ddgi", settings.ddgi.preset,
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
      auto value = readU32(ddgi, key, "settings.ddgi", *output);
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
      auto value = readDouble(ddgi, key, "settings.ddgi", *output);
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
      auto value = readBool(ddgi, key, "settings.ddgi", *output);
      if (value.hasError()) {
        return Result<bool, std::string>::makeError(value.error());
      }
      *output = value.value();
    }
    result = readEnumField(
        ddgi, "debugView", "settings.ddgi", settings.ddgi.debugView,
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
  }

  if (yyjson_val *visibility = optionalObject(object, "visibility")) {
    static constexpr std::array visibilityKeys{
        std::string_view("mainViewMode"),
        std::string_view("shadowMode"),
        std::string_view("occlusionMode"),
        std::string_view("enableMeshletFrustumCulling"),
        std::string_view("enableMeshletConeCulling"),
        std::string_view("enableGpuInstanceCulling"),
        std::string_view("enableGpuIndirectDraw"),
        std::string_view("enableIndirectMeshDispatch"),
        std::string_view("enableMeshletPreTaskCompaction"),
        std::string_view("visibleOnUncertain"),
        std::string_view("debug")};
    result =
        rejectUnknownKeys(visibility, visibilityKeys, "settings.visibility");
    if (result.hasError()) {
      return result;
    }
    result = readEnumField(visibility, "mainViewMode", "settings.visibility",
                           settings.visibility.mainViewMode,
                           {{"Disabled", VisibilityCullingMode::Disabled},
                            {"CpuCoarse", VisibilityCullingMode::CpuCoarse},
                            {"Hybrid", VisibilityCullingMode::Hybrid},
                            {"GpuDriven", VisibilityCullingMode::GpuDriven}});
    if (result.hasError()) {
      return result;
    }
    result = readEnumField(visibility, "shadowMode", "settings.visibility",
                           settings.visibility.shadowMode,
                           {{"Disabled", VisibilityCullingMode::Disabled},
                            {"CpuCoarse", VisibilityCullingMode::CpuCoarse},
                            {"Hybrid", VisibilityCullingMode::Hybrid},
                            {"GpuDriven", VisibilityCullingMode::GpuDriven}});
    if (result.hasError()) {
      return result;
    }
    result = readEnumField(
        visibility, "occlusionMode", "settings.visibility",
        settings.visibility.occlusionMode,
        {{"Disabled", VisibilityOcclusionMode::Disabled},
         {"PreviousFrameHiZ", VisibilityOcclusionMode::PreviousFrameHiZ},
         {"CurrentFrameHiZExperimental",
          VisibilityOcclusionMode::CurrentFrameHiZExperimental}});
    if (result.hasError()) {
      return result;
    }
    auto boolean = readBool(visibility, "enableMeshletFrustumCulling",
                            "settings.visibility",
                            settings.visibility.enableMeshletFrustumCulling);
    if (boolean.hasError()) {
      return Result<bool, std::string>::makeError(boolean.error());
    }
    settings.visibility.enableMeshletFrustumCulling = boolean.value();
    boolean =
        readBool(visibility, "enableMeshletConeCulling", "settings.visibility",
                 settings.visibility.enableMeshletConeCulling);
    if (boolean.hasError()) {
      return Result<bool, std::string>::makeError(boolean.error());
    }
    settings.visibility.enableMeshletConeCulling = boolean.value();
    boolean =
        readBool(visibility, "enableGpuInstanceCulling", "settings.visibility",
                 settings.visibility.enableGpuInstanceCulling);
    if (boolean.hasError()) {
      return Result<bool, std::string>::makeError(boolean.error());
    }
    settings.visibility.enableGpuInstanceCulling = boolean.value();
    boolean =
        readBool(visibility, "enableGpuIndirectDraw", "settings.visibility",
                 settings.visibility.enableGpuIndirectDraw);
    if (boolean.hasError()) {
      return Result<bool, std::string>::makeError(boolean.error());
    }
    settings.visibility.enableGpuIndirectDraw = boolean.value();
    boolean = readBool(visibility, "enableIndirectMeshDispatch",
                       "settings.visibility",
                       settings.visibility.enableIndirectMeshDispatch);
    if (boolean.hasError()) {
      return Result<bool, std::string>::makeError(boolean.error());
    }
    settings.visibility.enableIndirectMeshDispatch = boolean.value();
    boolean = readBool(visibility, "enableMeshletPreTaskCompaction",
                       "settings.visibility",
                       settings.visibility.enableMeshletPreTaskCompaction);
    if (boolean.hasError()) {
      return Result<bool, std::string>::makeError(boolean.error());
    }
    settings.visibility.enableMeshletPreTaskCompaction = boolean.value();
    boolean = readBool(visibility, "visibleOnUncertain", "settings.visibility",
                       settings.visibility.visibleOnUncertain);
    if (boolean.hasError()) {
      return Result<bool, std::string>::makeError(boolean.error());
    }
    settings.visibility.visibleOnUncertain = boolean.value();
    if (yyjson_val *debug = optionalObject(visibility, "debug")) {
      static constexpr std::array debugKeys{
          std::string_view("showObjectBounds"),
          std::string_view("showMeshletBounds"),
          std::string_view("visualizeCullReason"),
          std::string_view("logCounters"),
          std::string_view("forcedVisibleListCapacity")};
      result = rejectUnknownKeys(debug, debugKeys, "settings.visibility.debug");
      if (result.hasError()) {
        return result;
      }
      boolean = readBool(debug, "showObjectBounds", "settings.visibility.debug",
                         settings.visibility.debug.showObjectBounds);
      if (boolean.hasError()) {
        return Result<bool, std::string>::makeError(boolean.error());
      }
      settings.visibility.debug.showObjectBounds = boolean.value();
      boolean =
          readBool(debug, "showMeshletBounds", "settings.visibility.debug",
                   settings.visibility.debug.showMeshletBounds);
      if (boolean.hasError()) {
        return Result<bool, std::string>::makeError(boolean.error());
      }
      settings.visibility.debug.showMeshletBounds = boolean.value();
      boolean =
          readBool(debug, "visualizeCullReason", "settings.visibility.debug",
                   settings.visibility.debug.visualizeCullReason);
      if (boolean.hasError()) {
        return Result<bool, std::string>::makeError(boolean.error());
      }
      settings.visibility.debug.visualizeCullReason = boolean.value();
      boolean = readBool(debug, "logCounters", "settings.visibility.debug",
                         settings.visibility.debug.logCounters);
      if (boolean.hasError()) {
        return Result<bool, std::string>::makeError(boolean.error());
      }
      settings.visibility.debug.logCounters = boolean.value();
      auto u32 = readU32(debug, "forcedVisibleListCapacity",
                         "settings.visibility.debug",
                         settings.visibility.debug.forcedVisibleListCapacity);
      if (u32.hasError()) {
        return Result<bool, std::string>::makeError(u32.error());
      }
      settings.visibility.debug.forcedVisibleListCapacity = u32.value();
    }
  }

  if (yyjson_val *aa = optionalObject(object, "antiAliasing")) {
    static constexpr std::array aaKeys{std::string_view("mode"),
                                       std::string_view("temporalProvider"),
                                       std::string_view("qualityPreset")};
    result = rejectUnknownKeys(aa, aaKeys, "settings.antiAliasing");
    if (result.hasError()) {
      return result;
    }
    result = readEnumField(
        aa, "mode", "settings.antiAliasing", settings.antiAliasing.mode,
        {{"None", AntiAliasingMode::None},
         {"TAA", AntiAliasingMode::TAA},
         {"SpatialFallback", AntiAliasingMode::SpatialFallback},
         {"MSAA4x", AntiAliasingMode::MSAA4x},
         {"MSAA8x", AntiAliasingMode::MSAA8x}});
    if (result.hasError()) {
      return result;
    }
    result =
        readEnumField(aa, "temporalProvider", "settings.antiAliasing",
                      settings.antiAliasing.temporalProvider,
                      {{"Legacy", TemporalReconstructionProvider::Legacy},
                       {"Reference", TemporalReconstructionProvider::Reference},
                       {"External", TemporalReconstructionProvider::External}});
    if (result.hasError()) {
      return result;
    }
    result =
        readEnumField(aa, "qualityPreset", "settings.antiAliasing",
                      settings.antiAliasing.qualityPreset,
                      {{"Performance", TemporalAAQualityPreset::Performance},
                       {"Balanced", TemporalAAQualityPreset::Balanced},
                       {"Quality", TemporalAAQualityPreset::Quality},
                       {"Ultra", TemporalAAQualityPreset::Ultra},
                       {"Custom", TemporalAAQualityPreset::Custom}});
    if (result.hasError()) {
      return result;
    }
  }

  if (yyjson_val *ao = optionalObject(object, "ambientOcclusion")) {
    static constexpr std::array aoKeys{std::string_view("mode"),
                                       std::string_view("preset")};
    result = rejectUnknownKeys(ao, aoKeys, "settings.ambientOcclusion");
    if (result.hasError()) {
      return result;
    }
    result = readEnumField(ao, "mode", "settings.ambientOcclusion",
                           settings.ambientOcclusion.mode,
                           {{"Disabled", AmbientOcclusionMode::Disabled},
                            {"GTAO", AmbientOcclusionMode::GTAO}});
    if (result.hasError()) {
      return result;
    }
    result = readEnumField(ao, "preset", "settings.ambientOcclusion",
                           settings.ambientOcclusion.preset,
                           {{"Low", AmbientOcclusionPreset::Low},
                            {"Balanced", AmbientOcclusionPreset::Balanced},
                            {"High", AmbientOcclusionPreset::High},
                            {"Ultra", AmbientOcclusionPreset::Ultra},
                            {"Custom", AmbientOcclusionPreset::Custom}});
    if (result.hasError()) {
      return result;
    }
  }

  if (yyjson_val *shadow = optionalObject(object, "shadow")) {
    static constexpr std::array shadowKeys{
        std::string_view("enabled"),
        std::string_view("qualityPreset"),
        std::string_view("depthFormat"),
        std::string_view("maxDistance"),
        std::string_view("maxDistanceFadeFraction"),
        std::string_view("splitLambda"),
        std::string_view("cascadeBlendFraction"),
        std::string_view("pcfSampleCount"),
        std::string_view("sdsmTemporalBlend"),
        std::string_view("enableCascadeCasterCulling"),
        std::string_view("debug")};
    result = rejectUnknownKeys(shadow, shadowKeys, "settings.shadow");
    if (result.hasError()) {
      return result;
    }
    auto enabled =
        readBool(shadow, "enabled", "settings.shadow", settings.shadow.enabled);
    if (enabled.hasError()) {
      return Result<bool, std::string>::makeError(enabled.error());
    }
    settings.shadow.enabled = enabled.value();
    ShadowQualityPreset preset = settings.shadow.qualityPreset;
    result = readEnumField(shadow, "qualityPreset", "settings.shadow", preset,
                           {{"Custom", ShadowQualityPreset::Custom},
                            {"Low", ShadowQualityPreset::Low},
                            {"Medium", ShadowQualityPreset::Medium},
                            {"High", ShadowQualityPreset::High},
                            {"Ultra", ShadowQualityPreset::Ultra}});
    if (result.hasError()) {
      return result;
    }
    if (optionalObject(shadow, "qualityPreset") != nullptr) {
      applyShadowQualityPreset(settings.shadow, preset);
    }
    result = readEnumField(
        shadow, "depthFormat", "settings.shadow", settings.shadow.depthFormat,
        {{"D16_UNORM", Format::D16_UNORM}, {"D32_FLOAT", Format::D32_FLOAT}});
    if (result.hasError()) {
      return result;
    }
    auto number = readDouble(shadow, "maxDistance", "settings.shadow",
                             settings.shadow.maxDistance);
    if (number.hasError()) {
      return Result<bool, std::string>::makeError(number.error());
    }
    settings.shadow.maxDistance = static_cast<float>(number.value());
    number = readDouble(shadow, "maxDistanceFadeFraction", "settings.shadow",
                        settings.shadow.maxDistanceFadeFraction);
    if (number.hasError()) {
      return Result<bool, std::string>::makeError(number.error());
    }
    settings.shadow.maxDistanceFadeFraction =
        static_cast<float>(number.value());
    number = readDouble(shadow, "splitLambda", "settings.shadow",
                        settings.shadow.splitLambda);
    if (number.hasError()) {
      return Result<bool, std::string>::makeError(number.error());
    }
    settings.shadow.splitLambda = static_cast<float>(number.value());
    number = readDouble(shadow, "cascadeBlendFraction", "settings.shadow",
                        settings.shadow.cascadeBlendFraction);
    if (number.hasError()) {
      return Result<bool, std::string>::makeError(number.error());
    }
    settings.shadow.cascadeBlendFraction = static_cast<float>(number.value());
    auto count = readU32(shadow, "pcfSampleCount", "settings.shadow",
                         settings.shadow.pcfSampleCount);
    if (count.hasError()) {
      return Result<bool, std::string>::makeError(count.error());
    }
    settings.shadow.pcfSampleCount = count.value();
    number = readDouble(shadow, "sdsmTemporalBlend", "settings.shadow",
                        settings.shadow.sdsmTemporalBlend);
    if (number.hasError()) {
      return Result<bool, std::string>::makeError(number.error());
    }
    settings.shadow.sdsmTemporalBlend = static_cast<float>(number.value());
    enabled = readBool(shadow, "enableCascadeCasterCulling", "settings.shadow",
                       settings.shadow.debug.enableCascadeCasterCulling);
    if (enabled.hasError()) {
      return Result<bool, std::string>::makeError(enabled.error());
    }
    settings.shadow.debug.enableCascadeCasterCulling = enabled.value();
    if (yyjson_val *debug = optionalObject(shadow, "debug")) {
      static constexpr std::array debugKeys{
          std::string_view("showShadowMapViewport"),
          std::string_view("previewMode"),
          std::string_view("debugCascadeIndex"),
          std::string_view("previewDepthMin"),
          std::string_view("previewDepthMax"),
          std::string_view("previewDepthInvert"),
          std::string_view("previewDepthLog"),
          std::string_view("visualizeShadowFactor")};
      result = rejectUnknownKeys(debug, debugKeys, "settings.shadow.debug");
      if (result.hasError()) {
        return result;
      }
      enabled =
          readBool(debug, "showShadowMapViewport", "settings.shadow.debug",
                   settings.shadow.debug.showShadowMapViewport);
      if (enabled.hasError()) {
        return Result<bool, std::string>::makeError(enabled.error());
      }
      settings.shadow.debug.showShadowMapViewport = enabled.value();
      result = readEnumField(
          debug, "previewMode", "settings.shadow.debug",
          settings.shadow.debug.previewMode,
          {{"SelectedCascade", ShadowPreviewMode::SelectedCascade},
           {"TiledAllCascades", ShadowPreviewMode::TiledAllCascades}});
      if (result.hasError()) {
        return result;
      }
      auto u32 = readU32(debug, "debugCascadeIndex", "settings.shadow.debug",
                         settings.shadow.debug.debugCascadeIndex);
      if (u32.hasError()) {
        return Result<bool, std::string>::makeError(u32.error());
      }
      settings.shadow.debug.debugCascadeIndex = u32.value();
      auto real = readDouble(debug, "previewDepthMin", "settings.shadow.debug",
                             settings.shadow.debug.previewDepthMin);
      if (real.hasError()) {
        return Result<bool, std::string>::makeError(real.error());
      }
      settings.shadow.debug.previewDepthMin = static_cast<float>(real.value());
      real = readDouble(debug, "previewDepthMax", "settings.shadow.debug",
                        settings.shadow.debug.previewDepthMax);
      if (real.hasError()) {
        return Result<bool, std::string>::makeError(real.error());
      }
      settings.shadow.debug.previewDepthMax = static_cast<float>(real.value());
      enabled = readBool(debug, "previewDepthInvert", "settings.shadow.debug",
                         settings.shadow.debug.previewDepthInvert);
      if (enabled.hasError()) {
        return Result<bool, std::string>::makeError(enabled.error());
      }
      settings.shadow.debug.previewDepthInvert = enabled.value();
      enabled = readBool(debug, "previewDepthLog", "settings.shadow.debug",
                         settings.shadow.debug.previewDepthLog);
      if (enabled.hasError()) {
        return Result<bool, std::string>::makeError(enabled.error());
      }
      settings.shadow.debug.previewDepthLog = enabled.value();
      enabled =
          readBool(debug, "visualizeShadowFactor", "settings.shadow.debug",
                   settings.shadow.debug.visualizeShadowFactor);
      if (enabled.hasError()) {
        return Result<bool, std::string>::makeError(enabled.error());
      }
      settings.shadow.debug.visualizeShadowFactor = enabled.value();
    }
  }

  if (yyjson_val *hdr = optionalObject(object, "hdrPostProcess")) {
    static constexpr std::array hdrKeys{std::string_view("bloomEnabled"),
                                        std::string_view("adaptationEnabled")};
    result = rejectUnknownKeys(hdr, hdrKeys, "settings.hdrPostProcess");
    if (result.hasError()) {
      return result;
    }
    auto boolean = readBool(hdr, "bloomEnabled", "settings.hdrPostProcess",
                            settings.hdrPostProcess.bloomEnabled);
    if (boolean.hasError()) {
      return Result<bool, std::string>::makeError(boolean.error());
    }
    settings.hdrPostProcess.bloomEnabled = boolean.value();
    boolean = readBool(hdr, "adaptationEnabled", "settings.hdrPostProcess",
                       settings.hdrPostProcess.adaptationEnabled);
    if (boolean.hasError()) {
      return Result<bool, std::string>::makeError(boolean.error());
    }
    settings.hdrPostProcess.adaptationEnabled = boolean.value();
  }

  if (yyjson_val *transmission = optionalObject(object, "transmission")) {
    static constexpr std::array keys{std::string_view("enabled")};
    result = rejectUnknownKeys(transmission, keys, "settings.transmission");
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
  if (yyjson_val *transparent = optionalObject(object, "transparent")) {
    static constexpr std::array keys{std::string_view("enabled")};
    result = rejectUnknownKeys(transparent, keys, "settings.transparent");
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
  if (yyjson_val *filtering = optionalObject(object, "textureFiltering")) {
    static constexpr std::array keys{std::string_view("mode"),
                                     std::string_view("anisotropy")};
    result = rejectUnknownKeys(filtering, keys, "settings.textureFiltering");
    if (result.hasError()) {
      return result;
    }
    result = readEnumField(filtering, "mode", "settings.textureFiltering",
                           settings.textureFiltering.mode,
                           {{"Bilinear", TextureFilterMode::Bilinear},
                            {"Trilinear", TextureFilterMode::Trilinear},
                            {"Anisotropic", TextureFilterMode::Anisotropic}});
    if (result.hasError()) {
      return result;
    }
    auto anisotropy =
        readU32(filtering, "anisotropy", "settings.textureFiltering",
                settings.textureFiltering.anisotropy);
    if (anisotropy.hasError()) {
      return Result<bool, std::string>::makeError(anisotropy.error());
    }
    settings.textureFiltering.anisotropy =
        static_cast<uint8_t>(std::min<uint32_t>(anisotropy.value(), 255u));
  }

  sanitizeSnapshotRenderSettings(settings);
  return Result<bool, std::string>::makeResult(true);
}

[[nodiscard]] Result<bool, std::string> parseScene(yyjson_val *object,
                                                   SnapshotSceneConfig &scene) {
  static constexpr std::array keys{
      std::string_view("kind"),       std::string_view("pathBase"),
      std::string_view("path"),       std::string_view("importOptions"),
      std::string_view("generator"),  std::string_view("seed"),
      std::string_view("contentHash")};
  auto result = rejectUnknownKeys(object, keys, "scene");
  if (result.hasError()) {
    return result;
  }
  auto text = readString(object, "kind", "scene", false, scene.kind);
  if (text.hasError()) {
    return Result<bool, std::string>::makeError(text.error());
  }
  scene.kind = std::move(text.value());
  text = readString(object, "pathBase", "scene", false, scene.pathBase);
  if (text.hasError()) {
    return Result<bool, std::string>::makeError(text.error());
  }
  scene.pathBase = std::move(text.value());
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
  if (yyjson_val *importOptions = optionalObject(object, "importOptions")) {
    static constexpr std::array importKeys{std::string_view("mesh")};
    result =
        rejectUnknownKeys(importOptions, importKeys, "scene.importOptions");
    if (result.hasError()) {
      return result;
    }
    if (yyjson_val *mesh = optionalObject(importOptions, "mesh")) {
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
    }
  }
  return Result<bool, std::string>::makeResult(true);
}

[[nodiscard]] Result<bool, std::string>
parseCamera(yyjson_val *object, SnapshotCameraConfig &camera) {
  static constexpr std::array keys{std::string_view("position"),
                                   std::string_view("direction"),
                                   std::string_view("target"),
                                   std::string_view("positionDeltaPerFrame"),
                                   std::string_view("verticalFovDegrees"),
                                   std::string_view("nearPlane"),
                                   std::string_view("farPlane")};
  auto result = rejectUnknownKeys(object, keys, "camera");
  if (result.hasError()) {
    return result;
  }
  auto vec = readVec3(object, "position", "camera", camera.position);
  if (vec.hasError()) {
    return Result<bool, std::string>::makeError(vec.error());
  }
  camera.position = vec.value();
  if (optionalObject(object, "target") != nullptr) {
    vec = readVec3(object, "target", "camera",
                   camera.position + camera.direction);
    if (vec.hasError()) {
      return Result<bool, std::string>::makeError(vec.error());
    }
    camera.direction = glm::normalize(vec.value() - camera.position);
  } else {
    vec = readVec3(object, "direction", "camera", camera.direction);
    if (vec.hasError()) {
      return Result<bool, std::string>::makeError(vec.error());
    }
    camera.direction = glm::normalize(vec.value());
  }
  vec = readVec3(object, "positionDeltaPerFrame", "camera",
                 camera.positionDeltaPerFrame);
  if (vec.hasError()) {
    return Result<bool, std::string>::makeError(vec.error());
  }
  camera.positionDeltaPerFrame = vec.value();
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
  return Result<bool, std::string>::makeResult(true);
}

[[nodiscard]] Result<std::vector<SnapshotCaptureTarget>, std::string>
parseCaptures(yyjson_val *root) {
  std::vector<SnapshotCaptureTarget> captures;
  yyjson_val *array = optionalObject(root, "captures");
  if (array == nullptr) {
    return Result<std::vector<SnapshotCaptureTarget>, std::string>::makeResult(
        std::move(captures));
  }
  if (!yyjson_is_arr(array)) {
    return Result<std::vector<SnapshotCaptureTarget>, std::string>::makeError(
        "captures must be an array");
  }
  captures.reserve(yyjson_arr_size(array));
  std::set<std::string> captureNames;
  yyjson_arr_iter iter;
  yyjson_arr_iter_init(array, &iter);
  yyjson_val *entry = nullptr;
  while ((entry = yyjson_arr_iter_next(&iter)) != nullptr) {
    SnapshotCaptureTarget target{};
    if (yyjson_is_str(entry)) {
      target.name = std::string(yyjson_get_str(entry), yyjson_get_len(entry));
    } else if (yyjson_is_obj(entry)) {
      static constexpr std::array keys{std::string_view("target"),
                                       std::string_view("profile"),
                                       std::string_view("required")};
      auto result = rejectUnknownKeys(entry, keys, "captures[]");
      if (result.hasError()) {
        return Result<std::vector<SnapshotCaptureTarget>,
                      std::string>::makeError(result.error());
      }
      auto name = readString(entry, "target", "captures[]", true);
      if (name.hasError()) {
        return Result<std::vector<SnapshotCaptureTarget>,
                      std::string>::makeError(name.error());
      }
      target.name = std::move(name.value());
      auto profile = readString(entry, "profile", "captures[]", false, {});
      if (profile.hasError()) {
        return Result<std::vector<SnapshotCaptureTarget>,
                      std::string>::makeError(profile.error());
      }
      target.profile = std::move(profile.value());
      auto required = readBool(entry, "required", "captures[]", true);
      if (required.hasError()) {
        return Result<std::vector<SnapshotCaptureTarget>,
                      std::string>::makeError(required.error());
      }
      target.required = required.value();
    } else {
      return Result<std::vector<SnapshotCaptureTarget>, std::string>::makeError(
          "captures entries must be strings or objects");
    }
    const SnapshotCaptureCatalogEntry *catalog =
        findSnapshotCaptureCatalogEntry(target.name);
    if (catalog == nullptr) {
      return Result<std::vector<SnapshotCaptureTarget>, std::string>::makeError(
          "unknown capture target '" + target.name + "'");
    }
    if (catalog->availability ==
        SnapshotCaptureAvailability::KnownNotCapturable) {
      return Result<std::vector<SnapshotCaptureTarget>, std::string>::makeError(
          "capture target '" + target.name +
          "' is known but not capturable yet: " +
          std::string(catalog->diagnosticWork));
    }
    if (!captureNames.insert(target.name).second) {
      return Result<std::vector<SnapshotCaptureTarget>, std::string>::makeError(
          "duplicate capture target '" + target.name + "'");
    }
    if (target.profile.empty()) {
      target.profile = std::string(catalog->defaultCompareProfile);
    }
    auto profileIdentifier =
        validateSnapshotIdentifier(target.profile, "captures[].profile");
    if (profileIdentifier.hasError()) {
      return Result<std::vector<SnapshotCaptureTarget>, std::string>::makeError(
          profileIdentifier.error());
    }
    if (!isBuiltinSnapshotCompareProfile(target.profile)) {
      return Result<std::vector<SnapshotCaptureTarget>, std::string>::makeError(
          "unknown compare profile '" + target.profile + "'");
    }
    if (!snapshotCompareProfileSupportsKind(target.profile, catalog->kind)) {
      return Result<std::vector<SnapshotCaptureTarget>, std::string>::makeError(
          "compare profile '" + target.profile +
          "' is not compatible with capture target '" + target.name + "'");
    }
    captures.push_back(std::move(target));
  }
  return Result<std::vector<SnapshotCaptureTarget>, std::string>::makeResult(
      std::move(captures));
}

} // namespace

Result<bool, std::string> validateSnapshotIdentifier(std::string_view value,
                                                     std::string_view field,
                                                     bool allowDotted) {
  if (value.empty()) {
    return Result<bool, std::string>::makeError(std::string(field) +
                                                " must not be empty");
  }
  size_t segmentStart = 0u;
  while (segmentStart < value.size()) {
    const size_t dot = value.find('.', segmentStart);
    const size_t segmentEnd =
        dot == std::string_view::npos ? value.size() : dot;
    const std::string_view segment =
        value.substr(segmentStart, segmentEnd - segmentStart);
    if (segment.empty() || segment.size() > 64u) {
      return Result<bool, std::string>::makeError(
          std::string(field) + " contains an empty or overlong segment");
    }
    const auto isLowerAlphaNumeric = [](char c) {
      return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9');
    };
    if (!isLowerAlphaNumeric(segment.front())) {
      return Result<bool, std::string>::makeError(
          std::string(field) + " must use lowercase ASCII identifier segments");
    }
    for (const char c : segment) {
      if (!isLowerAlphaNumeric(c) && c != '_' && c != '-') {
        return Result<bool, std::string>::makeError(
            std::string(field) +
            " must use lowercase ASCII identifier segments");
      }
    }
    if (isReservedWindowsSegment(segment)) {
      return Result<bool, std::string>::makeError(
          std::string(field) + " contains a reserved device name");
    }
    if (dot == std::string_view::npos) {
      break;
    }
    if (!allowDotted) {
      return Result<bool, std::string>::makeError(std::string(field) +
                                                  " must be one segment");
    }
    segmentStart = dot + 1u;
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<std::filesystem::path, std::string>
resolveSnapshotPathUnder(const std::filesystem::path &root,
                         const std::filesystem::path &relative) {
  if (root.empty() || relative.empty() || relative.is_absolute() ||
      relative.has_root_name() || relative.has_root_directory()) {
    return Result<std::filesystem::path, std::string>::makeError(
        "resolveSnapshotPathUnder: expected a non-empty relative path");
  }
  for (const std::filesystem::path &component : relative) {
    if (component == "." || component == ".." || component.empty()) {
      return Result<std::filesystem::path, std::string>::makeError(
          "resolveSnapshotPathUnder: traversal is not allowed");
    }
  }

  std::error_code ec;
  const std::filesystem::path canonicalRoot =
      std::filesystem::weakly_canonical(root, ec);
  if (ec) {
    return Result<std::filesystem::path, std::string>::makeError(
        "resolveSnapshotPathUnder: failed to resolve root: " + ec.message());
  }
  const std::filesystem::path candidate =
      std::filesystem::weakly_canonical(canonicalRoot / relative, ec);
  if (ec || !pathIsUnder(canonicalRoot, candidate)) {
    return Result<std::filesystem::path, std::string>::makeError(
        "resolveSnapshotPathUnder: path escapes approved root");
  }

  std::filesystem::path current = canonicalRoot;
  for (const std::filesystem::path &component : relative) {
    current /= component;
    if (isReparsePoint(current)) {
      return Result<std::filesystem::path, std::string>::makeError(
          "resolveSnapshotPathUnder: reparse/symlink components are not "
          "allowed");
    }
  }
  return Result<std::filesystem::path, std::string>::makeResult(candidate);
}

std::filesystem::path defaultSnapshotCaseRoot() {
  return snapshotRepoRoot() / "tools" / "cases" / "snapshots";
}

Result<std::filesystem::path, std::string>
resolveSnapshotPath(std::string_view base, const std::filesystem::path &path) {
  if (path.is_absolute()) {
    return Result<std::filesystem::path, std::string>::makeResult(path);
  }
  if (base == "repoRoot") {
    return Result<std::filesystem::path, std::string>::makeResult(
        snapshotRepoRoot() / path);
  }
  auto configResult = loadRuntimeConfigFromEnvOrDefault();
  if (configResult.hasError()) {
    return Result<std::filesystem::path, std::string>::makeError(
        "resolveSnapshotPath: " + configResult.error());
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
      "resolveSnapshotPath: unsupported base '" + std::string(base) + "'");
}

Result<SnapshotCase, std::string>
loadSnapshotCaseManifest(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return Result<SnapshotCase, std::string>::makeError(
        "loadSnapshotCaseManifest: failed to open " + path.string());
  }
  std::string json((std::istreambuf_iterator<char>(file)),
                   std::istreambuf_iterator<char>());
  yyjson_read_err error{};
  JsonDocPtr doc(yyjson_read_opts(json.data(), json.size(), 0, nullptr, &error),
                 &yyjson_doc_free);
  if (!doc) {
    return Result<SnapshotCase, std::string>::makeError(
        "loadSnapshotCaseManifest: JSON parse failed at byte " +
        std::to_string(error.pos) + ": " + error.msg);
  }
  yyjson_val *root = yyjson_doc_get_root(doc.get());
  static constexpr std::array rootKeys{
      std::string_view("schemaVersion"), std::string_view("id"),
      std::string_view("suite"),         std::string_view("description"),
      std::string_view("scene"),         std::string_view("backend"),
      std::string_view("resolution"),    std::string_view("fixedDeltaSeconds"),
      std::string_view("warmupFrames"),  std::string_view("captureFrame"),
      std::string_view("authoritative"), std::string_view("presentMode"),
      std::string_view("windowMode"),    std::string_view("renderGraph"),
      std::string_view("camera"),        std::string_view("settings"),
      std::string_view("requirements"),  std::string_view("captures"),
  };
  auto keysResult = rejectUnknownKeys(root, rootKeys, "$");
  if (keysResult.hasError()) {
    return Result<SnapshotCase, std::string>::makeError(keysResult.error());
  }

  SnapshotCase out{};
  out.manifestPath = path;
  auto schema = readU32(root, "schemaVersion", "$", 1u);
  if (schema.hasError()) {
    return Result<SnapshotCase, std::string>::makeError(schema.error());
  }
  out.schemaVersion = schema.value();
  if (out.schemaVersion != 1u) {
    return Result<SnapshotCase, std::string>::makeError(
        "schemaVersion must be 1");
  }
  auto text = readString(root, "id", "$", true);
  if (text.hasError()) {
    return Result<SnapshotCase, std::string>::makeError(text.error());
  }
  out.id = std::move(text.value());
  auto identifier = validateSnapshotIdentifier(out.id, "$.id");
  if (identifier.hasError()) {
    return Result<SnapshotCase, std::string>::makeError(identifier.error());
  }
  text = readString(root, "suite", "$", true);
  if (text.hasError()) {
    return Result<SnapshotCase, std::string>::makeError(text.error());
  }
  out.suite = std::move(text.value());
  identifier = validateSnapshotIdentifier(out.suite, "$.suite");
  if (identifier.hasError()) {
    return Result<SnapshotCase, std::string>::makeError(identifier.error());
  }
  text = readString(root, "description", "$", false, out.description);
  if (text.hasError()) {
    return Result<SnapshotCase, std::string>::makeError(text.error());
  }
  out.description = std::move(text.value());
  text = readString(root, "backend", "$", false, out.backend);
  if (text.hasError()) {
    return Result<SnapshotCase, std::string>::makeError(text.error());
  }
  out.backend = std::move(text.value());
  if (out.backend != "default" && out.backend != "nvrhi") {
    return Result<SnapshotCase, std::string>::makeError(
        "$.backend must be default or nvrhi");
  }
  text = readString(root, "presentMode", "$", false, out.presentMode);
  if (text.hasError()) {
    return Result<SnapshotCase, std::string>::makeError(text.error());
  }
  out.presentMode = std::move(text.value());
  text = readString(root, "windowMode", "$", false, out.windowMode);
  if (text.hasError()) {
    return Result<SnapshotCase, std::string>::makeError(text.error());
  }
  out.windowMode = std::move(text.value());

  yyjson_val *resolution = optionalObject(root, "resolution");
  if (resolution != nullptr) {
    if (!yyjson_is_arr(resolution) || yyjson_arr_size(resolution) != 2u) {
      return Result<SnapshotCase, std::string>::makeError(
          "resolution must be [width, height]");
    }
    for (uint32_t i = 0u; i < 2u; ++i) {
      yyjson_val *entry = yyjson_arr_get(resolution, i);
      if (!yyjson_is_uint(entry) || yyjson_get_uint(entry) > UINT32_MAX) {
        return Result<SnapshotCase, std::string>::makeError(
            "resolution entries must be uint32");
      }
      out.resolution[i] = static_cast<uint32_t>(yyjson_get_uint(entry));
    }
  }

  auto number =
      readDouble(root, "fixedDeltaSeconds", "$", out.fixedDeltaSeconds);
  if (number.hasError()) {
    return Result<SnapshotCase, std::string>::makeError(number.error());
  }
  out.fixedDeltaSeconds = number.value();
  auto u32 = readU32(root, "warmupFrames", "$", out.warmupFrames);
  if (u32.hasError()) {
    return Result<SnapshotCase, std::string>::makeError(u32.error());
  }
  out.warmupFrames = u32.value();
  u32 = readU32(root, "captureFrame", "$", out.captureFrame);
  if (u32.hasError()) {
    return Result<SnapshotCase, std::string>::makeError(u32.error());
  }
  out.captureFrame = std::max(out.warmupFrames, u32.value());
  auto boolean = readBool(root, "authoritative", "$", out.authoritative);
  if (boolean.hasError()) {
    return Result<SnapshotCase, std::string>::makeError(boolean.error());
  }
  out.authoritative = boolean.value();

  if (yyjson_val *scene = optionalObject(root, "scene")) {
    auto parsed = parseScene(scene, out.scene);
    if (parsed.hasError()) {
      return Result<SnapshotCase, std::string>::makeError(parsed.error());
    }
  }
  if (yyjson_val *camera = optionalObject(root, "camera")) {
    auto parsed = parseCamera(camera, out.camera);
    if (parsed.hasError()) {
      return Result<SnapshotCase, std::string>::makeError(parsed.error());
    }
  }
  if (yyjson_val *settings = optionalObject(root, "settings")) {
    auto parsed = parseSettings(settings, out.settings);
    if (parsed.hasError()) {
      return Result<SnapshotCase, std::string>::makeError(parsed.error());
    }
  }
  if (yyjson_val *renderGraph = optionalObject(root, "renderGraph")) {
    static constexpr std::array keys{std::string_view("workerCount"),
                                     std::string_view("parallelCompile"),
                                     std::string_view("parallelRecording")};
    auto parsed = rejectUnknownKeys(renderGraph, keys, "renderGraph");
    if (parsed.hasError()) {
      return Result<SnapshotCase, std::string>::makeError(parsed.error());
    }
    u32 = readU32(renderGraph, "workerCount", "renderGraph",
                  out.renderGraph.workerCount);
    if (u32.hasError()) {
      return Result<SnapshotCase, std::string>::makeError(u32.error());
    }
    out.renderGraph.workerCount = std::max(1u, u32.value());
    boolean = readBool(renderGraph, "parallelCompile", "renderGraph",
                       out.renderGraph.parallelCompile);
    if (boolean.hasError()) {
      return Result<SnapshotCase, std::string>::makeError(boolean.error());
    }
    out.renderGraph.parallelCompile = boolean.value();
    boolean = readBool(renderGraph, "parallelRecording", "renderGraph",
                       out.renderGraph.parallelRecording);
    if (boolean.hasError()) {
      return Result<SnapshotCase, std::string>::makeError(boolean.error());
    }
    out.renderGraph.parallelRecording = boolean.value();
  }
  if (yyjson_val *requirements = optionalObject(root, "requirements")) {
    static constexpr std::array keys{std::string_view("assets"),
                                     std::string_view("backends"),
                                     std::string_view("allowVisibleWindow"),
                                     std::string_view("accelerationStructure"),
                                     std::string_view("rayQuery")};
    auto parsed = rejectUnknownKeys(requirements, keys, "requirements");
    if (parsed.hasError()) {
      return Result<SnapshotCase, std::string>::makeError(parsed.error());
    }
    auto array = readStringArray(requirements, "assets", "requirements");
    if (array.hasError()) {
      return Result<SnapshotCase, std::string>::makeError(array.error());
    }
    out.requirements.assets = std::move(array.value());
    array = readStringArray(requirements, "backends", "requirements");
    if (array.hasError()) {
      return Result<SnapshotCase, std::string>::makeError(array.error());
    }
    out.requirements.backends = std::move(array.value());
    for (const std::string &backend : out.requirements.backends) {
      if (backend != "default" && backend != "nvrhi") {
        return Result<SnapshotCase, std::string>::makeError(
            "requirements.backends contains unsupported backend '" + backend +
            "'");
      }
    }
    boolean = readBool(requirements, "allowVisibleWindow", "requirements",
                       out.requirements.allowVisibleWindow);
    if (boolean.hasError()) {
      return Result<SnapshotCase, std::string>::makeError(boolean.error());
    }
    out.requirements.allowVisibleWindow = boolean.value();
    boolean = readBool(requirements, "accelerationStructure", "requirements",
                       out.requirements.accelerationStructure);
    if (boolean.hasError()) {
      return Result<SnapshotCase, std::string>::makeError(boolean.error());
    }
    out.requirements.accelerationStructure = boolean.value();
    boolean = readBool(requirements, "rayQuery", "requirements",
                       out.requirements.rayQuery);
    if (boolean.hasError()) {
      return Result<SnapshotCase, std::string>::makeError(boolean.error());
    }
    out.requirements.rayQuery = boolean.value();
  }
  auto captures = parseCaptures(root);
  if (captures.hasError()) {
    return Result<SnapshotCase, std::string>::makeError(captures.error());
  }
  out.captures = std::move(captures.value());
  if (out.captures.empty()) {
    out.captures.push_back(SnapshotCaptureTarget{
        .name = "final_color", .profile = "ldr_color", .required = true});
  }
  return Result<SnapshotCase, std::string>::makeResult(std::move(out));
}

Result<std::vector<SnapshotCase>, std::string>
discoverSnapshotCases(const SnapshotManifestLoadOptions &options) {
  const std::filesystem::path root =
      options.caseRoot.empty() ? defaultSnapshotCaseRoot() : options.caseRoot;
  std::vector<SnapshotCase> cases;
  auto manifests = nuri::tools::core::discoverCaseManifestPaths(root);
  if (manifests.hasError()) {
    return Result<std::vector<SnapshotCase>, std::string>::makeError(
        manifests.error());
  }
  cases.reserve(manifests.value().size());
  for (const std::filesystem::path &manifest : manifests.value()) {
    auto loaded = loadSnapshotCaseManifest(manifest);
    if (loaded.hasError()) {
      return Result<std::vector<SnapshotCase>, std::string>::makeError(
          loaded.error());
    }
    cases.push_back(std::move(loaded.value()));
  }
  std::sort(cases.begin(), cases.end(),
            [](const SnapshotCase &lhs, const SnapshotCase &rhs) {
              return lhs.id < rhs.id;
            });
  std::vector<nuri::tools::core::CaseCatalogEntry> entries;
  entries.reserve(cases.size());
  for (const SnapshotCase &snapshotCase : cases) {
    entries.push_back({.id = snapshotCase.id,
                       .suite = snapshotCase.suite,
                       .manifestPath = snapshotCase.manifestPath});
  }
  auto validCatalog =
      nuri::tools::core::validateCaseCatalog(entries, "snapshot");
  if (validCatalog.hasError()) {
    return Result<std::vector<SnapshotCase>, std::string>::makeError(
        validCatalog.error());
  }
  return Result<std::vector<SnapshotCase>, std::string>::makeResult(
      std::move(cases));
}

const SnapshotCase *findSnapshotCaseById(const std::vector<SnapshotCase> &cases,
                                         std::string_view id) {
  for (const SnapshotCase &snapshotCase : cases) {
    if (snapshotCase.id == id) {
      return &snapshotCase;
    }
  }
  return nullptr;
}

std::vector<const SnapshotCase *>
filterSnapshotCasesBySuite(const std::vector<SnapshotCase> &cases,
                           std::string_view suite) {
  std::vector<const SnapshotCase *> out;
  std::vector<nuri::tools::core::CaseCatalogEntry> entries;
  entries.reserve(cases.size());
  for (const SnapshotCase &snapshotCase : cases) {
    entries.push_back({.id = snapshotCase.id,
                       .suite = snapshotCase.suite,
                       .manifestPath = snapshotCase.manifestPath});
  }
  auto selected = nuri::tools::core::selectCaseCatalog(
      entries,
      nuri::tools::core::CaseCatalogSelector{.suite = std::string(suite)},
      nuri::tools::core::CaseCatalogZeroMatchPolicy::Allow, "snapshot");
  if (selected.hasError()) {
    return out;
  }
  out.reserve(selected.value().size());
  for (const size_t index : selected.value()) {
    out.push_back(&cases[index]);
  }
  return out;
}

} // namespace nuri::tools::snapshot
