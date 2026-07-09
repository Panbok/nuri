#include "nuri/tools/autotest/autotest_manifest.h"

#include "nuri/tools/autotest/autotest_environment.h"
#include "nuri/tools/snapshot/snapshot_capture_point.h"
#include "nuri/tools/snapshot/snapshot_manifest.h"

#include <algorithm>
#include <array>
#include <climits>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <string_view>
#include <unordered_set>

#include <yyjson.h>

namespace nuri::tools::autotest {
namespace {

using JsonDocPtr = std::unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)>;

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

[[nodiscard]] Result<glm::vec3, std::string>
readVec3(yyjson_val *object, std::string_view key, std::string_view path,
         glm::vec3 defaultValue, bool required = false) {
  yyjson_val *value = optionalObject(object, key);
  if (value == nullptr) {
    if (required) {
      return Result<glm::vec3, std::string>::makeError(jsonPath(path, key) +
                                                       " is required");
    }
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
      std::string_view("textureFiltering"),
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
        std::string_view("enableCpuFrustumCulling"),
        std::string_view("enableTessellation"),
        std::string_view("forcedMeshLod"),
        std::string_view("meshletMode"),
        std::string_view("enableMeshletFrustumCulling"),
        std::string_view("enableMeshletConeCulling")};
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
    boolean = readBool(opaque, "enableCpuFrustumCulling", "settings.opaque",
                       settings.opaque.enableCpuFrustumCulling);
    if (boolean.hasError()) {
      return Result<bool, std::string>::makeError(boolean.error());
    }
    settings.opaque.enableCpuFrustumCulling = boolean.value();
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
    result = readEnumField(opaque, "meshletMode", "settings.opaque",
                           settings.opaque.meshletMode,
                           {{"Disabled", MeshletRenderMode::Disabled},
                            {"Opportunistic", MeshletRenderMode::Opportunistic},
                            {"Required", MeshletRenderMode::Required}});
    if (result.hasError()) {
      return result;
    }
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
  }

  if (yyjson_val *visibility = optionalObject(object, "visibility")) {
    static constexpr std::array visibilityKeys{
        std::string_view("mainViewMode"),
        std::string_view("shadowMode"),
        std::string_view("occlusionMode"),
        std::string_view("enableMeshletFrustumCulling"),
        std::string_view("enableMeshletConeCulling"),
        std::string_view("enableShadowMeshletCulling"),
        std::string_view("enableGpuInstanceCulling"),
        std::string_view("enableGpuIndirectDraw"),
        std::string_view("enableIndirectMeshDispatch"),
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
    boolean = readBool(visibility, "enableShadowMeshletCulling",
                       "settings.visibility",
                       settings.visibility.enableShadowMeshletCulling);
    if (boolean.hasError()) {
      return Result<bool, std::string>::makeError(boolean.error());
    }
    settings.visibility.enableShadowMeshletCulling = boolean.value();
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
         {"MSAA4x", AntiAliasingMode::MSAA4x}});
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
        std::string_view("enabled"), std::string_view("qualityPreset"),
        std::string_view("enableMeshletDepth"),
        std::string_view("enableMeshletCascadeCulling"),
        std::string_view("debug")};
    result = rejectUnknownKeys(shadow, shadowKeys, "settings.shadow");
    if (result.hasError()) {
      return result;
    }
    auto boolean =
        readBool(shadow, "enabled", "settings.shadow", settings.shadow.enabled);
    if (boolean.hasError()) {
      return Result<bool, std::string>::makeError(boolean.error());
    }
    settings.shadow.enabled = boolean.value();
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
    boolean = readBool(shadow, "enableMeshletDepth", "settings.shadow",
                       settings.shadow.enableMeshletDepth);
    if (boolean.hasError()) {
      return Result<bool, std::string>::makeError(boolean.error());
    }
    settings.shadow.enableMeshletDepth = boolean.value();
    boolean = readBool(shadow, "enableMeshletCascadeCulling", "settings.shadow",
                       settings.shadow.enableMeshletCascadeCulling);
    if (boolean.hasError()) {
      return Result<bool, std::string>::makeError(boolean.error());
    }
    settings.shadow.enableMeshletCascadeCulling = boolean.value();
    if (yyjson_val *debug = optionalObject(shadow, "debug")) {
      static constexpr std::array debugKeys{
          std::string_view("showShadowMapViewport"),
          std::string_view("previewMode"),
          std::string_view("debugCascadeIndex"),
          std::string_view("previewDepthMin"),
          std::string_view("previewDepthMax"),
          std::string_view("previewDepthInvert"),
          std::string_view("previewDepthLog")};
      result = rejectUnknownKeys(debug, debugKeys, "settings.shadow.debug");
      if (result.hasError()) {
        return result;
      }
      boolean =
          readBool(debug, "showShadowMapViewport", "settings.shadow.debug",
                   settings.shadow.debug.showShadowMapViewport);
      if (boolean.hasError()) {
        return Result<bool, std::string>::makeError(boolean.error());
      }
      settings.shadow.debug.showShadowMapViewport = boolean.value();
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
      boolean = readBool(debug, "previewDepthInvert", "settings.shadow.debug",
                         settings.shadow.debug.previewDepthInvert);
      if (boolean.hasError()) {
        return Result<bool, std::string>::makeError(boolean.error());
      }
      settings.shadow.debug.previewDepthInvert = boolean.value();
      boolean = readBool(debug, "previewDepthLog", "settings.shadow.debug",
                         settings.shadow.debug.previewDepthLog);
      if (boolean.hasError()) {
        return Result<bool, std::string>::makeError(boolean.error());
      }
      settings.shadow.debug.previewDepthLog = boolean.value();
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
    static constexpr std::array txKeys{std::string_view("enabled")};
    result = rejectUnknownKeys(transmission, txKeys, "settings.transmission");
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
    static constexpr std::array transparentKeys{std::string_view("enabled")};
    result =
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
  if (yyjson_val *filtering = optionalObject(object, "textureFiltering")) {
    static constexpr std::array filteringKeys{std::string_view("mode"),
                                              std::string_view("anisotropy")};
    result = rejectUnknownKeys(filtering, filteringKeys,
                               "settings.textureFiltering");
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

  sanitizeAutotestRenderSettings(settings);
  return Result<bool, std::string>::makeResult(true);
}

[[nodiscard]] Result<bool, std::string> parseScene(yyjson_val *object,
                                                   AutotestSceneConfig &scene) {
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
parseEnvironmentTexture(yyjson_val *object,
                        AutotestEnvironmentTextureConfig &texture,
                        std::string_view path) {
  static constexpr std::array keys{
      std::string_view("pathBase"), std::string_view("path"),
      std::string_view("kind"), std::string_view("debugName"),
      std::string_view("required")};
  auto result = rejectUnknownKeys(object, keys, path);
  if (result.hasError()) {
    return result;
  }
  texture.enabled = true;
  auto text = readString(object, "pathBase", path, true);
  if (text.hasError()) {
    return Result<bool, std::string>::makeError(text.error());
  }
  texture.pathBase = std::move(text.value());
  text = readString(object, "path", path, true);
  if (text.hasError()) {
    return Result<bool, std::string>::makeError(text.error());
  }
  texture.path = text.value();
  text = readString(object, "kind", path, false, texture.kind);
  if (text.hasError()) {
    return Result<bool, std::string>::makeError(text.error());
  }
  texture.kind = std::move(text.value());
  text = readString(object, "debugName", path, false, texture.debugName);
  if (text.hasError()) {
    return Result<bool, std::string>::makeError(text.error());
  }
  texture.debugName = std::move(text.value());
  auto required = readBool(object, "required", path, texture.required);
  if (required.hasError()) {
    return Result<bool, std::string>::makeError(required.error());
  }
  texture.required = required.value();
  return Result<bool, std::string>::makeResult(true);
}

[[nodiscard]] Result<bool, std::string>
parseEnvironment(yyjson_val *object, AutotestEnvironmentConfig &environment) {
  static constexpr std::array keys{
      std::string_view("cubemap"), std::string_view("irradiance"),
      std::string_view("prefilteredGgx"),
      std::string_view("prefilteredCharlie"), std::string_view("brdfLut")};
  auto result = rejectUnknownKeys(object, keys, "environment");
  if (result.hasError()) {
    return result;
  }
  struct TextureField {
    std::string_view key{};
    AutotestEnvironmentTextureConfig *texture = nullptr;
  };
  const std::array fields{
      TextureField{"cubemap", &environment.cubemap},
      TextureField{"irradiance", &environment.irradiance},
      TextureField{"prefilteredGgx", &environment.prefilteredGgx},
      TextureField{"prefilteredCharlie", &environment.prefilteredCharlie},
      TextureField{"brdfLut", &environment.brdfLut},
  };
  for (const TextureField &field : fields) {
    if (yyjson_val *texture = optionalObject(object, field.key)) {
      result = parseEnvironmentTexture(texture, *field.texture,
                                       jsonPath("environment", field.key));
      if (result.hasError()) {
        return result;
      }
    }
  }
  return Result<bool, std::string>::makeResult(true);
}

[[nodiscard]] Result<bool, std::string>
parseCamera(yyjson_val *object, AutotestCameraConfig &camera,
            std::string_view path = "camera") {
  static constexpr std::array keys{
      std::string_view("position"),  std::string_view("direction"),
      std::string_view("target"),    std::string_view("verticalFovDegrees"),
      std::string_view("nearPlane"), std::string_view("farPlane")};
  auto result = rejectUnknownKeys(object, keys, path);
  if (result.hasError()) {
    return result;
  }
  auto vec = readVec3(object, "position", path, camera.position);
  if (vec.hasError()) {
    return Result<bool, std::string>::makeError(vec.error());
  }
  camera.position = vec.value();
  if (optionalObject(object, "target") != nullptr) {
    vec = readVec3(object, "target", path, camera.target);
    if (vec.hasError()) {
      return Result<bool, std::string>::makeError(vec.error());
    }
    camera.target = vec.value();
    camera.hasTarget = true;
    const glm::vec3 direction = camera.target - camera.position;
    if (glm::length(direction) > 1.0e-6f) {
      camera.direction = glm::normalize(direction);
    }
  } else {
    vec = readVec3(object, "direction", path, camera.direction);
    if (vec.hasError()) {
      return Result<bool, std::string>::makeError(vec.error());
    }
    camera.direction = glm::length(vec.value()) > 1.0e-6f
                           ? glm::normalize(vec.value())
                           : glm::vec3(0.0f, 0.0f, -1.0f);
    camera.target = camera.position + camera.direction;
  }
  auto number =
      readDouble(object, "verticalFovDegrees", path, camera.verticalFovDegrees);
  if (number.hasError()) {
    return Result<bool, std::string>::makeError(number.error());
  }
  camera.verticalFovDegrees = static_cast<float>(number.value());
  number = readDouble(object, "nearPlane", path, camera.nearPlane);
  if (number.hasError()) {
    return Result<bool, std::string>::makeError(number.error());
  }
  camera.nearPlane = static_cast<float>(number.value());
  number = readDouble(object, "farPlane", path, camera.farPlane);
  if (number.hasError()) {
    return Result<bool, std::string>::makeError(number.error());
  }
  camera.farPlane = static_cast<float>(number.value());
  return Result<bool, std::string>::makeResult(true);
}

[[nodiscard]] Result<AutotestCaptureTarget, std::string>
parseCapture(yyjson_val *entry, std::string_view path) {
  AutotestCaptureTarget target{};
  if (yyjson_is_str(entry)) {
    target.target = std::string(yyjson_get_str(entry), yyjson_get_len(entry));
  } else if (yyjson_is_obj(entry)) {
    static constexpr std::array keys{
        std::string_view("target"), std::string_view("profile"),
        std::string_view("required"), std::string_view("compare")};
    auto result = rejectUnknownKeys(entry, keys, path);
    if (result.hasError()) {
      return Result<AutotestCaptureTarget, std::string>::makeError(
          result.error());
    }
    auto name = readString(entry, "target", path, true);
    if (name.hasError()) {
      return Result<AutotestCaptureTarget, std::string>::makeError(
          name.error());
    }
    target.target = std::move(name.value());
    auto profile = readString(entry, "profile", path, false, {});
    if (profile.hasError()) {
      return Result<AutotestCaptureTarget, std::string>::makeError(
          profile.error());
    }
    target.profile = std::move(profile.value());
    auto required = readBool(entry, "required", path, target.required);
    if (required.hasError()) {
      return Result<AutotestCaptureTarget, std::string>::makeError(
          required.error());
    }
    target.required = required.value();
    auto compare = readBool(entry, "compare", path, target.compare);
    if (compare.hasError()) {
      return Result<AutotestCaptureTarget, std::string>::makeError(
          compare.error());
    }
    target.compare = compare.value();
  } else {
    return Result<AutotestCaptureTarget, std::string>::makeError(
        std::string(path) + " entries must be strings or objects");
  }
  const nuri::tools::snapshot::SnapshotCaptureCatalogEntry *catalog =
      nuri::tools::snapshot::findSnapshotCaptureCatalogEntry(target.target);
  if (catalog == nullptr) {
    return Result<AutotestCaptureTarget, std::string>::makeError(
        "unknown capture target '" + target.target + "'");
  }
  if (catalog->availability ==
      nuri::tools::snapshot::SnapshotCaptureAvailability::KnownNotCapturable) {
    return Result<AutotestCaptureTarget, std::string>::makeError(
        "capture target '" + target.target +
        "' is known but not capturable yet: " +
        std::string(catalog->diagnosticWork));
  }
  if (target.profile.empty()) {
    target.profile = std::string(catalog->defaultCompareProfile);
  }
  return Result<AutotestCaptureTarget, std::string>::makeResult(
      std::move(target));
}

[[nodiscard]] Result<std::vector<AutotestMetricAssertion>, std::string>
parseAssertions(yyjson_val *array, std::string_view path) {
  std::vector<AutotestMetricAssertion> assertions;
  if (array == nullptr) {
    return Result<std::vector<AutotestMetricAssertion>,
                  std::string>::makeResult(std::move(assertions));
  }
  if (!yyjson_is_arr(array)) {
    return Result<std::vector<AutotestMetricAssertion>, std::string>::makeError(
        std::string(path) + " must be an array");
  }
  assertions.reserve(yyjson_arr_size(array));
  yyjson_arr_iter iter;
  yyjson_arr_iter_init(array, &iter);
  yyjson_val *entry = nullptr;
  while ((entry = yyjson_arr_iter_next(&iter)) != nullptr) {
    if (!yyjson_is_obj(entry)) {
      return Result<std::vector<AutotestMetricAssertion>,
                    std::string>::makeError(std::string(path) +
                                            " entries must be objects");
    }
    static constexpr std::array keys{std::string_view("id"),
                                     std::string_view("metric"),
                                     std::string_view("severity"),
                                     std::string_view("optional"),
                                     std::string_view("equals"),
                                     std::string_view("min"),
                                     std::string_view("max"),
                                     std::string_view("lessThan"),
                                     std::string_view("lessOrEqual"),
                                     std::string_view("greaterThan"),
                                     std::string_view("greaterOrEqual")};
    auto result = rejectUnknownKeys(entry, keys, path);
    if (result.hasError()) {
      return Result<std::vector<AutotestMetricAssertion>,
                    std::string>::makeError(result.error());
    }
    AutotestMetricAssertion assertion{};
    auto text = readString(entry, "id", path, true);
    if (text.hasError()) {
      return Result<std::vector<AutotestMetricAssertion>,
                    std::string>::makeError(text.error());
    }
    assertion.id = std::move(text.value());
    text = readString(entry, "metric", path, true);
    if (text.hasError()) {
      return Result<std::vector<AutotestMetricAssertion>,
                    std::string>::makeError(text.error());
    }
    assertion.metric = std::move(text.value());
    text = readString(entry, "severity", path, false, assertion.severity);
    if (text.hasError()) {
      return Result<std::vector<AutotestMetricAssertion>,
                    std::string>::makeError(text.error());
    }
    assertion.severity = std::move(text.value());
    if (assertion.severity != "fail" && assertion.severity != "warn") {
      return Result<std::vector<AutotestMetricAssertion>, std::string>::
          makeError("assertion severity must be 'fail' or 'warn'");
    }
    auto boolean = readBool(entry, "optional", path, assertion.optional);
    if (boolean.hasError()) {
      return Result<std::vector<AutotestMetricAssertion>,
                    std::string>::makeError(boolean.error());
    }
    assertion.optional = boolean.value();
    auto readComparator = [&](std::string_view key, bool &has,
                              double &value) -> Result<bool, std::string> {
      if (optionalObject(entry, key) == nullptr) {
        return Result<bool, std::string>::makeResult(true);
      }
      auto number = readDouble(entry, key, path);
      if (number.hasError()) {
        return Result<bool, std::string>::makeError(number.error());
      }
      has = true;
      value = number.value();
      return Result<bool, std::string>::makeResult(true);
    };
    for (auto comparator : {
             readComparator("equals", assertion.hasEquals, assertion.equals),
             readComparator("min", assertion.hasMin, assertion.min),
             readComparator("max", assertion.hasMax, assertion.max),
             readComparator("lessThan", assertion.hasLessThan,
                            assertion.lessThan),
             readComparator("lessOrEqual", assertion.hasLessOrEqual,
                            assertion.lessOrEqual),
             readComparator("greaterThan", assertion.hasGreaterThan,
                            assertion.greaterThan),
             readComparator("greaterOrEqual", assertion.hasGreaterOrEqual,
                            assertion.greaterOrEqual),
         }) {
      if (comparator.hasError()) {
        return Result<std::vector<AutotestMetricAssertion>,
                      std::string>::makeError(comparator.error());
      }
    }
    if (!assertion.hasEquals && !assertion.hasMin && !assertion.hasMax &&
        !assertion.hasLessThan && !assertion.hasLessOrEqual &&
        !assertion.hasGreaterThan && !assertion.hasGreaterOrEqual) {
      return Result<std::vector<AutotestMetricAssertion>, std::string>::
          makeError("assertion '" + assertion.id +
                    "' must specify at least one comparator");
    }
    assertions.push_back(std::move(assertion));
  }
  return Result<std::vector<AutotestMetricAssertion>, std::string>::makeResult(
      std::move(assertions));
}

[[nodiscard]] Result<std::vector<AutotestMetricWindowAssertion>, std::string>
parseMetricWindowAssertions(yyjson_val *array, std::string_view path) {
  std::vector<AutotestMetricWindowAssertion> assertions;
  if (array == nullptr) {
    return Result<std::vector<AutotestMetricWindowAssertion>,
                  std::string>::makeResult(std::move(assertions));
  }
  if (!yyjson_is_arr(array)) {
    return Result<std::vector<AutotestMetricWindowAssertion>,
                  std::string>::makeError(std::string(path) +
                                          " must be an array");
  }
  assertions.reserve(yyjson_arr_size(array));
  yyjson_arr_iter iter;
  yyjson_arr_iter_init(array, &iter);
  yyjson_val *entry = nullptr;
  while ((entry = yyjson_arr_iter_next(&iter)) != nullptr) {
    if (!yyjson_is_obj(entry)) {
      return Result<std::vector<AutotestMetricWindowAssertion>,
                    std::string>::makeError(std::string(path) +
                                            " entries must be objects");
    }
    static constexpr std::array keys{
        std::string_view("id"),        std::string_view("metric"),
        std::string_view("severity"),  std::string_view("optional"),
        std::string_view("equals"),    std::string_view("min"),
        std::string_view("max"),       std::string_view("medianMin"),
        std::string_view("medianMax"), std::string_view("p95Min"),
        std::string_view("p95Max"),    std::string_view("varianceMax")};
    auto result = rejectUnknownKeys(entry, keys, path);
    if (result.hasError()) {
      return Result<std::vector<AutotestMetricWindowAssertion>,
                    std::string>::makeError(result.error());
    }
    AutotestMetricWindowAssertion assertion{};
    auto text = readString(entry, "id", path, true);
    if (text.hasError()) {
      return Result<std::vector<AutotestMetricWindowAssertion>,
                    std::string>::makeError(text.error());
    }
    assertion.id = std::move(text.value());
    text = readString(entry, "metric", path, true);
    if (text.hasError()) {
      return Result<std::vector<AutotestMetricWindowAssertion>,
                    std::string>::makeError(text.error());
    }
    assertion.metric = std::move(text.value());
    text = readString(entry, "severity", path, false, assertion.severity);
    if (text.hasError()) {
      return Result<std::vector<AutotestMetricWindowAssertion>,
                    std::string>::makeError(text.error());
    }
    assertion.severity = std::move(text.value());
    if (assertion.severity != "fail" && assertion.severity != "warn") {
      return Result<std::vector<AutotestMetricWindowAssertion>, std::string>::
          makeError(
              "metric window assertion severity must be 'fail' or 'warn'");
    }
    auto boolean = readBool(entry, "optional", path, assertion.optional);
    if (boolean.hasError()) {
      return Result<std::vector<AutotestMetricWindowAssertion>,
                    std::string>::makeError(boolean.error());
    }
    assertion.optional = boolean.value();
    auto readComparator = [&](std::string_view key, bool &has,
                              double &value) -> Result<bool, std::string> {
      if (optionalObject(entry, key) == nullptr) {
        return Result<bool, std::string>::makeResult(true);
      }
      auto number = readDouble(entry, key, path);
      if (number.hasError()) {
        return Result<bool, std::string>::makeError(number.error());
      }
      has = true;
      value = number.value();
      return Result<bool, std::string>::makeResult(true);
    };
    for (auto comparator : {
             readComparator("equals", assertion.hasEquals, assertion.equals),
             readComparator("min", assertion.hasMin, assertion.min),
             readComparator("max", assertion.hasMax, assertion.max),
             readComparator("medianMin", assertion.hasMedianMin,
                            assertion.medianMin),
             readComparator("medianMax", assertion.hasMedianMax,
                            assertion.medianMax),
             readComparator("p95Min", assertion.hasP95Min, assertion.p95Min),
             readComparator("p95Max", assertion.hasP95Max, assertion.p95Max),
             readComparator("varianceMax", assertion.hasVarianceMax,
                            assertion.varianceMax),
         }) {
      if (comparator.hasError()) {
        return Result<std::vector<AutotestMetricWindowAssertion>,
                      std::string>::makeError(comparator.error());
      }
    }
    if (!assertion.hasEquals && !assertion.hasMin && !assertion.hasMax &&
        !assertion.hasMedianMin && !assertion.hasMedianMax &&
        !assertion.hasP95Min && !assertion.hasP95Max &&
        !assertion.hasVarianceMax) {
      return Result<std::vector<AutotestMetricWindowAssertion>, std::string>::
          makeError("metric window assertion '" + assertion.id +
                    "' must specify at least one comparator");
    }
    assertions.push_back(std::move(assertion));
  }
  return Result<std::vector<AutotestMetricWindowAssertion>,
                std::string>::makeResult(std::move(assertions));
}

[[nodiscard]] Result<std::vector<AutotestReadoutRequest>, std::string>
parseReadouts(yyjson_val *array, std::string_view path) {
  std::vector<AutotestReadoutRequest> readouts;
  if (array == nullptr) {
    return Result<std::vector<AutotestReadoutRequest>, std::string>::makeResult(
        std::move(readouts));
  }
  if (!yyjson_is_arr(array)) {
    return Result<std::vector<AutotestReadoutRequest>, std::string>::makeError(
        std::string(path) + " must be an array");
  }
  std::unordered_set<std::string> ids;
  yyjson_arr_iter iter;
  yyjson_arr_iter_init(array, &iter);
  yyjson_val *entry = nullptr;
  while ((entry = yyjson_arr_iter_next(&iter)) != nullptr) {
    if (!yyjson_is_obj(entry)) {
      return Result<std::vector<AutotestReadoutRequest>,
                    std::string>::makeError(std::string(path) +
                                            " entries must be objects");
    }
    static constexpr std::array keys{
        std::string_view("id"),       std::string_view("type"),
        std::string_view("x"),        std::string_view("y"),
        std::string_view("required"), std::string_view("assertions")};
    auto result = rejectUnknownKeys(entry, keys, path);
    if (result.hasError()) {
      return Result<std::vector<AutotestReadoutRequest>,
                    std::string>::makeError(result.error());
    }
    AutotestReadoutRequest readout{};
    auto text = readString(entry, "id", path, true);
    if (text.hasError()) {
      return Result<std::vector<AutotestReadoutRequest>,
                    std::string>::makeError(text.error());
    }
    readout.id = std::move(text.value());
    if (!ids.insert(readout.id).second) {
      return Result<std::vector<AutotestReadoutRequest>,
                    std::string>::makeError("duplicate readout id '" +
                                            readout.id + "'");
    }
    text = readString(entry, "type", path, true);
    if (text.hasError()) {
      return Result<std::vector<AutotestReadoutRequest>,
                    std::string>::makeError(text.error());
    }
    readout.type = std::move(text.value());
    if (readout.type != "shadowInspect" && readout.type != "opaquePick") {
      return Result<std::vector<AutotestReadoutRequest>, std::string>::
          makeError("readout '" + readout.id +
                    "' type must be shadowInspect or opaquePick");
    }
    auto u32 = readU32(entry, "x", path, 0u);
    if (u32.hasError()) {
      return Result<std::vector<AutotestReadoutRequest>,
                    std::string>::makeError(u32.error());
    }
    readout.x = u32.value();
    u32 = readU32(entry, "y", path, 0u);
    if (u32.hasError()) {
      return Result<std::vector<AutotestReadoutRequest>,
                    std::string>::makeError(u32.error());
    }
    readout.y = u32.value();
    auto boolean = readBool(entry, "required", path, readout.required);
    if (boolean.hasError()) {
      return Result<std::vector<AutotestReadoutRequest>,
                    std::string>::makeError(boolean.error());
    }
    readout.required = boolean.value();
    auto assertions = parseAssertions(optionalObject(entry, "assertions"),
                                      std::string(path) + ".assertions");
    if (assertions.hasError()) {
      return Result<std::vector<AutotestReadoutRequest>,
                    std::string>::makeError(assertions.error());
    }
    readout.assertions = std::move(assertions.value());
    readouts.push_back(std::move(readout));
  }
  return Result<std::vector<AutotestReadoutRequest>, std::string>::makeResult(
      std::move(readouts));
}

[[nodiscard]] Result<std::vector<AutotestMetricWindow>, std::string>
parseMetricWindows(yyjson_val *root, uint32_t endFrame) {
  std::vector<AutotestMetricWindow> windows;
  yyjson_val *array = optionalObject(root, "metricWindows");
  if (array == nullptr) {
    return Result<std::vector<AutotestMetricWindow>, std::string>::makeResult(
        std::move(windows));
  }
  if (!yyjson_is_arr(array)) {
    return Result<std::vector<AutotestMetricWindow>, std::string>::makeError(
        "metricWindows must be an array");
  }
  std::unordered_set<std::string> ids;
  yyjson_arr_iter iter;
  yyjson_arr_iter_init(array, &iter);
  yyjson_val *entry = nullptr;
  while ((entry = yyjson_arr_iter_next(&iter)) != nullptr) {
    if (!yyjson_is_obj(entry)) {
      return Result<std::vector<AutotestMetricWindow>, std::string>::makeError(
          "metricWindows entries must be objects");
    }
    static constexpr std::array keys{std::string_view("id"),
                                     std::string_view("frames"),
                                     std::string_view("assertions")};
    auto result = rejectUnknownKeys(entry, keys, "metricWindows[]");
    if (result.hasError()) {
      return Result<std::vector<AutotestMetricWindow>, std::string>::makeError(
          result.error());
    }
    AutotestMetricWindow window{};
    auto text = readString(entry, "id", "metricWindows[]", true);
    if (text.hasError()) {
      return Result<std::vector<AutotestMetricWindow>, std::string>::makeError(
          text.error());
    }
    window.id = std::move(text.value());
    if (!ids.insert(window.id).second) {
      return Result<std::vector<AutotestMetricWindow>, std::string>::makeError(
          "duplicate metric window id '" + window.id + "'");
    }
    yyjson_val *frames = optionalObject(entry, "frames");
    if (!yyjson_is_arr(frames) || yyjson_arr_size(frames) != 2u) {
      return Result<std::vector<AutotestMetricWindow>, std::string>::makeError(
          "metricWindows[].frames must be [startFrame, endFrame]");
    }
    for (uint32_t i = 0u; i < 2u; ++i) {
      yyjson_val *frame = yyjson_arr_get(frames, i);
      if (!yyjson_is_uint(frame) || yyjson_get_uint(frame) > UINT32_MAX) {
        return Result<std::vector<AutotestMetricWindow>, std::string>::
            makeError("metricWindows[].frames entries must be uint32");
      }
      if (i == 0u) {
        window.startFrame = static_cast<uint32_t>(yyjson_get_uint(frame));
      } else {
        window.endFrame = static_cast<uint32_t>(yyjson_get_uint(frame));
      }
    }
    if (window.startFrame > window.endFrame || window.endFrame > endFrame) {
      return Result<std::vector<AutotestMetricWindow>, std::string>::makeError(
          "metric window '" + window.id + "' has invalid frame range");
    }
    auto assertions = parseMetricWindowAssertions(
        optionalObject(entry, "assertions"), "metricWindows[].assertions");
    if (assertions.hasError()) {
      return Result<std::vector<AutotestMetricWindow>, std::string>::makeError(
          assertions.error());
    }
    if (assertions.value().empty()) {
      return Result<std::vector<AutotestMetricWindow>, std::string>::makeError(
          "metric window '" + window.id + "' must contain assertions");
    }
    window.assertions = std::move(assertions.value());
    windows.push_back(std::move(window));
  }
  return Result<std::vector<AutotestMetricWindow>, std::string>::makeResult(
      std::move(windows));
}

[[nodiscard]] Result<std::vector<AutotestCheckpoint>, std::string>
parseCheckpoints(yyjson_val *root, uint32_t endFrame) {
  std::vector<AutotestCheckpoint> checkpoints;
  yyjson_val *array = optionalObject(root, "checkpoints");
  if (array == nullptr) {
    checkpoints.push_back(AutotestCheckpoint{
        .id = "end",
        .frame = endFrame,
        .captures = {AutotestCaptureTarget{.target = "final_color",
                                           .profile = "ldr_color",
                                           .required = true,
                                           .compare = false}},
    });
    return Result<std::vector<AutotestCheckpoint>, std::string>::makeResult(
        std::move(checkpoints));
  }
  if (!yyjson_is_arr(array)) {
    return Result<std::vector<AutotestCheckpoint>, std::string>::makeError(
        "checkpoints must be an array");
  }
  std::unordered_set<std::string> ids;
  yyjson_arr_iter iter;
  yyjson_arr_iter_init(array, &iter);
  yyjson_val *entry = nullptr;
  while ((entry = yyjson_arr_iter_next(&iter)) != nullptr) {
    if (!yyjson_is_obj(entry)) {
      return Result<std::vector<AutotestCheckpoint>, std::string>::makeError(
          "checkpoints entries must be objects");
    }
    static constexpr std::array keys{
        std::string_view("id"), std::string_view("frame"),
        std::string_view("captures"), std::string_view("readouts"),
        std::string_view("assertions")};
    auto result = rejectUnknownKeys(entry, keys, "checkpoints[]");
    if (result.hasError()) {
      return Result<std::vector<AutotestCheckpoint>, std::string>::makeError(
          result.error());
    }
    AutotestCheckpoint checkpoint{};
    auto text = readString(entry, "id", "checkpoints[]", true);
    if (text.hasError()) {
      return Result<std::vector<AutotestCheckpoint>, std::string>::makeError(
          text.error());
    }
    checkpoint.id = std::move(text.value());
    if (!ids.insert(checkpoint.id).second) {
      return Result<std::vector<AutotestCheckpoint>, std::string>::makeError(
          "duplicate checkpoint id '" + checkpoint.id + "'");
    }
    auto frame = readU32(entry, "frame", "checkpoints[]", endFrame);
    if (frame.hasError()) {
      return Result<std::vector<AutotestCheckpoint>, std::string>::makeError(
          frame.error());
    }
    checkpoint.frame = frame.value();
    if (checkpoint.frame > endFrame) {
      return Result<std::vector<AutotestCheckpoint>, std::string>::makeError(
          "checkpoint '" + checkpoint.id + "' frame is outside endFrame");
    }
    yyjson_val *captures = optionalObject(entry, "captures");
    if (captures != nullptr) {
      if (!yyjson_is_arr(captures)) {
        return Result<std::vector<AutotestCheckpoint>, std::string>::makeError(
            "checkpoints[].captures must be an array");
      }
      yyjson_arr_iter captureIter;
      yyjson_arr_iter_init(captures, &captureIter);
      yyjson_val *capture = nullptr;
      while ((capture = yyjson_arr_iter_next(&captureIter)) != nullptr) {
        auto parsed = parseCapture(capture, "checkpoints[].captures[]");
        if (parsed.hasError()) {
          return Result<std::vector<AutotestCheckpoint>,
                        std::string>::makeError(parsed.error());
        }
        checkpoint.captures.push_back(std::move(parsed.value()));
      }
    }
    auto readouts = parseReadouts(optionalObject(entry, "readouts"),
                                  "checkpoints[].readouts");
    if (readouts.hasError()) {
      return Result<std::vector<AutotestCheckpoint>, std::string>::makeError(
          readouts.error());
    }
    checkpoint.readouts = std::move(readouts.value());
    auto assertions = parseAssertions(optionalObject(entry, "assertions"),
                                      "checkpoints[].assertions");
    if (assertions.hasError()) {
      return Result<std::vector<AutotestCheckpoint>, std::string>::makeError(
          assertions.error());
    }
    checkpoint.assertions = std::move(assertions.value());
    checkpoints.push_back(std::move(checkpoint));
  }
  if (checkpoints.empty()) {
    return Result<std::vector<AutotestCheckpoint>, std::string>::makeError(
        "checkpoints must contain at least one checkpoint");
  }
  std::sort(checkpoints.begin(), checkpoints.end(),
            [](const AutotestCheckpoint &lhs, const AutotestCheckpoint &rhs) {
              return lhs.frame < rhs.frame;
            });
  return Result<std::vector<AutotestCheckpoint>, std::string>::makeResult(
      std::move(checkpoints));
}

[[nodiscard]] Result<AutotestTimeline, std::string>
parseTimeline(yyjson_val *root, const AutotestCameraConfig &baseCamera,
              const RenderSettings &baseSettings, uint32_t endFrame) {
  AutotestTimeline timeline{};
  yyjson_val *object = optionalObject(root, "timeline");
  if (object == nullptr) {
    return Result<AutotestTimeline, std::string>::makeResult(timeline);
  }
  static constexpr std::array keys{std::string_view("cameraPaths"),
                                   std::string_view("events")};
  auto result = rejectUnknownKeys(object, keys, "timeline");
  if (result.hasError()) {
    return Result<AutotestTimeline, std::string>::makeError(result.error());
  }
  if (yyjson_val *paths = optionalObject(object, "cameraPaths")) {
    if (!yyjson_is_arr(paths)) {
      return Result<AutotestTimeline, std::string>::makeError(
          "timeline.cameraPaths must be an array");
    }
    yyjson_arr_iter pathIter;
    yyjson_arr_iter_init(paths, &pathIter);
    yyjson_val *pathValue = nullptr;
    while ((pathValue = yyjson_arr_iter_next(&pathIter)) != nullptr) {
      if (!yyjson_is_obj(pathValue)) {
        return Result<AutotestTimeline, std::string>::makeError(
            "timeline.cameraPaths entries must be objects");
      }
      static constexpr std::array pathKeys{
          std::string_view("id"), std::string_view("startFrame"),
          std::string_view("endFrame"), std::string_view("interpolation"),
          std::string_view("keyframes")};
      result = rejectUnknownKeys(pathValue, pathKeys, "timeline.cameraPaths[]");
      if (result.hasError()) {
        return Result<AutotestTimeline, std::string>::makeError(result.error());
      }
      AutotestCameraPath path{};
      auto text = readString(pathValue, "id", "timeline.cameraPaths[]", true);
      if (text.hasError()) {
        return Result<AutotestTimeline, std::string>::makeError(text.error());
      }
      path.id = std::move(text.value());
      auto u32 = readU32(pathValue, "startFrame", "timeline.cameraPaths[]", 0u);
      if (u32.hasError()) {
        return Result<AutotestTimeline, std::string>::makeError(u32.error());
      }
      path.startFrame = u32.value();
      u32 = readU32(pathValue, "endFrame", "timeline.cameraPaths[]", endFrame);
      if (u32.hasError()) {
        return Result<AutotestTimeline, std::string>::makeError(u32.error());
      }
      path.endFrame = u32.value();
      if (path.startFrame > path.endFrame || path.endFrame > endFrame) {
        return Result<AutotestTimeline, std::string>::makeError(
            "timeline.cameraPaths[] has invalid frame range");
      }
      text = readString(pathValue, "interpolation", "timeline.cameraPaths[]",
                        false, path.interpolation);
      if (text.hasError()) {
        return Result<AutotestTimeline, std::string>::makeError(text.error());
      }
      path.interpolation = std::move(text.value());
      if (path.interpolation != "linear" &&
          path.interpolation != "smoothstep") {
        return Result<AutotestTimeline, std::string>::makeError(
            "timeline.cameraPaths[] interpolation must be linear or "
            "smoothstep");
      }
      yyjson_val *keyframes = optionalObject(pathValue, "keyframes");
      if (!yyjson_is_arr(keyframes) || yyjson_arr_size(keyframes) == 0u) {
        return Result<AutotestTimeline, std::string>::makeError(
            "timeline.cameraPaths[].keyframes must be a non-empty array");
      }
      yyjson_arr_iter keyframeIter;
      yyjson_arr_iter_init(keyframes, &keyframeIter);
      yyjson_val *keyframeValue = nullptr;
      while ((keyframeValue = yyjson_arr_iter_next(&keyframeIter)) != nullptr) {
        if (!yyjson_is_obj(keyframeValue)) {
          return Result<AutotestTimeline, std::string>::makeError(
              "timeline.cameraPaths[].keyframes entries must be objects");
        }
        static constexpr std::array keyframeKeys{std::string_view("frame"),
                                                 std::string_view("position"),
                                                 std::string_view("target")};
        result = rejectUnknownKeys(keyframeValue, keyframeKeys,
                                   "timeline.cameraPaths[].keyframes[]");
        if (result.hasError()) {
          return Result<AutotestTimeline, std::string>::makeError(
              result.error());
        }
        AutotestCameraKeyframe keyframe{};
        u32 = readU32(keyframeValue, "frame",
                      "timeline.cameraPaths[].keyframes[]", path.startFrame);
        if (u32.hasError()) {
          return Result<AutotestTimeline, std::string>::makeError(u32.error());
        }
        keyframe.frame = u32.value();
        if (keyframe.frame < path.startFrame ||
            keyframe.frame > path.endFrame) {
          return Result<AutotestTimeline, std::string>::makeError(
              "camera keyframe is outside its path range");
        }
        auto vec = readVec3(keyframeValue, "position",
                            "timeline.cameraPaths[].keyframes[]",
                            baseCamera.position, true);
        if (vec.hasError()) {
          return Result<AutotestTimeline, std::string>::makeError(vec.error());
        }
        keyframe.position = vec.value();
        if (optionalObject(keyframeValue, "target") != nullptr) {
          vec = readVec3(keyframeValue, "target",
                         "timeline.cameraPaths[].keyframes[]",
                         baseCamera.target, true);
          if (vec.hasError()) {
            return Result<AutotestTimeline, std::string>::makeError(
                vec.error());
          }
          keyframe.target = vec.value();
          keyframe.hasTarget = true;
        }
        path.keyframes.push_back(keyframe);
      }
      std::sort(path.keyframes.begin(), path.keyframes.end(),
                [](const AutotestCameraKeyframe &lhs,
                   const AutotestCameraKeyframe &rhs) {
                  return lhs.frame < rhs.frame;
                });
      timeline.cameraPaths.push_back(std::move(path));
    }
  }
  if (yyjson_val *events = optionalObject(object, "events")) {
    if (!yyjson_is_arr(events)) {
      return Result<AutotestTimeline, std::string>::makeError(
          "timeline.events must be an array");
    }
    RenderSettings currentSettings = baseSettings;
    uint32_t lastEventFrame = 0u;
    bool hasLastEventFrame = false;
    yyjson_arr_iter eventIter;
    yyjson_arr_iter_init(events, &eventIter);
    yyjson_val *eventValue = nullptr;
    while ((eventValue = yyjson_arr_iter_next(&eventIter)) != nullptr) {
      if (!yyjson_is_obj(eventValue)) {
        return Result<AutotestTimeline, std::string>::makeError(
            "timeline.events entries must be objects");
      }
      static constexpr std::array eventKeys{
          std::string_view("frame"),       std::string_view("type"),
          std::string_view("eventReason"), std::string_view("camera"),
          std::string_view("settings"),    std::string_view("preserveHistory")};
      result = rejectUnknownKeys(eventValue, eventKeys, "timeline.events[]");
      if (result.hasError()) {
        return Result<AutotestTimeline, std::string>::makeError(result.error());
      }
      AutotestTimelineEvent event{};
      auto u32 = readU32(eventValue, "frame", "timeline.events[]", 0u);
      if (u32.hasError()) {
        return Result<AutotestTimeline, std::string>::makeError(u32.error());
      }
      event.frame = u32.value();
      if (event.frame > endFrame) {
        return Result<AutotestTimeline, std::string>::makeError(
            "timeline event frame is outside endFrame");
      }
      if (hasLastEventFrame && event.frame < lastEventFrame) {
        return Result<AutotestTimeline, std::string>::makeError(
            "timeline events must be sorted by frame");
      }
      lastEventFrame = event.frame;
      hasLastEventFrame = true;
      auto text = readString(eventValue, "type", "timeline.events[]", true);
      if (text.hasError()) {
        return Result<AutotestTimeline, std::string>::makeError(text.error());
      }
      event.type = std::move(text.value());
      text = readString(eventValue, "eventReason", "timeline.events[]", false,
                        event.eventReason);
      if (text.hasError()) {
        return Result<AutotestTimeline, std::string>::makeError(text.error());
      }
      event.eventReason = std::move(text.value());
      auto boolean = readBool(eventValue, "preserveHistory",
                              "timeline.events[]", event.preserveHistory);
      if (boolean.hasError()) {
        return Result<AutotestTimeline, std::string>::makeError(
            boolean.error());
      }
      event.preserveHistory = boolean.value();
      event.camera = baseCamera;
      if (yyjson_val *camera = optionalObject(eventValue, "camera")) {
        auto parsed =
            parseCamera(camera, event.camera, "timeline.events[].camera");
        if (parsed.hasError()) {
          return Result<AutotestTimeline, std::string>::makeError(
              parsed.error());
        }
      }
      event.settings = currentSettings;
      if (yyjson_val *settings = optionalObject(eventValue, "settings")) {
        auto parsed = parseSettings(settings, event.settings);
        if (parsed.hasError()) {
          return Result<AutotestTimeline, std::string>::makeError(
              parsed.error());
        }
        event.hasSettings = true;
      }
      if (event.type == "setSettings") {
        if (!event.hasSettings) {
          return Result<AutotestTimeline, std::string>::makeError(
              "setSettings event requires settings");
        }
        currentSettings = event.settings;
      }
      if (event.type != "resetTemporalHistory" && event.type != "setCamera" &&
          event.type != "setSettings") {
        return Result<AutotestTimeline, std::string>::makeError(
            "unsupported timeline event type '" + event.type + "'");
      }
      timeline.events.push_back(std::move(event));
    }
  }
  return Result<AutotestTimeline, std::string>::makeResult(std::move(timeline));
}

} // namespace

std::filesystem::path defaultAutotestCaseRoot() {
  return autotestRepoRoot() / "tools" / "cases" / "autotests";
}

Result<std::filesystem::path, std::string>
resolveAutotestPath(std::string_view base, const std::filesystem::path &path) {
  return nuri::tools::snapshot::resolveSnapshotPath(base, path);
}

Result<AutotestCase, std::string>
loadAutotestCaseManifest(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return Result<AutotestCase, std::string>::makeError(
        "loadAutotestCaseManifest: failed to open " + path.string());
  }
  std::string json((std::istreambuf_iterator<char>(file)),
                   std::istreambuf_iterator<char>());
  yyjson_read_err error{};
  JsonDocPtr doc(yyjson_read_opts(json.data(), json.size(), 0, nullptr, &error),
                 &yyjson_doc_free);
  if (!doc) {
    return Result<AutotestCase, std::string>::makeError(
        "loadAutotestCaseManifest: JSON parse failed at byte " +
        std::to_string(error.pos) + ": " + error.msg);
  }
  yyjson_val *root = yyjson_doc_get_root(doc.get());
  static constexpr std::array rootKeys{
      std::string_view("schemaVersion"),
      std::string_view("id"),
      std::string_view("suite"),
      std::string_view("description"),
      std::string_view("scene"),
      std::string_view("backend"),
      std::string_view("environment"),
      std::string_view("resolution"),
      std::string_view("fixedDeltaSeconds"),
      std::string_view("warmupFrames"),
      std::string_view("endFrame"),
      std::string_view("authoritative"),
      std::string_view("presentMode"),
      std::string_view("windowMode"),
      std::string_view("renderGraph"),
      std::string_view("camera"),
      std::string_view("settings"),
      std::string_view("requirements"),
      std::string_view("timeline"),
      std::string_view("checkpoints"),
      std::string_view("metricWindows"),
  };
  auto keysResult = rejectUnknownKeys(root, rootKeys, "$");
  if (keysResult.hasError()) {
    return Result<AutotestCase, std::string>::makeError(keysResult.error());
  }

  AutotestCase out{};
  out.manifestPath = path;
  auto schema = readU32(root, "schemaVersion", "$", 1u);
  if (schema.hasError()) {
    return Result<AutotestCase, std::string>::makeError(schema.error());
  }
  out.schemaVersion = schema.value();
  if (out.schemaVersion != 1u) {
    return Result<AutotestCase, std::string>::makeError(
        "schemaVersion must be 1");
  }
  auto text = readString(root, "id", "$", true);
  if (text.hasError()) {
    return Result<AutotestCase, std::string>::makeError(text.error());
  }
  out.id = std::move(text.value());
  text = readString(root, "suite", "$", true);
  if (text.hasError()) {
    return Result<AutotestCase, std::string>::makeError(text.error());
  }
  out.suite = std::move(text.value());
  text = readString(root, "description", "$", false, out.description);
  if (text.hasError()) {
    return Result<AutotestCase, std::string>::makeError(text.error());
  }
  out.description = std::move(text.value());
  text = readString(root, "backend", "$", false, out.backend);
  if (text.hasError()) {
    return Result<AutotestCase, std::string>::makeError(text.error());
  }
  out.backend = std::move(text.value());
  text = readString(root, "presentMode", "$", false, out.presentMode);
  if (text.hasError()) {
    return Result<AutotestCase, std::string>::makeError(text.error());
  }
  out.presentMode = std::move(text.value());
  text = readString(root, "windowMode", "$", false, out.windowMode);
  if (text.hasError()) {
    return Result<AutotestCase, std::string>::makeError(text.error());
  }
  out.windowMode = std::move(text.value());

  if (yyjson_val *resolution = optionalObject(root, "resolution")) {
    if (!yyjson_is_arr(resolution) || yyjson_arr_size(resolution) != 2u) {
      return Result<AutotestCase, std::string>::makeError(
          "resolution must be [width, height]");
    }
    for (uint32_t i = 0u; i < 2u; ++i) {
      yyjson_val *entry = yyjson_arr_get(resolution, i);
      if (!yyjson_is_uint(entry) || yyjson_get_uint(entry) > UINT32_MAX) {
        return Result<AutotestCase, std::string>::makeError(
            "resolution entries must be uint32");
      }
      out.resolution[i] = static_cast<uint32_t>(yyjson_get_uint(entry));
    }
  }

  auto number =
      readDouble(root, "fixedDeltaSeconds", "$", out.fixedDeltaSeconds);
  if (number.hasError()) {
    return Result<AutotestCase, std::string>::makeError(number.error());
  }
  out.fixedDeltaSeconds = number.value();
  auto u32 = readU32(root, "warmupFrames", "$", out.warmupFrames);
  if (u32.hasError()) {
    return Result<AutotestCase, std::string>::makeError(u32.error());
  }
  out.warmupFrames = u32.value();
  u32 = readU32(root, "endFrame", "$", out.endFrame);
  if (u32.hasError()) {
    return Result<AutotestCase, std::string>::makeError(u32.error());
  }
  out.endFrame = std::max(out.warmupFrames, u32.value());
  auto boolean = readBool(root, "authoritative", "$", out.authoritative);
  if (boolean.hasError()) {
    return Result<AutotestCase, std::string>::makeError(boolean.error());
  }
  out.authoritative = boolean.value();

  if (yyjson_val *scene = optionalObject(root, "scene")) {
    auto parsed = parseScene(scene, out.scene);
    if (parsed.hasError()) {
      return Result<AutotestCase, std::string>::makeError(parsed.error());
    }
  }
  if (yyjson_val *environment = optionalObject(root, "environment")) {
    auto parsed = parseEnvironment(environment, out.environment);
    if (parsed.hasError()) {
      return Result<AutotestCase, std::string>::makeError(parsed.error());
    }
  }
  if (yyjson_val *camera = optionalObject(root, "camera")) {
    auto parsed = parseCamera(camera, out.camera);
    if (parsed.hasError()) {
      return Result<AutotestCase, std::string>::makeError(parsed.error());
    }
  }
  if (yyjson_val *settings = optionalObject(root, "settings")) {
    auto parsed = parseSettings(settings, out.settings);
    if (parsed.hasError()) {
      return Result<AutotestCase, std::string>::makeError(parsed.error());
    }
  }
  if (yyjson_val *renderGraph = optionalObject(root, "renderGraph")) {
    static constexpr std::array keys{std::string_view("workerCount"),
                                     std::string_view("parallelCompile"),
                                     std::string_view("parallelRecording")};
    auto parsed = rejectUnknownKeys(renderGraph, keys, "renderGraph");
    if (parsed.hasError()) {
      return Result<AutotestCase, std::string>::makeError(parsed.error());
    }
    u32 = readU32(renderGraph, "workerCount", "renderGraph",
                  out.renderGraph.workerCount);
    if (u32.hasError()) {
      return Result<AutotestCase, std::string>::makeError(u32.error());
    }
    out.renderGraph.workerCount = std::max(1u, u32.value());
    boolean = readBool(renderGraph, "parallelCompile", "renderGraph",
                       out.renderGraph.parallelCompile);
    if (boolean.hasError()) {
      return Result<AutotestCase, std::string>::makeError(boolean.error());
    }
    out.renderGraph.parallelCompile = boolean.value();
    boolean = readBool(renderGraph, "parallelRecording", "renderGraph",
                       out.renderGraph.parallelRecording);
    if (boolean.hasError()) {
      return Result<AutotestCase, std::string>::makeError(boolean.error());
    }
    out.renderGraph.parallelRecording = boolean.value();
  }
  if (yyjson_val *requirements = optionalObject(root, "requirements")) {
    static constexpr std::array keys{std::string_view("assets"),
                                     std::string_view("backends"),
                                     std::string_view("allowVisibleWindow")};
    auto parsed = rejectUnknownKeys(requirements, keys, "requirements");
    if (parsed.hasError()) {
      return Result<AutotestCase, std::string>::makeError(parsed.error());
    }
    auto array = readStringArray(requirements, "assets", "requirements");
    if (array.hasError()) {
      return Result<AutotestCase, std::string>::makeError(array.error());
    }
    out.requirements.assets = std::move(array.value());
    array = readStringArray(requirements, "backends", "requirements");
    if (array.hasError()) {
      return Result<AutotestCase, std::string>::makeError(array.error());
    }
    out.requirements.backends = std::move(array.value());
    boolean = readBool(requirements, "allowVisibleWindow", "requirements",
                       out.requirements.allowVisibleWindow);
    if (boolean.hasError()) {
      return Result<AutotestCase, std::string>::makeError(boolean.error());
    }
    out.requirements.allowVisibleWindow = boolean.value();
  }

  auto timeline = parseTimeline(root, out.camera, out.settings, out.endFrame);
  if (timeline.hasError()) {
    return Result<AutotestCase, std::string>::makeError(timeline.error());
  }
  out.timeline = std::move(timeline.value());

  auto checkpoints = parseCheckpoints(root, out.endFrame);
  if (checkpoints.hasError()) {
    return Result<AutotestCase, std::string>::makeError(checkpoints.error());
  }
  out.checkpoints = std::move(checkpoints.value());

  auto metricWindows = parseMetricWindows(root, out.endFrame);
  if (metricWindows.hasError()) {
    return Result<AutotestCase, std::string>::makeError(metricWindows.error());
  }
  out.metricWindows = std::move(metricWindows.value());
  return Result<AutotestCase, std::string>::makeResult(std::move(out));
}

Result<std::vector<AutotestCase>, std::string>
discoverAutotestCases(const AutotestManifestLoadOptions &options) {
  const std::filesystem::path root =
      options.caseRoot.empty() ? defaultAutotestCaseRoot() : options.caseRoot;
  std::vector<AutotestCase> cases;
  if (!std::filesystem::exists(root)) {
    return Result<std::vector<AutotestCase>, std::string>::makeResult(
        std::move(cases));
  }
  for (const std::filesystem::directory_entry &entry :
       std::filesystem::recursive_directory_iterator(root)) {
    if (!entry.is_regular_file() || entry.path().extension() != ".json") {
      continue;
    }
    auto loaded = loadAutotestCaseManifest(entry.path());
    if (loaded.hasError()) {
      return Result<std::vector<AutotestCase>, std::string>::makeError(
          loaded.error());
    }
    cases.push_back(std::move(loaded.value()));
  }
  std::sort(cases.begin(), cases.end(),
            [](const AutotestCase &lhs, const AutotestCase &rhs) {
              return lhs.id < rhs.id;
            });
  return Result<std::vector<AutotestCase>, std::string>::makeResult(
      std::move(cases));
}

const AutotestCase *findAutotestCaseById(const std::vector<AutotestCase> &cases,
                                         std::string_view id) {
  for (const AutotestCase &testCase : cases) {
    if (testCase.id == id) {
      return &testCase;
    }
  }
  return nullptr;
}

std::vector<const AutotestCase *>
filterAutotestCasesBySuite(const std::vector<AutotestCase> &cases,
                           std::string_view suite) {
  std::vector<const AutotestCase *> out;
  for (const AutotestCase &testCase : cases) {
    if (suite.empty() || testCase.suite == suite) {
      out.push_back(&testCase);
    }
  }
  return out;
}

} // namespace nuri::tools::autotest
