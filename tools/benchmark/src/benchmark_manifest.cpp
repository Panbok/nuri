#include "nuri/tools/benchmark/benchmark_manifest.h"

#include "nuri/core/runtime_config.h"
#include "nuri/tools/benchmark/benchmark_environment.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <climits>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <string_view>

#include <yyjson.h>

namespace nuri::tools::benchmark {
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
      std::string_view("enableCpuFrustumCulling"),
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
  b = readBool(object, "enableCpuFrustumCulling", path,
               settings.opaque.enableCpuFrustumCulling);
  if (b.hasError()) {
    return Result<bool, std::string>::makeError(b.error());
  }
  settings.opaque.enableCpuFrustumCulling = b.value();
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
      return Result<bool, std::string>::makeResult(true);
    }
    if (!yyjson_is_sint(forced) || yyjson_get_sint(forced) < INT32_MIN ||
        yyjson_get_sint(forced) > INT32_MAX) {
      return Result<bool, std::string>::makeError(
          jsonPath(path, "forcedMeshLod") + " must be an int32");
    }
    settings.opaque.forcedMeshLod =
        static_cast<int32_t>(yyjson_get_sint(forced));
  }
  return Result<bool, std::string>::makeResult(true);
}

[[nodiscard]] Result<bool, std::string>
parseAntiAliasingSettings(yyjson_val *object, RenderSettings &settings,
                          std::string_view path) {
  static constexpr std::array keys{std::string_view("mode"),
                                   std::string_view("qualityPreset")};
  auto keysResult = rejectUnknownKeys(object, keys, path);
  if (keysResult.hasError()) {
    return keysResult;
  }
  auto result =
      readEnumField(object, "mode", path, settings.antiAliasing.mode,
                    {{"None", AntiAliasingMode::None},
                     {"TAA", AntiAliasingMode::TAA},
                     {"SpatialFallback", AntiAliasingMode::SpatialFallback},
                     {"MSAA4x", AntiAliasingMode::MSAA4x}});
  if (result.hasError()) {
    return result;
  }
  return readEnumField(object, "qualityPreset", path,
                       settings.antiAliasing.qualityPreset,
                       {{"Performance", TemporalAAQualityPreset::Performance},
                        {"Balanced", TemporalAAQualityPreset::Balanced},
                        {"Quality", TemporalAAQualityPreset::Quality},
                        {"Ultra", TemporalAAQualityPreset::Ultra},
                        {"Custom", TemporalAAQualityPreset::Custom}});
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
      std::string_view("enabled"), std::string_view("qualityPreset"),
      std::string_view("enableMeshletDepth"),
      std::string_view("enableMeshletCascadeCulling")};
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
  enabled = readBool(object, "enableMeshletDepth", path,
                     settings.shadow.enableMeshletDepth);
  if (enabled.hasError()) {
    return Result<bool, std::string>::makeError(enabled.error());
  }
  settings.shadow.enableMeshletDepth = enabled.value();
  enabled = readBool(object, "enableMeshletCascadeCulling", path,
                     settings.shadow.enableMeshletCascadeCulling);
  if (enabled.hasError()) {
    return Result<bool, std::string>::makeError(enabled.error());
  }
  settings.shadow.enableMeshletCascadeCulling = enabled.value();
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
      std::string_view("enableShadowMeshletCulling"),
      std::string_view("enableGpuInstanceCulling"),
      std::string_view("enableGpuIndirectDraw"),
      std::string_view("enableIndirectMeshDispatch"),
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
  b = readBool(object, "enableShadowMeshletCulling", path,
               settings.visibility.enableShadowMeshletCulling);
  if (b.hasError()) {
    return Result<bool, std::string>::makeError(b.error());
  }
  settings.visibility.enableShadowMeshletCulling = b.value();
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
parseSettings(yyjson_val *object, RenderSettings &settings) {
  static constexpr std::array keys{
      std::string_view("opaque"),
      std::string_view("antiAliasing"),
      std::string_view("ambientOcclusion"),
      std::string_view("shadow"),
      std::string_view("hdrPostProcess"),
      std::string_view("transmission"),
      std::string_view("transparent"),
      std::string_view("textureFiltering"),
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
      static constexpr std::array meshKeys{std::string_view("flipUVs")};
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
      std::string_view("schemaVersion"),  std::string_view("id"),
      std::string_view("suite"),          std::string_view("description"),
      std::string_view("scene"),          std::string_view("backend"),
      std::string_view("resolution"),     std::string_view("fixedDeltaSeconds"),
      std::string_view("warmupFrames"),   std::string_view("measurementFrames"),
      std::string_view("cooldownFrames"), std::string_view("maxDrainFrames"),
      std::string_view("drainTimeoutMs"), std::string_view("samples"),
      std::string_view("authoritative"),  std::string_view("presentMode"),
      std::string_view("renderGraph"),    std::string_view("camera"),
      std::string_view("settings"),       std::string_view("requirements"),
      std::string_view("thresholds"),     std::string_view("requiredMetrics"),
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
  text = readString(root, "suite", "$", true);
  if (text.hasError()) {
    return Result<BenchmarkCase, std::string>::makeError(text.error());
  }
  out.suite = std::move(text.value());
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
  text = readString(root, "presentMode", "$", false, out.presentMode);
  if (text.hasError()) {
    return Result<BenchmarkCase, std::string>::makeError(text.error());
  }
  out.presentMode = std::move(text.value());

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
    }
  }

  auto number =
      readDouble(root, "fixedDeltaSeconds", "$", out.fixedDeltaSeconds);
  if (number.hasError()) {
    return Result<BenchmarkCase, std::string>::makeError(number.error());
  }
  out.fixedDeltaSeconds = number.value();
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
  out.samples = std::max(1u, u32.value());
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
    out.renderGraph.workerCount = std::max(1u, u32.value());
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
                                     std::string_view("allowVisibleWindow")};
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
    boolean = readBool(requirements, "allowVisibleWindow", "requirements",
                       out.requirements.allowVisibleWindow);
    if (boolean.hasError()) {
      return Result<BenchmarkCase, std::string>::makeError(boolean.error());
    }
    out.requirements.allowVisibleWindow = boolean.value();
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
  }

  auto metrics = readStringArray(root, "requiredMetrics", "$");
  if (metrics.hasError()) {
    return Result<BenchmarkCase, std::string>::makeError(metrics.error());
  }
  out.requiredMetrics = std::move(metrics.value());
  if (out.requiredMetrics.empty()) {
    out.requiredMetrics.push_back("cpu.render_submit_ms");
  }
  return Result<BenchmarkCase, std::string>::makeResult(std::move(out));
}

Result<std::vector<BenchmarkCase>, std::string>
discoverBenchmarkCases(const BenchmarkManifestLoadOptions &options) {
  const std::filesystem::path root =
      options.caseRoot.empty() ? defaultBenchmarkCaseRoot() : options.caseRoot;
  std::vector<BenchmarkCase> cases;
  if (!std::filesystem::exists(root)) {
    return Result<std::vector<BenchmarkCase>, std::string>::makeResult(
        std::move(cases));
  }
  for (const std::filesystem::directory_entry &entry :
       std::filesystem::recursive_directory_iterator(root)) {
    if (!entry.is_regular_file() || entry.path().extension() != ".json") {
      continue;
    }
    auto loaded = loadBenchmarkCaseManifest(entry.path());
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
  for (const BenchmarkCase &benchmarkCase : cases) {
    if (suite.empty() || benchmarkCase.suite == suite) {
      out.push_back(&benchmarkCase);
    }
  }
  return out;
}

} // namespace nuri::tools::benchmark
