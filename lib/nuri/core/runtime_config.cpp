#include "nuri/core/runtime_config.h"
#include "nuri/core/profiling.h"
#include "nuri/pch.h"
namespace nuri {
namespace {
constexpr std::string_view kDefaultConfigPath = "app.config.json";
constexpr const char kAppConfigEnvVar[] = "NURI_APP_CONFIG";
template <typename T> struct PathField {
  const char *key;
  const char *fallback;
  std::filesystem::path T::*member;
  bool textureRoot = false;
};
template <typename T> struct RootField {
  const char *key;
  std::filesystem::path T::*member;
};
constexpr std::array kRootFields = {
    RootField<RuntimeRootsConfig>{"assets", &RuntimeRootsConfig::assets},
    RootField<RuntimeRootsConfig>{"shaders", &RuntimeRootsConfig::shaders},
    RootField<RuntimeRootsConfig>{"models", &RuntimeRootsConfig::models},
    RootField<RuntimeRootsConfig>{"textures", &RuntimeRootsConfig::textures},
    RootField<RuntimeRootsConfig>{"fonts", &RuntimeRootsConfig::fonts},
};
constexpr std::array kDebugFields = {
    PathField<RuntimeRasterShaderConfig>{"vertex", "grid.vert",
                                         &RuntimeRasterShaderConfig::vertex},
    PathField<RuntimeRasterShaderConfig>{"fragment", "grid.frag",
                                         &RuntimeRasterShaderConfig::fragment},
};
constexpr std::array kSkyboxFields = {
    PathField<RuntimeRasterShaderConfig>{"vertex", "skybox.vert",
                                         &RuntimeRasterShaderConfig::vertex},
    PathField<RuntimeRasterShaderConfig>{"fragment", "skybox.frag",
                                         &RuntimeRasterShaderConfig::fragment},
};
constexpr std::array kOpaqueFields = {
    PathField<RuntimeOpaqueShaderConfig>{
        "mesh_vertex", "main.vert", &RuntimeOpaqueShaderConfig::meshVertex},
    PathField<RuntimeOpaqueShaderConfig>{
        "mesh_fragment", "main.frag", &RuntimeOpaqueShaderConfig::meshFragment},
    PathField<RuntimeOpaqueShaderConfig>{
        "meshlet_task", "opaque_meshlet.task.glsl",
        &RuntimeOpaqueShaderConfig::meshletTask},
    PathField<RuntimeOpaqueShaderConfig>{
        "meshlet_mesh", "opaque_meshlet.mesh.glsl",
        &RuntimeOpaqueShaderConfig::meshletMesh},
    PathField<RuntimeOpaqueShaderConfig>{
        "meshlet_fragment", "opaque_meshlet.frag",
        &RuntimeOpaqueShaderConfig::meshletFragment},
    PathField<RuntimeOpaqueShaderConfig>{
        "meshlet_depth_fragment", "opaque_meshlet_depth.frag",
        &RuntimeOpaqueShaderConfig::meshletDepthFragment},
    PathField<RuntimeOpaqueShaderConfig>{
        "meshlet_depth_alpha_fragment", "opaque_meshlet_depth_alpha.frag",
        &RuntimeOpaqueShaderConfig::meshletDepthAlphaFragment},
    PathField<RuntimeOpaqueShaderConfig>{
        "pick_fragment", "main_id.frag",
        &RuntimeOpaqueShaderConfig::pickFragment},
    PathField<RuntimeOpaqueShaderConfig>{
        "shadow_inspect_fragment", "shadow_inspect.frag",
        &RuntimeOpaqueShaderConfig::shadowInspectFragment},
    PathField<RuntimeOpaqueShaderConfig>{
        "compute_instances", "duck_instances.comp",
        &RuntimeOpaqueShaderConfig::computeInstances},
    PathField<RuntimeOpaqueShaderConfig>{
        "tess_vertex", "main_tess.vert",
        &RuntimeOpaqueShaderConfig::tessVertex},
    PathField<RuntimeOpaqueShaderConfig>{
        "tess_control", "main.tesc", &RuntimeOpaqueShaderConfig::tessControl},
    PathField<RuntimeOpaqueShaderConfig>{"tess_eval", "main.tese",
                                         &RuntimeOpaqueShaderConfig::tessEval},
    PathField<RuntimeOpaqueShaderConfig>{
        "overlay_geometry", "mesh_debug_overlay.geom",
        &RuntimeOpaqueShaderConfig::overlayGeometry},
    PathField<RuntimeOpaqueShaderConfig>{
        "overlay_fragment", "mesh_debug_overlay.frag",
        &RuntimeOpaqueShaderConfig::overlayFragment},
};
constexpr std::array kCompositeFields = {
    PathField<RuntimeCompositeConfig>{
        "fullscreen_vertex", "fullscreen_copy.vert",
        &RuntimeCompositeConfig::fullscreenVertex},
    PathField<RuntimeCompositeConfig>{
        "scene_copy_fragment", "scene_copy.frag",
        &RuntimeCompositeConfig::sceneCopyFragment},
    PathField<RuntimeCompositeConfig>{"present_fragment",
                                      "tonemap_present.frag",
                                      &RuntimeCompositeConfig::presentFragment},
    PathField<RuntimeCompositeConfig>{
        "hdr_luminance_reduce_fragment", "hdr_luminance_reduce.frag",
        &RuntimeCompositeConfig::hdrLuminanceReduceFragment},
    PathField<RuntimeCompositeConfig>{
        "hdr_exposure_adapt_fragment", "hdr_exposure_adapt.frag",
        &RuntimeCompositeConfig::hdrExposureAdaptFragment},
    PathField<RuntimeCompositeConfig>{
        "hdr_bloom_fragment", "hdr_bloom.frag",
        &RuntimeCompositeConfig::hdrBloomFragment},
    PathField<RuntimeCompositeConfig>{
        "hdr_bloom_composite_fragment", "hdr_bloom_composite.frag",
        &RuntimeCompositeConfig::hdrBloomCompositeFragment},
    PathField<RuntimeCompositeConfig>{
        "aces2_sdr_lut", "tonemap_aces2_sdr_64.ktx2",
        &RuntimeCompositeConfig::aces2SdrLut, true},
    PathField<RuntimeCompositeConfig>{"agx_lut", "tonemap_agx_sdr_64.ktx2",
                                      &RuntimeCompositeConfig::agxLut, true},
};
constexpr std::array kTextFields = {
    PathField<RuntimeTextMtsdfShaderConfig>{
        "ui_vertex", "text_2d_mtsdf.vert",
        &RuntimeTextMtsdfShaderConfig::uiVertex},
    PathField<RuntimeTextMtsdfShaderConfig>{
        "ui_fragment", "text_2d_mtsdf.frag",
        &RuntimeTextMtsdfShaderConfig::uiFragment},
    PathField<RuntimeTextMtsdfShaderConfig>{
        "world_vertex", "text_3d_mtsdf.vert",
        &RuntimeTextMtsdfShaderConfig::worldVertex},
    PathField<RuntimeTextMtsdfShaderConfig>{
        "world_fragment", "text_3d_mtsdf.frag",
        &RuntimeTextMtsdfShaderConfig::worldFragment},
};
constexpr std::array kDDGIFields = {
    PathField<RuntimeDDGIShaderConfig>{
        "decode_positions", "rt_decode_positions.comp",
        &RuntimeDDGIShaderConfig::decodePositions},
    PathField<RuntimeDDGIShaderConfig>{
        "prepare_dynamic_vertices", "rt_prepare_dynamic_vertices.comp",
        &RuntimeDDGIShaderConfig::prepareDynamicVertices},
    PathField<RuntimeDDGIShaderConfig>{"trace", "ddgi_trace.comp",
                                       &RuntimeDDGIShaderConfig::trace},
    PathField<RuntimeDDGIShaderConfig>{"trace_inspect",
                                       "ddgi_trace_inspect.comp",
                                       &RuntimeDDGIShaderConfig::traceInspect},
    PathField<RuntimeDDGIShaderConfig>{
        "blend_irradiance", "ddgi_blend_irradiance.comp",
        &RuntimeDDGIShaderConfig::blendIrradiance},
    PathField<RuntimeDDGIShaderConfig>{"blend_distance",
                                       "ddgi_blend_distance.comp",
                                       &RuntimeDDGIShaderConfig::blendDistance},
    PathField<RuntimeDDGIShaderConfig>{
        "update_probe_state", "ddgi_update_probe_state.comp",
        &RuntimeDDGIShaderConfig::updateProbeState},
    PathField<RuntimeDDGIShaderConfig>{
        "probe_debug_vertex", "ddgi_probe_debug.vert",
        &RuntimeDDGIShaderConfig::probeDebugVertex},
    PathField<RuntimeDDGIShaderConfig>{
        "probe_debug_fragment", "ddgi_probe_debug.frag",
        &RuntimeDDGIShaderConfig::probeDebugFragment},
    PathField<RuntimeDDGIShaderConfig>{
        "ray_debug_vertex", "ddgi_ray_debug.vert",
        &RuntimeDDGIShaderConfig::rayDebugVertex},
    PathField<RuntimeDDGIShaderConfig>{
        "ray_debug_fragment", "ddgi_ray_debug.frag",
        &RuntimeDDGIShaderConfig::rayDebugFragment},
};
constexpr auto kDDGIConfigFields = std::to_array<std::string_view>(
    {"persistent_memory_limit_mb", "peak_memory_limit_mb"});
template <typename T>
[[nodiscard]] Result<T, std::string> makeError(std::string message) {
  return Result<T, std::string>::makeError(std::move(message));
}
[[nodiscard]] std::filesystem::path
normalizePath(const std::filesystem::path &path) {
  std::error_code ec;
  auto normalized = std::filesystem::weakly_canonical(path, ec);
  if (ec) {
    normalized = std::filesystem::absolute(path, ec);
  }
  return ec ? path.lexically_normal() : normalized.lexically_normal();
}
[[nodiscard]] std::string fieldPath(std::string_view parent,
                                    std::string_view child) {
  return parent.empty() ? std::string(child)
                        : std::string(parent) + "." + std::string(child);
}
class ConfigReader {
public:
  void fail(std::string message) {
    if (error_.empty()) {
      error_ = std::move(message);
    }
  }
  template <typename Allowed>
  void validate(yyjson_val *object, std::string_view name,
                const Allowed &allowed) {
    if (!object || !error_.empty()) {
      return;
    }
    size_t index = 0;
    size_t count = 0;
    yyjson_val *key = nullptr;
    yyjson_val *value = nullptr;
    yyjson_obj_foreach(object, index, count, key, value) {
      (void)value;
      const std::string_view keyName{yyjson_get_str(key)};
      if (!allowed(keyName)) {
        fail("Unknown config field '" + fieldPath(name, keyName) + "'");
        return;
      }
    }
  }
  [[nodiscard]] yyjson_val *object(yyjson_val *parent, const char *key,
                                   std::string_view parentName, bool required) {
    if (!parent || !error_.empty()) {
      return nullptr;
    }
    yyjson_val *value = yyjson_obj_get(parent, key);
    if (!value) {
      if (required) {
        fail("Missing required config field '" + fieldPath(parentName, key) +
             "'");
      }
      return nullptr;
    }
    if (!yyjson_is_obj(value)) {
      fail("Config field '" + fieldPath(parentName, key) +
           "' must be a JSON object");
      return nullptr;
    }
    return value;
  }
  [[nodiscard]] std::string text(yyjson_val *object, const char *key,
                                 std::string_view parentName,
                                 const char *fallback = nullptr) {
    if (!error_.empty() || !object) {
      return fallback ? std::string(fallback) : std::string{};
    }
    yyjson_val *value = yyjson_obj_get(object, key);
    if (!value) {
      if (fallback) {
        return fallback;
      }
      fail("Missing required config field '" + fieldPath(parentName, key) +
           "'");
      return {};
    }
    const char *raw = yyjson_is_str(value) ? yyjson_get_str(value) : nullptr;
    if (!raw || raw[0] == '\0') {
      fail("Config field '" + fieldPath(parentName, key) +
           (raw ? "' must not be empty" : "' must be a string"));
      return {};
    }
    return raw;
  }
  [[nodiscard]] int32_t positiveInt(yyjson_val *object, const char *key,
                                    std::string_view parentName) {
    if (!object || !error_.empty()) {
      return 0;
    }
    yyjson_val *value = yyjson_obj_get(object, key);
    if (!value) {
      fail("Missing required config field '" + fieldPath(parentName, key) +
           "'");
      return 0;
    }
    if (!yyjson_is_sint(value) && !yyjson_is_uint(value)) {
      fail("Config field '" + fieldPath(parentName, key) +
           "' must be an integer");
      return 0;
    }
    const uint64_t raw = yyjson_is_uint(value)
                             ? yyjson_get_uint(value)
                             : static_cast<uint64_t>(yyjson_get_sint(value));
    if (raw == 0 || raw > std::numeric_limits<int32_t>::max()) {
      fail("Config field '" + fieldPath(parentName, key) +
           "' must be in range [1, 2147483647]");
      return 0;
    }
    return static_cast<int32_t>(raw);
  }
  [[nodiscard]] int32_t positiveIntOr(yyjson_val *object, const char *key,
                                      std::string_view parentName,
                                      int32_t fallback) {
    if (!object || !yyjson_obj_get(object, key)) {
      return fallback;
    }
    return positiveInt(object, key, parentName);
  }
  [[nodiscard]] std::filesystem::path path(std::string_view raw,
                                           const std::filesystem::path &base,
                                           std::string_view fieldName,
                                           bool directory) {
    if (!error_.empty()) {
      return {};
    }
    std::filesystem::path resolved{raw};
    if (!resolved.is_absolute()) {
      resolved = base / resolved;
    }
    resolved = normalizePath(resolved);
    std::error_code ec;
    const auto status = std::filesystem::status(resolved, ec);
    if (ec || !std::filesystem::exists(status)) {
      fail("Config field '" + std::string(fieldName) + "' resolves to '" +
           resolved.string() + "' but it does not exist");
    } else if (directory != std::filesystem::is_directory(status)) {
      fail("Config field '" + std::string(fieldName) + "' resolves to '" +
           resolved.string() +
           (directory ? "' but it is not a directory"
                      : "' but it is not a regular file"));
    } else if (!directory && !std::filesystem::is_regular_file(status)) {
      fail("Config field '" + std::string(fieldName) + "' resolves to '" +
           resolved.string() + "' but it is not a regular file");
    }
    return resolved;
  }
  [[nodiscard]] const std::string &error() const { return error_; }

private:
  std::string error_;
};
template <typename Fields>
[[nodiscard]] auto fieldPredicate(const Fields &fields) {
  return [&fields](std::string_view key) {
    return std::ranges::any_of(
        fields, [key](const auto &field) { return key == field.key; });
  };
}
template <typename T, size_t N>
void readRoots(ConfigReader &reader, yyjson_val *object,
               const std::filesystem::path &base, T &target,
               const std::array<RootField<T>, N> &fields) {
  reader.validate(object, "roots", fieldPredicate(fields));
  for (const auto &field : fields) {
    target.*field.member =
        reader.path(reader.text(object, field.key, "roots"), base,
                    fieldPath("roots", field.key), true);
  }
}
template <typename T, size_t N>
void readPaths(ConfigReader &reader, yyjson_val *object,
               std::string_view section,
               const std::filesystem::path &shaderRoot,
               const std::filesystem::path &textureRoot, T &target,
               const std::array<PathField<T>, N> &fields,
               std::span<const std::string_view> additionalFields = {}) {
  const auto pathField = fieldPredicate(fields);
  reader.validate(object, section, [&](std::string_view key) {
    return pathField(key) ||
           std::ranges::find(additionalFields, key) != additionalFields.end();
  });
  for (const auto &field : fields) {
    const auto &root = field.textureRoot ? textureRoot : shaderRoot;
    target.*field.member =
        reader.path(reader.text(object, field.key, section, field.fallback),
                    root, fieldPath(section, field.key), false);
  }
}
[[nodiscard]] Result<std::string, std::string>
readTextFile(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return makeError<std::string>("Failed to open config file '" +
                                  path.string() + "'");
  }
  std::ostringstream stream;
  stream << input.rdbuf();
  if (input.bad()) {
    return makeError<std::string>("Failed to read config file '" +
                                  path.string() + "'");
  }
  return Result<std::string, std::string>::makeResult(stream.str());
}
} // namespace

Result<RuntimeConfig, std::string>
loadRuntimeConfig(const std::filesystem::path &configPath) {
  NURI_PROFILER_FUNCTION();
  const auto normalizedPath = normalizePath(configPath);
  auto text = readTextFile(normalizedPath);
  if (text.hasError()) {
    return makeError<RuntimeConfig>(text.error());
  }
  yyjson_read_err parseError{};
  yyjson_doc *rawDocument = yyjson_read_opts(
      text.value().data(), text.value().size(), 0, nullptr, &parseError);
  if (!rawDocument) {
    return makeError<RuntimeConfig>(
        "Failed to parse app config '" + normalizedPath.string() +
        "': " + std::to_string(parseError.pos) + ": " +
        (parseError.msg ? parseError.msg : "unknown parse error"));
  }
  std::unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)> document(
      rawDocument, &yyjson_doc_free);
  ConfigReader reader;
  yyjson_val *root = yyjson_doc_get_root(document.get());
  if (!yyjson_is_obj(root)) {
    reader.fail("App config root must be a JSON object");
    root = nullptr;
  }
  constexpr auto rootKeys =
      std::to_array<std::string_view>({"window", "roots", "shaders"});
  constexpr auto windowKeys =
      std::to_array<std::string_view>({"title", "width", "height", "mode"});
  constexpr auto shaderKeys = std::to_array<std::string_view>(
      {"debug_grid", "skybox", "opaque", "composite", "text_mtsdf", "ddgi"});
  const auto listed = [](const auto &keys) {
    return [&keys](std::string_view key) {
      return std::ranges::find(keys, key) != keys.end();
    };
  };
  reader.validate(root, "", listed(rootKeys));
  yyjson_val *window = reader.object(root, "window", "", true);
  yyjson_val *roots = reader.object(root, "roots", "", true);
  yyjson_val *shaders = reader.object(root, "shaders", "", false);
  reader.validate(window, "window", listed(windowKeys));
  reader.validate(shaders, "shaders", listed(shaderKeys));
  RuntimeConfig config{};
  config.sourcePath = normalizedPath;
  config.window.title = reader.text(window, "title", "window");
  config.window.width = reader.positiveInt(window, "width", "window");
  config.window.height = reader.positiveInt(window, "height", "window");
  const std::string mode = reader.text(window, "mode", "window");
  if (mode == "windowed") {
    config.window.mode = WindowMode::Windowed;
  } else if (mode == "fullscreen") {
    config.window.mode = WindowMode::Fullscreen;
  } else if (mode == "borderless_fullscreen") {
    config.window.mode = WindowMode::BorderlessFullscreen;
  } else if (reader.error().empty()) {
    reader.fail(
        "Invalid window.mode '" + mode +
        "'. Allowed values: windowed, fullscreen, borderless_fullscreen");
  }
  const auto configDirectory = normalizedPath.parent_path();
  readRoots(reader, roots, configDirectory, config.roots, kRootFields);
  yyjson_val *debug = reader.object(shaders, "debug_grid", "shaders", false);
  yyjson_val *skybox = reader.object(shaders, "skybox", "shaders", false);
  yyjson_val *opaque = reader.object(shaders, "opaque", "shaders", false);
  yyjson_val *composite = reader.object(shaders, "composite", "shaders", false);
  yyjson_val *textMtsdf =
      reader.object(shaders, "text_mtsdf", "shaders", false);
  yyjson_val *ddgi = reader.object(shaders, "ddgi", "shaders", false);
  readPaths(reader, debug, "shaders.debug_grid", config.roots.shaders,
            config.roots.textures, config.shaders.debugGrid, kDebugFields);
  readPaths(reader, skybox, "shaders.skybox", config.roots.shaders,
            config.roots.textures, config.shaders.skybox, kSkyboxFields);
  config.shaders.opaque.shaderBasePath = config.roots.shaders;
  readPaths(reader, opaque, "shaders.opaque", config.roots.shaders,
            config.roots.textures, config.shaders.opaque, kOpaqueFields);
  config.shaders.composite.shaderBasePath = config.roots.shaders;
  readPaths(reader, composite, "shaders.composite", config.roots.shaders,
            config.roots.textures, config.shaders.composite, kCompositeFields);
  readPaths(reader, textMtsdf, "shaders.text_mtsdf", config.roots.shaders,
            config.roots.textures, config.shaders.textMtsdf, kTextFields);
  config.shaders.ddgi.shaderBasePath = config.roots.shaders;
  readPaths(reader, ddgi, "shaders.ddgi", config.roots.shaders,
            config.roots.textures, config.shaders.ddgi, kDDGIFields,
            kDDGIConfigFields);
  constexpr uint64_t kMiB = 1024ull * 1024ull;
  config.shaders.ddgi.persistentMemoryLimitBytes =
      static_cast<uint64_t>(reader.positiveIntOr(
          ddgi, "persistent_memory_limit_mb", "shaders.ddgi", 256)) *
      kMiB;
  config.shaders.ddgi.peakMemoryLimitBytes =
      static_cast<uint64_t>(reader.positiveIntOr(ddgi, "peak_memory_limit_mb",
                                                 "shaders.ddgi", 512)) *
      kMiB;
  if (config.shaders.ddgi.peakMemoryLimitBytes <
          config.shaders.ddgi.persistentMemoryLimitBytes &&
      reader.error().empty()) {
    reader.fail("Config field 'shaders.ddgi.peak_memory_limit_mb' must be "
                "greater than or equal to persistent_memory_limit_mb");
  }
  if (!reader.error().empty()) {
    return makeError<RuntimeConfig>(reader.error());
  }
  return Result<RuntimeConfig, std::string>::makeResult(std::move(config));
}

Result<RuntimeConfig, std::string> loadRuntimeConfig() {
  return loadRuntimeConfig(std::filesystem::path{kDefaultConfigPath});
}

Result<RuntimeConfig, std::string> loadRuntimeConfigFromEnvOrDefault() {
#if defined(_WIN32)
  char *raw = nullptr;
  size_t size = 0;
  const int error = _dupenv_s(&raw, &size, kAppConfigEnvVar);
  std::unique_ptr<char, decltype(&std::free)> value(raw, &std::free);
  if (error != 0) {
    return makeError<RuntimeConfig>("Failed to read environment variable '" +
                                    std::string(kAppConfigEnvVar) +
                                    "' (error " + std::to_string(error) + ")");
  }
  if (value && value.get()[0] != '\0') {
    return loadRuntimeConfig(value.get());
  }
#else
  const char *value = std::getenv(kAppConfigEnvVar);
  if (value && value[0] != '\0') {
    return loadRuntimeConfig(value);
  }
#endif
  return loadRuntimeConfig();
}

} // namespace nuri
