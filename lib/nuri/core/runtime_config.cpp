#include "nuri/pch.h"

#include "nuri/core/profiling.h"
#include "nuri/core/runtime_config.h"

namespace nuri {
namespace {

constexpr std::string_view kDefaultDebugGridVertexShader = "grid.vert";
constexpr std::string_view kDefaultDebugGridFragmentShader = "grid.frag";
constexpr std::string_view kDefaultSkyboxVertexShader = "skybox.vert";
constexpr std::string_view kDefaultSkyboxFragmentShader = "skybox.frag";
constexpr std::string_view kDefaultOpaqueMeshVertexShader = "main.vert";
constexpr std::string_view kDefaultOpaqueMeshFragmentShader = "main.frag";
constexpr std::string_view kDefaultOpaquePickFragmentShader = "main_id.frag";
constexpr std::string_view kDefaultOpaqueComputeShader = "duck_instances.comp";
constexpr std::string_view kDefaultOpaqueTessVertexShader = "main_tess.vert";
constexpr std::string_view kDefaultOpaqueTessControlShader = "main.tesc";
constexpr std::string_view kDefaultOpaqueTessEvalShader = "main.tese";
constexpr std::string_view kDefaultOpaqueOverlayGeometryShader =
    "mesh_debug_overlay.geom";
constexpr std::string_view kDefaultOpaqueOverlayFragmentShader =
    "mesh_debug_overlay.frag";
constexpr std::string_view kDefaultConfigPath = "app.config.json";
constexpr const char kAppConfigEnvVarCStr[] = "NURI_APP_CONFIG";
constexpr std::string_view kAppConfigEnvVar = kAppConfigEnvVarCStr;

constexpr std::array<std::string_view, 3> kRootObjectKeys = {"window", "roots",
                                                             "shaders"};
constexpr std::array<std::string_view, 4> kWindowKeys = {"title", "width",
                                                         "height", "mode"};
constexpr std::array<std::string_view, 4> kRootsKeys = {"assets", "shaders",
                                                        "models", "textures"};
constexpr std::array<std::string_view, 3> kShadersKeys = {"debug_grid",
                                                          "skybox", "opaque"};
constexpr std::array<std::string_view, 2> kDebugGridShaderKeys = {"vertex",
                                                                  "fragment"};
constexpr std::array<std::string_view, 2> kSkyboxShaderKeys = {"vertex",
                                                               "fragment"};
constexpr std::array<std::string_view, 9> kOpaqueShaderKeys = {
    "mesh_vertex",  "mesh_fragment", "pick_fragment",   "compute_instances",
    "tess_vertex",  "tess_control",  "tess_eval",       "overlay_geometry",
    "overlay_fragment",
};

template <typename T>
[[nodiscard]] Result<T, std::string> makeError(std::string message) {
  return Result<T, std::string>::makeError(std::move(message));
}

[[nodiscard]] std::filesystem::path
normalizePath(const std::filesystem::path &filePath) {
  std::error_code ec;
  auto normalized = std::filesystem::weakly_canonical(filePath, ec);
  if (ec) {
    normalized = std::filesystem::absolute(filePath, ec);
  }
  if (ec) {
    return filePath.lexically_normal();
  }
  return normalized.lexically_normal();
}

[[nodiscard]] std::string fieldPath(std::string_view parent,
                                    std::string_view child) {
  if (parent.empty()) {
    return std::string(child);
  }
  return std::string(parent) + "." + std::string(child);
}

[[nodiscard]] Result<std::string, std::string>
readTextFile(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input.is_open()) {
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

template <size_t N>
[[nodiscard]] bool
isAllowedKey(std::string_view key,
             const std::array<std::string_view, N> &allowed) {
  return std::find(allowed.begin(), allowed.end(), key) != allowed.end();
}

template <size_t N>
[[nodiscard]] Result<bool, std::string>
validateUnknownKeys(yyjson_val *obj, std::string_view objectName,
                    const std::array<std::string_view, N> &allowedKeys) {
  if (!yyjson_is_obj(obj)) {
    return makeError<bool>("Config field '" + std::string(objectName) +
                           "' must be a JSON object");
  }

  size_t idx = 0;
  size_t max = 0;
  yyjson_val *key = nullptr;
  yyjson_val *value = nullptr;
  yyjson_obj_foreach(obj, idx, max, key, value) {
    (void)value;
    const char *keyRaw = yyjson_get_str(key);
    if (!keyRaw) {
      continue;
    }
    const std::string_view keyView{keyRaw};
    if (!isAllowedKey(keyView, allowedKeys)) {
      return makeError<bool>("Unknown config field '" +
                             fieldPath(objectName, keyView) + "'");
    }
  }
  return Result<bool, std::string>::makeResult(true);
}

[[nodiscard]] Result<yyjson_val *, std::string>
requireObjectField(yyjson_val *obj, const char *key,
                   std::string_view objectName) {
  yyjson_val *value = yyjson_obj_get(obj, key);
  if (!value) {
    return makeError<yyjson_val *>("Missing required config field '" +
                                   fieldPath(objectName, key) + "'");
  }
  if (!yyjson_is_obj(value)) {
    return makeError<yyjson_val *>("Config field '" +
                                   fieldPath(objectName, key) +
                                   "' must be a JSON object");
  }
  return Result<yyjson_val *, std::string>::makeResult(value);
}

[[nodiscard]] Result<yyjson_val *, std::string>
optionalObjectField(yyjson_val *obj, const char *key,
                    std::string_view objectName) {
  if (obj == nullptr) {
    return Result<yyjson_val *, std::string>::makeResult(nullptr);
  }
  yyjson_val *value = yyjson_obj_get(obj, key);
  if (!value) {
    return Result<yyjson_val *, std::string>::makeResult(nullptr);
  }
  if (!yyjson_is_obj(value)) {
    return makeError<yyjson_val *>("Config field '" +
                                   fieldPath(objectName, key) +
                                   "' must be a JSON object");
  }
  return Result<yyjson_val *, std::string>::makeResult(value);
}

[[nodiscard]] Result<std::string, std::string>
requireStringField(yyjson_val *obj, const char *key,
                   std::string_view objectName) {
  yyjson_val *value = yyjson_obj_get(obj, key);
  if (!value) {
    return makeError<std::string>("Missing required config field '" +
                                  fieldPath(objectName, key) + "'");
  }
  if (!yyjson_is_str(value)) {
    return makeError<std::string>(
        "Config field '" + fieldPath(objectName, key) + "' must be a string");
  }
  const char *text = yyjson_get_str(value);
  if (!text || text[0] == '\0') {
    return makeError<std::string>(
        "Config field '" + fieldPath(objectName, key) + "' must not be empty");
  }
  return Result<std::string, std::string>::makeResult(std::string(text));
}

[[nodiscard]] Result<std::string, std::string>
stringFieldOrDefault(yyjson_val *obj, const char *key,
                     std::string_view objectName,
                     std::string_view defaultValue) {
  if (obj == nullptr) {
    return Result<std::string, std::string>::makeResult(
        std::string(defaultValue));
  }
  yyjson_val *value = yyjson_obj_get(obj, key);
  if (!value) {
    return Result<std::string, std::string>::makeResult(
        std::string(defaultValue));
  }
  if (!yyjson_is_str(value)) {
    return makeError<std::string>(
        "Config field '" + fieldPath(objectName, key) + "' must be a string");
  }
  const char *text = yyjson_get_str(value);
  if (!text || text[0] == '\0') {
    return makeError<std::string>(
        "Config field '" + fieldPath(objectName, key) + "' must not be empty");
  }
  return Result<std::string, std::string>::makeResult(std::string(text));
}

[[nodiscard]] Result<int32_t, std::string>
requirePositiveIntField(yyjson_val *obj, const char *key,
                        std::string_view objectName) {
  yyjson_val *value = yyjson_obj_get(obj, key);
  if (!value) {
    return makeError<int32_t>("Missing required config field '" +
                              fieldPath(objectName, key) + "'");
  }
  if (!yyjson_is_sint(value) && !yyjson_is_uint(value)) {
    return makeError<int32_t>("Config field '" + fieldPath(objectName, key) +
                              "' must be an integer");
  }

  if (yyjson_is_uint(value)) {
    const uint64_t raw = yyjson_get_uint(value);
    if (raw == 0 ||
        raw > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
      return makeError<int32_t>("Config field '" + fieldPath(objectName, key) +
                                "' must be in range [1, 2147483647]");
    }
    return Result<int32_t, std::string>::makeResult(static_cast<int32_t>(raw));
  }

  const int64_t raw = yyjson_get_sint(value);
  if (raw <= 0 ||
      raw > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
    return makeError<int32_t>("Config field '" + fieldPath(objectName, key) +
                              "' must be in range [1, 2147483647]");
  }
  return Result<int32_t, std::string>::makeResult(static_cast<int32_t>(raw));
}

[[nodiscard]] Result<WindowMode, std::string>
parseWindowMode(std::string_view modeValue) {
  if (modeValue == "windowed") {
    return Result<WindowMode, std::string>::makeResult(WindowMode::Windowed);
  }
  if (modeValue == "fullscreen") {
    return Result<WindowMode, std::string>::makeResult(WindowMode::Fullscreen);
  }
  if (modeValue == "borderless_fullscreen") {
    return Result<WindowMode, std::string>::makeResult(
        WindowMode::BorderlessFullscreen);
  }
  return makeError<WindowMode>(
      "Invalid window.mode '" + std::string(modeValue) +
      "'. Allowed values: windowed, fullscreen, borderless_fullscreen");
}

[[nodiscard]] Result<std::filesystem::path, std::string>
resolvePath(std::string_view rawPath, const std::filesystem::path &baseDir,
            std::string_view fieldName) {
  if (rawPath.empty()) {
    return makeError<std::filesystem::path>(
        "Config field '" + std::string(fieldName) + "' must not be empty");
  }

  std::filesystem::path resolved = std::filesystem::path(rawPath);
  if (!resolved.is_absolute()) {
    resolved = baseDir / resolved;
  }
  return Result<std::filesystem::path, std::string>::makeResult(
      normalizePath(resolved));
}

[[nodiscard]] Result<std::filesystem::path, std::string>
resolveDirectory(std::string_view rawPath, const std::filesystem::path &baseDir,
                 std::string_view fieldName) {
  auto resolvedResult = resolvePath(rawPath, baseDir, fieldName);
  if (resolvedResult.hasError()) {
    return resolvedResult;
  }
  const std::filesystem::path &resolved = resolvedResult.value();

  std::error_code ec;
  const bool exists = std::filesystem::exists(resolved, ec);
  if (ec || !exists) {
    return makeError<std::filesystem::path>(
        "Config field '" + std::string(fieldName) + "' resolves to '" +
        resolved.string() + "' but it does not exist");
  }
  if (!std::filesystem::is_directory(resolved, ec) || ec) {
    return makeError<std::filesystem::path>(
        "Config field '" + std::string(fieldName) + "' resolves to '" +
        resolved.string() + "' but it is not a directory");
  }
  return resolvedResult;
}

[[nodiscard]] Result<std::filesystem::path, std::string>
resolveFile(std::string_view rawPath, const std::filesystem::path &baseDir,
            std::string_view fieldName) {
  auto resolvedResult = resolvePath(rawPath, baseDir, fieldName);
  if (resolvedResult.hasError()) {
    return resolvedResult;
  }
  const std::filesystem::path &resolved = resolvedResult.value();

  std::error_code ec;
  const bool exists = std::filesystem::exists(resolved, ec);
  if (ec || !exists) {
    return makeError<std::filesystem::path>(
        "Config field '" + std::string(fieldName) + "' resolves to '" +
        resolved.string() + "' but it does not exist");
  }
  if (!std::filesystem::is_regular_file(resolved, ec) || ec) {
    return makeError<std::filesystem::path>(
        "Config field '" + std::string(fieldName) + "' resolves to '" +
        resolved.string() + "' but it is not a regular file");
  }
  return resolvedResult;
}

[[nodiscard]] Result<std::filesystem::path, std::string>
resolveShaderFileWithDefault(yyjson_val *sectionObj, const char *key,
                             std::string_view sectionFieldName,
                             std::string_view defaultRelativePath,
                             const std::filesystem::path &shadersRoot) {
  auto shaderPathText = stringFieldOrDefault(sectionObj, key, sectionFieldName,
                                             defaultRelativePath);
  if (shaderPathText.hasError()) {
    return makeError<std::filesystem::path>(shaderPathText.error());
  }

  const std::string resolvedFieldName = fieldPath(sectionFieldName, key);
  return resolveFile(shaderPathText.value(), shadersRoot, resolvedFieldName);
}

} // namespace

Result<RuntimeConfig, std::string>
loadRuntimeConfig(const std::filesystem::path &configPath) {
  NURI_PROFILER_FUNCTION();
  const std::filesystem::path normalizedConfigPath = normalizePath(configPath);

  std::error_code ec;
  if (!std::filesystem::exists(normalizedConfigPath, ec) || ec) {
    return makeError<RuntimeConfig>("App config file does not exist: '" +
                                    normalizedConfigPath.string() + "'");
  }
  if (!std::filesystem::is_regular_file(normalizedConfigPath, ec) || ec) {
    return makeError<RuntimeConfig>("App config path is not a file: '" +
                                    normalizedConfigPath.string() + "'");
  }

  auto textResult = readTextFile(normalizedConfigPath);
  if (textResult.hasError()) {
    return makeError<RuntimeConfig>(textResult.error());
  }

  std::string jsonText = std::move(textResult.value());
  yyjson_read_err parseError{};
  yyjson_doc *rawDoc = yyjson_read_opts(jsonText.data(), jsonText.size(), 0,
                                        nullptr, &parseError);
  if (!rawDoc) {
    const std::string message =
        parseError.msg != nullptr ? parseError.msg : "unknown parse error";
    return makeError<RuntimeConfig>(
        "Failed to parse app config '" + normalizedConfigPath.string() +
        "': " + std::to_string(parseError.pos) + ": " + message);
  }

  std::unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)> doc(rawDoc,
                                                              &yyjson_doc_free);
  yyjson_val *root = yyjson_doc_get_root(doc.get());
  if (!yyjson_is_obj(root)) {
    return makeError<RuntimeConfig>("App config root must be a JSON object");
  }

  auto rootKeysResult = validateUnknownKeys(root, "", kRootObjectKeys);
  if (rootKeysResult.hasError()) {
    return makeError<RuntimeConfig>(rootKeysResult.error());
  }

  auto windowObjResult = requireObjectField(root, "window", "");
  if (windowObjResult.hasError()) {
    return makeError<RuntimeConfig>(windowObjResult.error());
  }
  auto rootsObjResult = requireObjectField(root, "roots", "");
  if (rootsObjResult.hasError()) {
    return makeError<RuntimeConfig>(rootsObjResult.error());
  }
  auto shadersObjResult = optionalObjectField(root, "shaders", "");
  if (shadersObjResult.hasError()) {
    return makeError<RuntimeConfig>(shadersObjResult.error());
  }

  yyjson_val *windowObj = windowObjResult.value();
  yyjson_val *rootsObj = rootsObjResult.value();
  yyjson_val *shadersObj = shadersObjResult.value();

  {
    auto result = validateUnknownKeys(windowObj, "window", kWindowKeys);
    if (result.hasError()) {
      return makeError<RuntimeConfig>(result.error());
    }
  }
  {
    auto result = validateUnknownKeys(rootsObj, "roots", kRootsKeys);
    if (result.hasError()) {
      return makeError<RuntimeConfig>(result.error());
    }
  }
  if (shadersObj != nullptr) {
    auto result = validateUnknownKeys(shadersObj, "shaders", kShadersKeys);
    if (result.hasError()) {
      return makeError<RuntimeConfig>(result.error());
    }
  }

  auto debugGridObjResult =
      optionalObjectField(shadersObj, "debug_grid", "shaders");
  if (debugGridObjResult.hasError()) {
    return makeError<RuntimeConfig>(debugGridObjResult.error());
  }
  auto skyboxObjResult = optionalObjectField(shadersObj, "skybox", "shaders");
  if (skyboxObjResult.hasError()) {
    return makeError<RuntimeConfig>(skyboxObjResult.error());
  }
  auto opaqueObjResult = optionalObjectField(shadersObj, "opaque", "shaders");
  if (opaqueObjResult.hasError()) {
    return makeError<RuntimeConfig>(opaqueObjResult.error());
  }

  yyjson_val *debugGridObj = debugGridObjResult.value();
  yyjson_val *skyboxObj = skyboxObjResult.value();
  yyjson_val *opaqueObj = opaqueObjResult.value();

  if (debugGridObj != nullptr) {
    auto result = validateUnknownKeys(debugGridObj, "shaders.debug_grid",
                                      kDebugGridShaderKeys);
    if (result.hasError()) {
      return makeError<RuntimeConfig>(result.error());
    }
  }
  if (skyboxObj != nullptr) {
    auto result =
        validateUnknownKeys(skyboxObj, "shaders.skybox", kSkyboxShaderKeys);
    if (result.hasError()) {
      return makeError<RuntimeConfig>(result.error());
    }
  }
  if (opaqueObj != nullptr) {
    auto result =
        validateUnknownKeys(opaqueObj, "shaders.opaque", kOpaqueShaderKeys);
    if (result.hasError()) {
      return makeError<RuntimeConfig>(result.error());
    }
  }

  auto windowTitle = requireStringField(windowObj, "title", "window");
  if (windowTitle.hasError()) {
    return makeError<RuntimeConfig>(windowTitle.error());
  }
  auto windowWidth = requirePositiveIntField(windowObj, "width", "window");
  if (windowWidth.hasError()) {
    return makeError<RuntimeConfig>(windowWidth.error());
  }
  auto windowHeight = requirePositiveIntField(windowObj, "height", "window");
  if (windowHeight.hasError()) {
    return makeError<RuntimeConfig>(windowHeight.error());
  }
  auto windowModeText = requireStringField(windowObj, "mode", "window");
  if (windowModeText.hasError()) {
    return makeError<RuntimeConfig>(windowModeText.error());
  }
  auto windowMode = parseWindowMode(windowModeText.value());
  if (windowMode.hasError()) {
    return makeError<RuntimeConfig>(windowMode.error());
  }

  auto assetsRootText = requireStringField(rootsObj, "assets", "roots");
  if (assetsRootText.hasError()) {
    return makeError<RuntimeConfig>(assetsRootText.error());
  }
  auto shadersRootText = requireStringField(rootsObj, "shaders", "roots");
  if (shadersRootText.hasError()) {
    return makeError<RuntimeConfig>(shadersRootText.error());
  }
  auto modelsRootText = requireStringField(rootsObj, "models", "roots");
  if (modelsRootText.hasError()) {
    return makeError<RuntimeConfig>(modelsRootText.error());
  }
  auto texturesRootText = requireStringField(rootsObj, "textures", "roots");
  if (texturesRootText.hasError()) {
    return makeError<RuntimeConfig>(texturesRootText.error());
  }

  const std::filesystem::path configDir = normalizedConfigPath.parent_path();
  auto assetsRoot =
      resolveDirectory(assetsRootText.value(), configDir, "roots.assets");
  if (assetsRoot.hasError()) {
    return makeError<RuntimeConfig>(assetsRoot.error());
  }
  auto shadersRoot =
      resolveDirectory(shadersRootText.value(), configDir, "roots.shaders");
  if (shadersRoot.hasError()) {
    return makeError<RuntimeConfig>(shadersRoot.error());
  }
  auto modelsRoot =
      resolveDirectory(modelsRootText.value(), configDir, "roots.models");
  if (modelsRoot.hasError()) {
    return makeError<RuntimeConfig>(modelsRoot.error());
  }
  auto texturesRoot =
      resolveDirectory(texturesRootText.value(), configDir, "roots.textures");
  if (texturesRoot.hasError()) {
    return makeError<RuntimeConfig>(texturesRoot.error());
  }

  auto debugGridVertexPath = resolveShaderFileWithDefault(
      debugGridObj, "vertex", "shaders.debug_grid",
      kDefaultDebugGridVertexShader, shadersRoot.value());
  if (debugGridVertexPath.hasError()) {
    return makeError<RuntimeConfig>(debugGridVertexPath.error());
  }
  auto debugGridFragmentPath = resolveShaderFileWithDefault(
      debugGridObj, "fragment", "shaders.debug_grid",
      kDefaultDebugGridFragmentShader, shadersRoot.value());
  if (debugGridFragmentPath.hasError()) {
    return makeError<RuntimeConfig>(debugGridFragmentPath.error());
  }
  auto skyboxVertexPath = resolveShaderFileWithDefault(
      skyboxObj, "vertex", "shaders.skybox", kDefaultSkyboxVertexShader,
      shadersRoot.value());
  if (skyboxVertexPath.hasError()) {
    return makeError<RuntimeConfig>(skyboxVertexPath.error());
  }
  auto skyboxFragmentPath = resolveShaderFileWithDefault(
      skyboxObj, "fragment", "shaders.skybox", kDefaultSkyboxFragmentShader,
      shadersRoot.value());
  if (skyboxFragmentPath.hasError()) {
    return makeError<RuntimeConfig>(skyboxFragmentPath.error());
  }

  auto meshVertexPath = resolveShaderFileWithDefault(
      opaqueObj, "mesh_vertex", "shaders.opaque",
      kDefaultOpaqueMeshVertexShader, shadersRoot.value());
  if (meshVertexPath.hasError()) {
    return makeError<RuntimeConfig>(meshVertexPath.error());
  }
  auto meshFragmentPath = resolveShaderFileWithDefault(
      opaqueObj, "mesh_fragment", "shaders.opaque",
      kDefaultOpaqueMeshFragmentShader, shadersRoot.value());
  if (meshFragmentPath.hasError()) {
    return makeError<RuntimeConfig>(meshFragmentPath.error());
  }
  auto pickFragmentPath = resolveShaderFileWithDefault(
      opaqueObj, "pick_fragment", "shaders.opaque",
      kDefaultOpaquePickFragmentShader, shadersRoot.value());
  if (pickFragmentPath.hasError()) {
    return makeError<RuntimeConfig>(pickFragmentPath.error());
  }
  auto computeInstancesPath = resolveShaderFileWithDefault(
      opaqueObj, "compute_instances", "shaders.opaque",
      kDefaultOpaqueComputeShader, shadersRoot.value());
  if (computeInstancesPath.hasError()) {
    return makeError<RuntimeConfig>(computeInstancesPath.error());
  }
  auto tessVertexPath = resolveShaderFileWithDefault(
      opaqueObj, "tess_vertex", "shaders.opaque",
      kDefaultOpaqueTessVertexShader, shadersRoot.value());
  if (tessVertexPath.hasError()) {
    return makeError<RuntimeConfig>(tessVertexPath.error());
  }
  auto tessControlPath = resolveShaderFileWithDefault(
      opaqueObj, "tess_control", "shaders.opaque",
      kDefaultOpaqueTessControlShader, shadersRoot.value());
  if (tessControlPath.hasError()) {
    return makeError<RuntimeConfig>(tessControlPath.error());
  }
  auto tessEvalPath = resolveShaderFileWithDefault(
      opaqueObj, "tess_eval", "shaders.opaque", kDefaultOpaqueTessEvalShader,
      shadersRoot.value());
  if (tessEvalPath.hasError()) {
    return makeError<RuntimeConfig>(tessEvalPath.error());
  }
  auto overlayGeometryPath = resolveShaderFileWithDefault(
      opaqueObj, "overlay_geometry", "shaders.opaque",
      kDefaultOpaqueOverlayGeometryShader, shadersRoot.value());
  if (overlayGeometryPath.hasError()) {
    return makeError<RuntimeConfig>(overlayGeometryPath.error());
  }
  auto overlayFragmentPath = resolveShaderFileWithDefault(
      opaqueObj, "overlay_fragment", "shaders.opaque",
      kDefaultOpaqueOverlayFragmentShader, shadersRoot.value());
  if (overlayFragmentPath.hasError()) {
    return makeError<RuntimeConfig>(overlayFragmentPath.error());
  }

  RuntimeConfig config{};
  config.sourcePath = normalizedConfigPath;
  config.window = RuntimeWindowConfig{
      .title = std::move(windowTitle.value()),
      .width = windowWidth.value(),
      .height = windowHeight.value(),
      .mode = windowMode.value(),
  };
  config.roots = RuntimeRootsConfig{
      .assets = assetsRoot.value(),
      .shaders = shadersRoot.value(),
      .models = modelsRoot.value(),
      .textures = texturesRoot.value(),
  };
  config.shaders = RuntimeShaderConfig{
      .debugGrid =
          RuntimeDebugShaderConfig{
              .vertex = debugGridVertexPath.value(),
              .fragment = debugGridFragmentPath.value(),
          },
      .skybox =
          RuntimeSkyboxShaderConfig{
              .vertex = skyboxVertexPath.value(),
              .fragment = skyboxFragmentPath.value(),
          },
      .opaque =
          RuntimeOpaqueShaderConfig{
              .meshVertex = meshVertexPath.value(),
              .meshFragment = meshFragmentPath.value(),
              .pickFragment = pickFragmentPath.value(),
              .computeInstances = computeInstancesPath.value(),
              .tessVertex = tessVertexPath.value(),
              .tessControl = tessControlPath.value(),
              .tessEval = tessEvalPath.value(),
              .overlayGeometry = overlayGeometryPath.value(),
              .overlayFragment = overlayFragmentPath.value(),
          },
  };

  return Result<RuntimeConfig, std::string>::makeResult(std::move(config));
}

Result<RuntimeConfig, std::string> loadRuntimeConfig() {
  return loadRuntimeConfig(std::filesystem::path{kDefaultConfigPath});
}

Result<RuntimeConfig, std::string> loadRuntimeConfigFromEnvOrDefault() {
#if defined(_WIN32)
  struct CFreeDeleter {
    void operator()(char *value) const noexcept { std::free(value); }
  };

  char *envConfigPathRaw = nullptr;
  size_t envConfigPathSize = 0;
  const int envReadError =
      _dupenv_s(&envConfigPathRaw, &envConfigPathSize, kAppConfigEnvVarCStr);
  std::unique_ptr<char, CFreeDeleter> envConfigPath(envConfigPathRaw);
  if (envReadError != 0) {
    return makeError<RuntimeConfig>("Failed to read environment variable '" +
                                    std::string(kAppConfigEnvVar) +
                                    "' (error " + std::to_string(envReadError) +
                                    ")");
  }

  if (envConfigPath != nullptr && envConfigPath.get()[0] != '\0') {
    return loadRuntimeConfig(std::filesystem::path{envConfigPath.get()});
  }
#else
  const char *envConfigPath = std::getenv(kAppConfigEnvVarCStr);
  if (envConfigPath != nullptr && envConfigPath[0] != '\0') {
    return loadRuntimeConfig(std::filesystem::path{envConfigPath});
  }
#endif
  return loadRuntimeConfig();
}

} // namespace nuri
