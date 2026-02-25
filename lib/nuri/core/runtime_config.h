#pragma once

#include "nuri/core/result.h"
#include "nuri/core/window.h"
#include "nuri/defines.h"

#include <cstdint>
#include <filesystem>
#include <string>

namespace nuri {

struct NURI_API RuntimeWindowConfig {
  std::string title;
  int32_t width = 0;
  int32_t height = 0;
  WindowMode mode = WindowMode::Windowed;
};

struct NURI_API RuntimeRootsConfig {
  std::filesystem::path assets;
  std::filesystem::path shaders;
  std::filesystem::path models;
  std::filesystem::path textures;
  std::filesystem::path fonts;
};

struct NURI_API RuntimeDebugShaderConfig {
  std::filesystem::path vertex;
  std::filesystem::path fragment;
};

struct NURI_API RuntimeSkyboxShaderConfig {
  std::filesystem::path vertex;
  std::filesystem::path fragment;
};

struct NURI_API RuntimeOpaqueShaderConfig {
  std::filesystem::path meshVertex;
  std::filesystem::path meshFragment;
  std::filesystem::path pickFragment;
  std::filesystem::path computeInstances;
  std::filesystem::path tessVertex;
  std::filesystem::path tessControl;
  std::filesystem::path tessEval;
  std::filesystem::path overlayGeometry;
  std::filesystem::path overlayFragment;
};

struct NURI_API RuntimeTextMtsdfShaderConfig {
  std::filesystem::path uiVertex;
  std::filesystem::path uiFragment;
  std::filesystem::path worldVertex;
  std::filesystem::path worldFragment;
};

struct NURI_API RuntimeShaderConfig {
  RuntimeDebugShaderConfig debugGrid;
  RuntimeSkyboxShaderConfig skybox;
  RuntimeOpaqueShaderConfig opaque;
  RuntimeTextMtsdfShaderConfig textMtsdf;
};

struct NURI_API RuntimeConfig {
  std::filesystem::path sourcePath;
  RuntimeWindowConfig window;
  RuntimeRootsConfig roots;
  RuntimeShaderConfig shaders;
};

[[nodiscard]] NURI_API Result<RuntimeConfig, std::string>
loadRuntimeConfig(const std::filesystem::path &configPath);
[[nodiscard]] NURI_API Result<RuntimeConfig, std::string> loadRuntimeConfig();
[[nodiscard]] NURI_API Result<RuntimeConfig, std::string>
loadRuntimeConfigFromEnvOrDefault();

} // namespace nuri
