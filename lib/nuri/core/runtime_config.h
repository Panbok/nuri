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

struct NURI_API RuntimeRasterShaderConfig {
  std::filesystem::path vertex;
  std::filesystem::path fragment;
};

struct NURI_API RuntimeOpaqueShaderConfig {
  std::filesystem::path shaderBasePath;
  std::filesystem::path meshVertex;
  std::filesystem::path meshFragment;
  std::filesystem::path meshletTask;
  std::filesystem::path meshletMesh;
  std::filesystem::path meshletFragment;
  std::filesystem::path meshletDepthFragment;
  std::filesystem::path meshletDepthAlphaFragment;
  std::filesystem::path pickFragment;
  std::filesystem::path shadowInspectFragment;
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

struct NURI_API RuntimeCompositeConfig {
  std::filesystem::path shaderBasePath;
  std::filesystem::path fullscreenVertex;
  std::filesystem::path sceneCopyFragment;
  std::filesystem::path presentFragment;
  std::filesystem::path hdrLuminanceReduceFragment;
  std::filesystem::path hdrExposureAdaptFragment;
  std::filesystem::path hdrBloomFragment;
  std::filesystem::path hdrBloomCompositeFragment;
  std::filesystem::path aces2SdrLut;
  std::filesystem::path agxLut;
};

struct NURI_API RuntimeDDGIShaderConfig {
  uint64_t persistentMemoryLimitBytes = 256ull * 1024ull * 1024ull;
  uint64_t peakMemoryLimitBytes = 512ull * 1024ull * 1024ull;
  std::filesystem::path shaderBasePath;
  std::filesystem::path decodePositions;
  std::filesystem::path prepareDynamicVertices;
  std::filesystem::path trace;
  std::filesystem::path traceInspect;
  std::filesystem::path blendIrradiance;
  std::filesystem::path blendDistance;
  std::filesystem::path updateProbeState;
  std::filesystem::path opaqueSurfaceCache;
  std::filesystem::path probeDebugVertex;
  std::filesystem::path probeDebugFragment;
  std::filesystem::path rayDebugVertex;
  std::filesystem::path rayDebugFragment;
};

struct NURI_API RuntimeShaderConfig {
  RuntimeRasterShaderConfig debugGrid;
  RuntimeRasterShaderConfig skybox;
  RuntimeOpaqueShaderConfig opaque;
  RuntimeCompositeConfig composite;
  RuntimeTextMtsdfShaderConfig textMtsdf;
  RuntimeDDGIShaderConfig ddgi;
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
