#pragma once

#include "nuri/gfx/gpu_types.h"

#include <cstdint>

#include <glm/glm.hpp>

namespace nuri {

class RenderScene;

enum class OpaqueDebugVisualization : uint8_t {
  None = 0,
  WireOverlay = 1,
  TessPatchEdgesHeatmap = 2,
};

struct RenderSettings {
  struct SkyboxSettings {
    bool enabled = true;
  };

  struct OpaqueSettings {
    bool enabled = true;
    OpaqueDebugVisualization debugVisualization =
        OpaqueDebugVisualization::None;
    bool enableInstanceCompute = true;
    bool enableMeshLod = true;
    int32_t forcedMeshLod = -1;
    glm::vec3 meshLodDistanceThresholds{8.0f, 16.0f, 32.0f};
    bool enableInstanceAnimation = true;
    bool enableTessellation = false;
    float tessNearDistance = 1.0f;
    float tessFarDistance = 8.0f;
    float tessMinFactor = 1.0f;
    float tessMaxFactor = 6.0f;
    // 0 means "no cap".
    uint32_t tessMaxInstances = 256;
  };

  struct DebugSettings {
    bool enabled = false;
    bool modelBounds = false;
    bool grid = false;
  };

  SkyboxSettings skybox{};
  OpaqueSettings opaque{};
  DebugSettings debug{};
};

struct CameraFrameState {
  glm::mat4 view{1.0f};
  glm::mat4 proj{1.0f};
  glm::vec4 cameraPos{0.0f, 0.0f, 0.0f, 1.0f};
  float aspectRatio = 1.0f;
};

struct OpaqueFrameMetrics {
  uint32_t totalInstances = 0;
  uint32_t visibleInstances = 0;
  uint32_t instancedDraws = 0;
  uint32_t indirectDrawCalls = 0;
  uint32_t indirectCommands = 0;
  uint32_t tessellatedDraws = 0;
  uint32_t tessellatedInstances = 0;
  uint32_t debugOverlayDraws = 0;
  uint32_t debugOverlayFallbackDraws = 0;
  uint32_t debugPatchHeatmapDraws = 0;
  uint32_t computeDispatches = 0;
  uint32_t computeDispatchX = 0;
};

struct RenderFrameMetrics {
  OpaqueFrameMetrics opaque{};
};

struct RenderFrameContext {
  const RenderScene *scene = nullptr;
  CameraFrameState camera{};
  RenderSettings *settings = nullptr;
  RenderFrameMetrics metrics{};
  TextureHandle sharedDepthTexture{};
  double timeSeconds = 0.0;
  uint64_t frameIndex = 0;
};

} // namespace nuri
