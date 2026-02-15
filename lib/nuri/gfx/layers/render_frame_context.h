#pragma once

#include "nuri/gfx/gpu_types.h"

#include <cstdint>

#include <glm/glm.hpp>

namespace nuri {

class RenderScene;

struct RenderSettings {
  bool drawModelBounds = true;
  bool drawSkybox = true;
  bool drawOpaque = true;
  bool drawDebug = false;
  bool enableMeshLod = true;
  int32_t forcedMeshLod = -1;
  glm::vec3 meshLodDistanceThresholds{8.0f, 16.0f, 32.0f};
};

struct CameraFrameState {
  glm::mat4 view{1.0f};
  glm::mat4 proj{1.0f};
  glm::vec4 cameraPos{0.0f, 0.0f, 0.0f, 1.0f};
  float aspectRatio = 1.0f;
};

struct RenderFrameContext {
  const RenderScene *scene = nullptr;
  CameraFrameState camera{};
  RenderSettings *settings = nullptr;
  TextureHandle sharedDepthTexture{};
  double timeSeconds = 0.0;
  uint64_t frameIndex = 0;
};

} // namespace nuri
