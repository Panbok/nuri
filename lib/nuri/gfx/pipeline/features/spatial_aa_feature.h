#pragma once
#include "nuri/core/runtime_config.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/pipeline/render_pipeline.h"
#include "nuri/gfx/shader.h"
#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>
namespace nuri {

class RenderPipeline;

enum class SpatialAAPlacement : uint8_t {
  SceneColor = 0,
  PostTransparent = 1,
};

class NURI_API SpatialAAPass final {
public:
  explicit SpatialAAPass(
      GPUDevice &gpu, RuntimeCompositeConfig config,
      SpatialAAPlacement placement = SpatialAAPlacement::SceneColor);
  ~SpatialAAPass();
  [[nodiscard]] bool isEnabled(const FrameBuildContext &ctx) const;
  Result<bool, std::string> prepare(FrameBuildContext &ctx);
  Result<bool, std::string> build(FrameBuildContext &ctx);

private:
  GPUDevice &gpu_;
  RuntimeCompositeConfig config_{};
  SpatialAAPlacement placement_;
  ShaderHandle vertexShader_{};
  std::array<ShaderHandle, 3> fragmentShaders_{};
  std::array<RenderPipelineHandle, 3> pipelines_{};
  std::array<SamplerHandle, 2> samplers_{};
  std::array<TextureHandle, 2> luts_{};
  std::array<std::vector<TextureHandle>, 3> scratchTextures_{};
  std::string initializationError_{};
  uint32_t scratchWidth_ = 0u;
  uint32_t scratchHeight_ = 0u;
  uint32_t scratchRingCount_ = 0u;
  Format outputScratchFormat_ = Format::Count;
  Result<bool, std::string> initialize();
  Result<bool, std::string> ensureScratchTextures(FrameBuildContext &ctx);
  void destroyResources();
};

NURI_API void registerSpatialAAStage(
    RenderPipeline &pipeline, GPUDevice &gpu, RuntimeCompositeConfig config,
    SpatialAAPlacement placement = SpatialAAPlacement::SceneColor);

} // namespace nuri
