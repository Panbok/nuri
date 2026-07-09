#include "nuri/pch.h"

#include "nuri/gfx/pipeline/default_render_pipeline.h"

#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/pipeline/features/composite_feature.h"
#include "nuri/gfx/pipeline/features/debug_feature.h"
#include "nuri/gfx/pipeline/features/gtao_feature.h"
#include "nuri/gfx/pipeline/features/msaa_resolve_feature.h"
#include "nuri/gfx/pipeline/features/opaque_feature.h"
#include "nuri/gfx/pipeline/features/shadow_feature.h"
#include "nuri/gfx/pipeline/features/skybox_feature.h"
#include "nuri/gfx/pipeline/features/spatial_aa_feature.h"
#include "nuri/gfx/pipeline/features/temporal_aa_feature.h"
#include "nuri/gfx/pipeline/features/transmission_feature.h"
#include "nuri/gfx/pipeline/features/transparent_feature.h"
#include "nuri/gfx/pipeline/providers/frame_composition_provider.h"
#include "nuri/gfx/pipeline/providers/material_table_gpu_provider.h"
#include "nuri/gfx/pipeline/providers/scene_lighting_provider.h"
#include "nuri/gfx/pipeline/render_pipeline.h"

namespace nuri {

Result<bool, std::string>
registerDefaultRenderPipeline(RenderPipeline &pipeline, GPUDevice &gpu,
                              const RuntimeShaderConfig &shaderConfig,
                              std::pmr::memory_resource *memory) {
  pipeline.addProvider(std::make_unique<MaterialTableGpuProvider>(gpu));
  pipeline.addProvider(std::make_unique<FrameCompositionProvider>(gpu, memory));
  pipeline.addProvider(std::make_unique<SceneLightingProvider>(gpu));
  pipeline.addFeature(
      std::make_unique<ShadowFeature>(gpu, shaderConfig.opaque, memory));
  auto opaqueRenderer =
      makeSharedOpaqueRenderer(gpu, shaderConfig.opaque, memory);
  pipeline.addFeature(std::make_unique<OpaquePrepassFeature>(opaqueRenderer));
  pipeline.addFeature(std::make_unique<GTAOFeature>(gpu, shaderConfig.opaque));
  pipeline.addFeature(std::make_unique<OpaqueMainFeature>(opaqueRenderer));
  pipeline.addFeature(
      std::make_unique<SkyboxFeature>(gpu, shaderConfig.skybox));
  pipeline.addFeature(std::make_unique<MsaaResolveFeature>(gpu));
  pipeline.addFeature(
      std::make_unique<TemporalAAFeature>(gpu, shaderConfig.composite));
  pipeline.addFeature(
      std::make_unique<SpatialAAFeature>(gpu, shaderConfig.composite));
  pipeline.addFeature(
      std::make_unique<FrameCompositionFeature>(gpu, shaderConfig.composite));
  pipeline.addFeature(
      std::make_unique<TransmissionFeature>(gpu, shaderConfig.opaque, memory));
  pipeline.addFeature(
      std::make_unique<TransparentFeature>(gpu, shaderConfig.opaque, memory));
  pipeline.addFeature(std::make_unique<SpatialAAFeature>(
      gpu, shaderConfig.composite, SpatialAAPlacement::PostTransparent));
  pipeline.addFeature(
      std::make_unique<HDRPostProcessFeature>(gpu, shaderConfig.composite));
  pipeline.addFeature(
      std::make_unique<DebugFeature>(gpu, shaderConfig.debugGrid, memory));
  pipeline.addFeature(
      std::make_unique<FramePresentFeature>(gpu, shaderConfig.composite));
  return Result<bool, std::string>::makeResult(true);
}

} // namespace nuri
