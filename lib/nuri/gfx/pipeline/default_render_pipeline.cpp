#include "nuri/gfx/pipeline/default_render_pipeline.h"
#include "nuri/gfx/ddgi/ddgi_feature.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/pipeline/features/composite_feature.h"
#include "nuri/gfx/pipeline/features/gtao_feature.h"
#include "nuri/gfx/pipeline/features/reference_taa_feature.h"
#include "nuri/gfx/pipeline/features/skybox_feature.h"
#include "nuri/gfx/pipeline/features/spatial_aa_feature.h"
#include "nuri/gfx/pipeline/features/temporal_aa_feature.h"
#include "nuri/gfx/pipeline/providers/frame_composition_provider.h"
#include "nuri/gfx/pipeline/providers/material_table_gpu_provider.h"
#include "nuri/gfx/pipeline/providers/scene_lighting_provider.h"
#include "nuri/gfx/pipeline/render_pipeline.h"
#include "nuri/gfx/ray_tracing/ray_tracing_scene.h"
#include "nuri/gfx/renderers/debug_renderer.h"
#include "nuri/gfx/renderers/opaque_renderer.h"
#include "nuri/gfx/renderers/scene_draw_database.h"
#include "nuri/gfx/renderers/shadow_renderer.h"
#include "nuri/gfx/renderers/transmission_renderer.h"
#include "nuri/gfx/renderers/transparent_renderer.h"
namespace nuri {
Result<bool, std::string>
registerDefaultRenderPipeline(RenderPipeline &pipeline, GPUDevice &gpu,
                              const RuntimeShaderConfig &shaderConfig,
                              std::pmr::memory_resource *memory) {
  pipeline.addProvider(std::make_unique<MaterialTableGpuProvider>(gpu));
  auto *drawDatabase =
      pipeline.addProvider(std::make_unique<SceneDrawDatabase>(gpu, memory));
  pipeline.addProvider(
      std::make_unique<RayTracingScene>(gpu, shaderConfig.ddgi, memory));
  auto *ddgiFeature = pipeline.addProvider(
      std::make_unique<DDGIFeature>(gpu, shaderConfig.ddgi, memory));
  pipeline.addProvider(std::make_unique<FrameCompositionProvider>(gpu, memory));
  pipeline.addProvider(std::make_unique<SceneLightingProvider>(gpu));
  registerShadowStage(pipeline, gpu, shaderConfig.opaque, memory, drawDatabase);
  OpaqueRenderer *opaqueRenderer = registerOpaquePrepassStages(
      pipeline, gpu, shaderConfig.opaque, memory, drawDatabase);
  registerTemporalInputStages(pipeline, gpu, shaderConfig.composite);
  registerGTAOStage(pipeline, gpu, shaderConfig.opaque);
  pipeline.addStage(PipelineStageDesc{
      .componentName = "DDGIFeature",
      .name = "DDGIOpaqueSurfaceCache",
      .state = ddgiFeature,
      .enabled =
          [](const void *state, const FrameBuildContext &ctx) {
            return static_cast<const DDGIFeature *>(state)
                ->opaqueSurfaceCacheActive(ctx);
          },
      .build =
          [](void *state, FrameBuildContext &ctx) {
            return static_cast<DDGIFeature *>(state)->buildOpaqueSurfaceCache(
                ctx);
          },
  });
  registerOpaqueMainStage(pipeline, *opaqueRenderer);
  registerSkyboxStage(pipeline, gpu, shaderConfig.skybox);
  registerMsaaResolveStage(pipeline, gpu);
  registerTemporalAAStages(pipeline, gpu, shaderConfig.composite);
  registerSpatialAAStage(pipeline, gpu, shaderConfig.composite);
  registerFrameCompositionStages(pipeline, gpu, shaderConfig.composite);
  registerTransmissionStage(pipeline, gpu, shaderConfig.opaque, memory,
                            drawDatabase);
  registerTransparentStages(pipeline, gpu, shaderConfig.opaque, memory,
                            drawDatabase);
  registerReferenceTAAStage(pipeline, gpu, shaderConfig.composite);
  registerSpatialAAStage(pipeline, gpu, shaderConfig.composite,
                         SpatialAAPlacement::PostTransparent);
  registerHDRPostProcessStages(pipeline, gpu, shaderConfig.composite);
  registerDebugStages(pipeline, gpu,
                      DebugRendererConfig{.grid = shaderConfig.debugGrid,
                                          .ddgi = shaderConfig.ddgi},
                      memory);
  registerFramePresentStage(pipeline, gpu, shaderConfig.composite);
  return Result<bool, std::string>::makeResult(true);
}

} // namespace nuri
