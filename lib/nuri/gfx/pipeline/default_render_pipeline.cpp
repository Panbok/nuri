#include "nuri/gfx/pipeline/default_render_pipeline.h"
#include "nuri/gfx/ddgi/ddgi_feature.h"
#include "nuri/gfx/frame/render_capture.h"
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
#include "nuri/pch.h"
namespace nuri {
namespace {
bool needsMsaaResolve(const FrameBuildContext &ctx) {
  const AntiAliasingFrameMetrics &metrics = ctx.frame.metrics.antiAliasing;
  return ctx.frame.presentationAA.coverage != CoverageMode::Sample1 &&
         !(metrics.msaaColorResolveTargetBound &&
           metrics.msaaDepthResolveTargetBound);
}
Result<bool, std::string> appendMsaaResolve(FrameBuildContext &ctx,
                                            GPUDevice &gpu) {
  const auto import = [&ctx](RenderGraphTextureId id, TextureHandle handle,
                             std::string_view name) {
    return nuri::isValid(id) ? id
                             : ctx.graph.importTexture(handle, name).value();
  };
  ctx.shared.msaaSceneColorGraphTexture =
      import(ctx.shared.msaaSceneColorGraphTexture,
             ctx.shared.msaaSceneColorTexture, "msaa_resolve_scene_color");
  ctx.shared.msaaSceneDepthGraphTexture =
      import(ctx.shared.msaaSceneDepthGraphTexture,
             ctx.shared.msaaSceneDepthTexture, "msaa_resolve_scene_depth");
  const RenderGraphTextureId colorTarget =
      ctx.graph
          .importTexture(ctx.shared.sceneColorTexture,
                         "msaa_resolve_scene_color_target")
          .value();
  const RenderGraphTextureId depthTarget =
      ctx.graph
          .importTexture(ctx.shared.sceneDepthTexture,
                         "msaa_resolve_scene_depth_target")
          .value();
  [[maybe_unused]] const RenderGraphPassId resolvePass =
      ctx.graph
          .addGraphicsPass(RenderGraphGraphicsPassDesc{
              .color = {.loadOp = LoadOp::Load,
                        .storeOp = StoreOp::MsaaResolve,
                        .resolveMode = ResolveMode::Average},
              .colorTexture = ctx.shared.msaaSceneColorGraphTexture,
              .colorResolveTexture = colorTarget,
              .depth = {.loadOp = LoadOp::Load,
                        .storeOp = StoreOp::MsaaResolve,
                        .resolveMode = ResolveMode::Min},
              .depthTexture = ctx.shared.msaaSceneDepthGraphTexture,
              .depthResolveTexture = depthTarget,
              .gpuTimingScope = GpuTimingScope::MsaaResolve,
              .debugLabel = "MSAA Resolve Pass",
              .debugColor = 0xff44ccff,
              .markImplicitOutputSideEffect = true,
          })
          .value();
  ctx.shared.sceneColorGraphTexture = colorTarget;
  ctx.shared.sceneDepthGraphTexture = depthTarget;
  AntiAliasingFrameMetrics &metrics = ctx.frame.metrics.antiAliasing;
  metrics.msaaResolvePassCount = 1u;
  metrics.msaaColorResolveCount = 1u;
  metrics.msaaDepthResolveCount = 1u;
  metrics.msaaResolvedSampleCount =
      coverageSampleCount(ctx.frame.presentationAA.coverage);
  metrics.msaaResolvePlacement = MsaaResolvePlacement::ExplicitPass;
  metrics.msaaColorGraphPublished = true;
  metrics.msaaDepthGraphPublished = true;
  metrics.msaaColorResolveTargetBound = true;
  metrics.msaaDepthResolveTargetBound = true;
  publishRequestedCapture(ctx.frame, gpu, "scene_color_hdr",
                          ctx.shared.sceneColorTexture,
                          RenderCaptureValueKind::LinearHdrColor,
                          RenderCaptureLifetimeClass::FrameSharedRingTexture,
                          "linear_hdr", "hdr_color", "MSAA Resolve Pass");
  publishRequestedCapture(ctx.frame, gpu, "scene_depth",
                          ctx.shared.sceneDepthTexture,
                          RenderCaptureValueKind::Depth,
                          RenderCaptureLifetimeClass::FrameSharedRingTexture,
                          "linear_depth", "depth", "MSAA Resolve Pass");
  metrics.msaaResolveBandwidthEstimateBytes =
      metrics.msaaResolveReadEstimateBytes;
  if (hasGpuTimingScope(ctx.frame.gpuTiming, GpuTimingScope::MsaaResolve)) {
    metrics.msaaResolveGpuTimeMs = ctx.frame.gpuTiming.msaaResolveTimeMs;
    metrics.msaaResolveGpuTimingSourceFrameIndex =
        ctx.frame.gpuTiming.msaaResolveSourceFrameIndex;
    metrics.msaaResolveGpuTimingAvailable = 1u;
  }
  return Result<bool, std::string>::makeResult(true);
}
void registerMsaaResolveStage(RenderPipeline &pipeline, GPUDevice &gpu) {
  pipeline.addStage(PipelineStageDesc{
      .componentName = "MsaaResolveFeature",
      .name = "MsaaResolvePass",
      .state = &gpu,
      .enabled =
          [](const void *, const FrameBuildContext &ctx) {
            return needsMsaaResolve(ctx);
          },
      .build =
          [](void *state, FrameBuildContext &ctx) {
            return appendMsaaResolve(ctx, *static_cast<GPUDevice *>(state));
          },
  });
}
} // namespace

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
