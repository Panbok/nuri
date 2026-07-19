#include "nuri/gfx/pipeline/render_pipeline.h"
#include "nuri/core/profiling.h"
#include "nuri/gfx/frame/presentation_aa_plan.h"
#include "nuri/pch.h"
#include "nuri/resources/gpu/resource_manager.h"
namespace nuri {

RenderPipeline::RenderPipeline(std::pmr::memory_resource *memory)
    : components_(memory ? memory : std::pmr::get_default_resource()),
      stages_(memory ? memory : std::pmr::get_default_resource()) {}

void RenderPipeline::addStage(PipelineStageDesc stage) {
  auto position = stage.terminal ? stages_.end()
                                 : std::find_if(stages_.begin(), stages_.end(),
                                                [](const Stage &entry) {
                                                  return entry.desc.terminal;
                                                });
  stages_.insert(position, Stage{.desc = stage});
}

std::optional<RenderPipelinePassInfo>
RenderPipeline::passInfo(size_t index) const noexcept {
  if (index >= stages_.size()) {
    return std::nullopt;
  }
  const Stage &entry = stages_[index];
  return RenderPipelinePassInfo{
      .index = index,
      .featureName = entry.desc.componentName,
      .passName = entry.desc.name,
      .enabled = entry.enabled,
  };
}

std::optional<bool> RenderPipeline::isPassEnabled(size_t index) const noexcept {
  if (index >= stages_.size()) {
    return std::nullopt;
  }
  return stages_[index].enabled;
}

bool RenderPipeline::setPassEnabled(size_t index, bool enabled) noexcept {
  if (index >= stages_.size()) {
    return false;
  }
  stages_[index].enabled = enabled;
  return true;
}

Result<bool, std::string>
RenderPipeline::buildRenderGraph(RenderFrameContext &frame,
                                 ResourceManager &resources,
                                 RenderGraphBuilder &graph) {
  NURI_PROFILER_FUNCTION();
  const PresentationAAGpuCapabilities gpuCapabilities =
      resources.gpuMultisampleCapabilities();
  AntiAliasingFrameMetrics &aaMetrics = frame.metrics.antiAliasing;
  aaMetrics.msaaSample4ColorSupported = gpuCapabilities.sample4Color;
  aaMetrics.msaaSample4DepthSupported = gpuCapabilities.sample4Depth;
  aaMetrics.msaaDepthResolveMinSupported = gpuCapabilities.depthResolveMin;
  aaMetrics.msaaAlphaToCoverageSupported = gpuCapabilities.alphaToCoverage;
  aaMetrics.msaaSampleRateShadingSupported = gpuCapabilities.sampleRateShading;
  aaMetrics.msaaUnsupportedReason =
      sanitizeAntiAliasingMode(
          renderSettingsOrDefault(frame).antiAliasing.mode) ==
              AntiAliasingMode::MSAA4x
          ? msaa4xUnsupportedReason(gpuCapabilities)
          : PresentationAAUnsupportedReason::None;
  auto presentationPlan = buildPresentationAAPlan(
      renderSettingsOrDefault(frame), {}, gpuCapabilities);
  if (presentationPlan.hasError()) {
    return Result<bool, std::string>::makeError(presentationPlan.error());
  }
  frame.presentationAA = presentationPlan.value();
  aaMetrics.msaaAlphaCoveragePolicy = frame.presentationAA.alphaCoverage;
  aaMetrics.msaaTransparencyPolicy = frame.presentationAA.transparency;
  FrameBuildContext ctx{
      .frame = frame,
      .graph = graph,
      .resources = resources,
      .shared = frame.sharedResources,
  };
  {
    NURI_PROFILER_ZONE("RenderPipeline.prepare_components",
                       NURI_PROFILER_COLOR_CMD_COPY);
    constexpr std::array phases{&PipelineComponentDesc::publish,
                                &PipelineComponentDesc::provide,
                                &PipelineComponentDesc::prepare};
    for (auto phase : phases) {
      for (Component &component : components_) {
        PipelineFrameCallback callback = component.desc.*phase;
        if (!callback) {
          continue;
        }
        auto result = callback(component.owner.get(), ctx);
        if (result.hasError()) {
          return Result<bool, std::string>::makeError(result.error());
        }
      }
    }
    NURI_PROFILER_ZONE_END();
  }
  {
    NURI_PROFILER_ZONE("RenderPipeline.build_stages",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    for (const Stage &entry : stages_) {
      const PipelineStageDesc &stage = entry.desc;
      if (!entry.enabled ||
          (stage.enabled && !stage.enabled(stage.state, ctx))) {
        continue;
      }
      for (PipelineFrameCallback callback : {stage.prepare, stage.build}) {
        if (callback) {
          auto result = callback(stage.state, ctx);
          if (result.hasError()) {
            return Result<bool, std::string>::makeError(result.error());
          }
        }
      }
    }
    NURI_PROFILER_ZONE_END();
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> RenderPipeline::prepareSceneStep(
    RenderScene &scene, ResourceManager &resources, uint32_t maxOperations,
    const RenderSettings *settings, const Camera *camera, float aspectRatio,
    uint32_t renderWidth, uint32_t renderHeight) {
  NURI_PROFILER_FUNCTION();
  RenderScenePreparationContext ctx{
      .scene = scene,
      .resources = resources,
      .maxOperations = std::max(1u, maxOperations),
      .settings = settings,
      .camera = camera,
      .aspectRatio = aspectRatio,
      .renderWidth = renderWidth,
      .renderHeight = renderHeight,
  };
  bool complete = true;
  for (Component &component : components_) {
    if (!component.desc.prepareScene) {
      continue;
    }
    auto result = component.desc.prepareScene(component.owner.get(), ctx);
    if (result.hasError()) {
      return Result<bool, std::string>::makeError(result.error());
    }
    complete = complete && result.value();
  }
  return Result<bool, std::string>::makeResult(complete);
}

void RenderPipeline::onFrameSubmitted(
    const RenderFrameContext &frame) noexcept {
  for (Component &component : components_) {
    if (component.desc.submitted) {
      component.desc.submitted(component.owner.get(), frame);
    }
  }
}

void RenderPipeline::onFrameAbandoned(
    const RenderFrameContext &frame) noexcept {
  for (Component &component : components_) {
    if (component.desc.abandoned) {
      component.desc.abandoned(component.owner.get(), frame);
    }
  }
}

} // namespace nuri
