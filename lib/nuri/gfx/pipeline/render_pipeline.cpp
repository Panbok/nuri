#include "nuri/pch.h"

#include "nuri/gfx/pipeline/render_pipeline.h"

#include "nuri/core/profiling.h"
#include "nuri/gfx/frame/presentation_aa_plan.h"
#include "nuri/resources/gpu/resource_manager.h"

namespace nuri {

RenderPipeline::RenderPipeline(std::pmr::memory_resource *memory)
    : providers_(memory != nullptr ? memory : std::pmr::get_default_resource()),
      features_(memory != nullptr ? memory : std::pmr::get_default_resource()),
      passes_(memory != nullptr ? memory : std::pmr::get_default_resource()) {}

FrameDataProvider *
RenderPipeline::addProvider(std::unique_ptr<FrameDataProvider> provider) {
  if (!provider) {
    return nullptr;
  }
  providers_.push_back(std::move(provider));
  return providers_.back().get();
}

RenderFeature *
RenderPipeline::addFeature(std::unique_ptr<RenderFeature> feature) {
  if (!feature) {
    return nullptr;
  }
  features_.push_back(std::move(feature));
  registerFeaturePasses(*features_.back());
  return features_.back().get();
}

std::optional<RenderPipelinePassInfo>
RenderPipeline::passInfo(size_t index) const noexcept {
  if (index >= passes_.size()) {
    return std::nullopt;
  }
  const RegisteredPass &entry = passes_[index];
  return RenderPipelinePassInfo{
      .index = index,
      .featureName =
          entry.feature != nullptr ? entry.feature->name() : std::string_view{},
      .passName =
          entry.pass != nullptr ? entry.pass->name() : std::string_view{},
      .enabled = entry.enabled,
  };
}

std::optional<bool> RenderPipeline::isPassEnabled(size_t index) const noexcept {
  if (index >= passes_.size()) {
    return std::nullopt;
  }
  return passes_[index].enabled;
}

bool RenderPipeline::setPassEnabled(size_t index, bool enabled) noexcept {
  if (index >= passes_.size()) {
    return false;
  }
  passes_[index].enabled = enabled;
  return true;
}

void RenderPipeline::registerFeaturePasses(RenderFeature &feature) {
  size_t insertIndex = passes_.size();
  if (!feature.isTerminalFeature()) {
    while (insertIndex > 0u) {
      const RegisteredPass &entry = passes_[insertIndex - 1u];
      if (entry.feature == nullptr || !entry.feature->isTerminalFeature()) {
        break;
      }
      --insertIndex;
    }
  }
  for (RenderFeaturePass *const pass : feature.passes()) {
    if (pass == nullptr) {
      continue;
    }
    passes_.insert(
        passes_.begin() + static_cast<std::ptrdiff_t>(insertIndex),
        RegisteredPass{.feature = &feature, .pass = pass, .enabled = true});
    ++insertIndex;
  }
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
    NURI_PROFILER_ZONE("RenderPipeline.publish_features",
                       NURI_PROFILER_COLOR_CMD_COPY);
    for (const std::unique_ptr<RenderFeature> &feature : features_) {
      if (!feature) {
        continue;
      }
      auto publishResult = feature->publishFrameData(ctx);
      if (publishResult.hasError()) {
        return Result<bool, std::string>::makeError(publishResult.error());
      }
    }
    NURI_PROFILER_ZONE_END();
  }

  {
    NURI_PROFILER_ZONE("RenderPipeline.prepare_providers",
                       NURI_PROFILER_COLOR_CMD_COPY);
    for (const std::unique_ptr<FrameDataProvider> &provider : providers_) {
      if (!provider) {
        continue;
      }
      auto result = provider->prepare(ctx);
      if (result.hasError()) {
        return Result<bool, std::string>::makeError(result.error());
      }
    }
    NURI_PROFILER_ZONE_END();
  }

  {
    NURI_PROFILER_ZONE("RenderPipeline.prepare_features",
                       NURI_PROFILER_COLOR_CMD_COPY);
    for (const std::unique_ptr<RenderFeature> &feature : features_) {
      if (!feature) {
        continue;
      }
      auto prepareResult = feature->prepare(ctx);
      if (prepareResult.hasError()) {
        return Result<bool, std::string>::makeError(prepareResult.error());
      }
    }
    NURI_PROFILER_ZONE_END();
  }

  {
    NURI_PROFILER_ZONE("RenderPipeline.prepare_build_passes",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    for (const RegisteredPass &entry : passes_) {
      if (entry.feature == nullptr || entry.pass == nullptr || !entry.enabled) {
        continue;
      }
      RenderFeaturePass &pass = *entry.pass;
      if (!pass.isEnabled(ctx)) {
        continue;
      }
      {
        NURI_PROFILER_ZONE("RenderPipeline.pass_prepare",
                           NURI_PROFILER_COLOR_CMD_COPY);
        auto prepareResult = pass.prepare(ctx);
        if (prepareResult.hasError()) {
          return Result<bool, std::string>::makeError(prepareResult.error());
        }
        NURI_PROFILER_ZONE_END();
      }
      {
        NURI_PROFILER_ZONE("RenderPipeline.pass_build",
                           NURI_PROFILER_COLOR_CMD_DRAW);
        auto buildResult = pass.build(ctx);
        if (buildResult.hasError()) {
          return Result<bool, std::string>::makeError(buildResult.error());
        }
        NURI_PROFILER_ZONE_END();
      }
    }
    NURI_PROFILER_ZONE_END();
  }

  return Result<bool, std::string>::makeResult(true);
}

void RenderPipeline::onFrameSubmitted(
    const RenderFrameContext &frame) noexcept {
  for (const std::unique_ptr<FrameDataProvider> &provider : providers_) {
    if (provider) {
      provider->onFrameSubmitted(frame);
    }
  }
  for (const std::unique_ptr<RenderFeature> &feature : features_) {
    if (feature) {
      feature->onFrameSubmitted(frame);
    }
  }
}

void RenderPipeline::onFrameAbandoned(
    const RenderFrameContext &frame) noexcept {
  for (const std::unique_ptr<FrameDataProvider> &provider : providers_) {
    if (provider) {
      provider->onFrameAbandoned(frame);
    }
  }
  for (const std::unique_ptr<RenderFeature> &feature : features_) {
    if (feature) {
      feature->onFrameAbandoned(frame);
    }
  }
}

} // namespace nuri
