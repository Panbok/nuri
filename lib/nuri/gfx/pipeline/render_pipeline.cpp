#include "nuri/pch.h"

#include "nuri/gfx/pipeline/render_pipeline.h"

#include "nuri/core/profiling.h"
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

bool RenderPipeline::isPassEnabled(size_t index) const noexcept {
  return index < passes_.size() && passes_[index].enabled;
}

bool RenderPipeline::setPassEnabled(size_t index, bool enabled) noexcept {
  if (index >= passes_.size()) {
    return false;
  }
  passes_[index].enabled = enabled;
  return true;
}

void RenderPipeline::registerFeaturePasses(RenderFeature &feature) {
  for (RenderFeaturePass *const pass : feature.passes()) {
    if (pass == nullptr) {
      continue;
    }
    passes_.push_back(
        RegisteredPass{.feature = &feature, .pass = pass, .enabled = true});
  }
}

Result<bool, std::string>
RenderPipeline::buildRenderGraph(RenderFrameContext &frame,
                                 ResourceManager &resources,
                                 RenderGraphBuilder &graph) {
  NURI_PROFILER_FUNCTION();

  FrameBuildContext ctx{
      .frame = frame,
      .graph = graph,
      .resources = resources,
      .shared = frame.sharedResources,
  };

  for (const std::unique_ptr<RenderFeature> &feature : features_) {
    if (!feature) {
      continue;
    }
    auto publishResult = feature->publishFrameData(ctx);
    if (publishResult.hasError()) {
      return Result<bool, std::string>::makeError(publishResult.error());
    }
  }

  for (const std::unique_ptr<FrameDataProvider> &provider : providers_) {
    if (!provider) {
      continue;
    }
    auto result = provider->prepare(ctx);
    if (result.hasError()) {
      return Result<bool, std::string>::makeError(result.error());
    }
  }

  for (const std::unique_ptr<RenderFeature> &feature : features_) {
    if (!feature) {
      continue;
    }
    auto prepareResult = feature->prepare(ctx);
    if (prepareResult.hasError()) {
      return Result<bool, std::string>::makeError(prepareResult.error());
    }
  }

  for (RegisteredPass &entry : passes_) {
    if (entry.feature == nullptr || entry.pass == nullptr || !entry.enabled) {
      continue;
    }
    RenderFeaturePass &pass = *entry.pass;
    if (!pass.isEnabled(ctx)) {
      continue;
    }
    auto prepareResult = pass.prepare(ctx);
    if (prepareResult.hasError()) {
      return Result<bool, std::string>::makeError(prepareResult.error());
    }
    auto buildResult = pass.build(ctx);
    if (buildResult.hasError()) {
      return Result<bool, std::string>::makeError(buildResult.error());
    }
  }

  return Result<bool, std::string>::makeResult(true);
}

} // namespace nuri
