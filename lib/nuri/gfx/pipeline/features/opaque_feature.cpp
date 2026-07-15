#include "nuri/pch.h"

#include "nuri/gfx/pipeline/features/opaque_feature.h"

namespace nuri {

bool OpaqueMainPass::isEnabled(const FrameBuildContext &ctx) const {
  return (ctx.frame.settings == nullptr ||
          renderSettingsOrDefault(ctx.frame).opaque.enabled) &&
         renderer_.hasPreparedOpaqueMainPasses();
}

Result<bool, std::string> OpaqueMainPass::build(FrameBuildContext &ctx) {
  return renderer_.appendOpaqueMainPasses(ctx.frame, ctx.graph);
}

bool OpaquePrepassPass::isEnabled(const FrameBuildContext &ctx) const {
  return (ctx.frame.settings == nullptr ||
          renderSettingsOrDefault(ctx.frame).opaque.enabled) &&
         renderer_.hasPreparedOpaquePrepassPasses();
}

Result<bool, std::string> OpaquePrepassPass::build(FrameBuildContext &ctx) {
  return renderer_.appendOpaquePrepassPasses(ctx.frame, ctx.graph);
}

bool OpaqueMainLightingPass::isEnabled(const FrameBuildContext &ctx) const {
  return (ctx.frame.settings == nullptr ||
          renderSettingsOrDefault(ctx.frame).opaque.enabled) &&
         renderer_.hasPreparedOpaqueMainLightingPasses();
}

Result<bool, std::string>
OpaqueMainLightingPass::build(FrameBuildContext &ctx) {
  NURI_ASSERT(renderer_.hasPreparedOpaqueMainLightingPasses(),
              "OpaqueMainFeature requires OpaquePrepassFeature prepare() "
              "before build");
  return renderer_.appendOpaqueMainLightingPasses(ctx.frame, ctx.graph);
}

bool OpaquePickPass::isEnabled(const FrameBuildContext &ctx) const {
  return (ctx.frame.settings == nullptr ||
          renderSettingsOrDefault(ctx.frame).opaque.enabled) &&
         renderer_.hasPreparedOpaquePickPasses();
}

Result<bool, std::string> OpaquePickPass::build(FrameBuildContext &ctx) {
  return renderer_.appendOpaquePickPasses(ctx.frame, ctx.graph);
}

SharedOpaqueRenderer
makeSharedOpaqueRenderer(GPUDevice &gpu, OpaqueRendererConfig config,
                         std::pmr::memory_resource *memory) {
  return std::make_shared<OpaqueRenderer>(gpu, std::move(config), memory);
}

OpaqueFeature::OpaqueFeature(GPUDevice &gpu, OpaqueRendererConfig config,
                             std::pmr::memory_resource *memory)
    : renderer_(
          std::make_unique<OpaqueRenderer>(gpu, std::move(config), memory)),
      mainPass_(*renderer_), pickPass_(*renderer_),
      passes_{&pickPass_, &mainPass_} {
  renderer_->onAttach();
}

OpaqueFeature::~OpaqueFeature() {
  if (renderer_) {
    renderer_->onDetach();
  }
}

Result<bool, std::string> OpaqueFeature::prepare(FrameBuildContext &ctx) {
  return renderer_->prepareOpaqueGraphPasses(ctx.frame);
}

Result<bool, std::string>
OpaqueFeature::publishFrameData(FrameBuildContext &ctx) {
  renderer_->publishFrameData(ctx.frame);
  return Result<bool, std::string>::makeResult(true);
}

void OpaqueFeature::onFrameSubmitted(const RenderFrameContext &frame) noexcept {
  renderer_->commitSubmittedFrame(frame.frameIndex);
}

void OpaqueFeature::onFrameAbandoned(const RenderFrameContext &frame) noexcept {
  renderer_->abandonPreparedFrame(frame.frameIndex);
}

std::span<RenderFeaturePass *const> OpaqueFeature::passes() noexcept {
  return passes_;
}

OpaquePrepassFeature::OpaquePrepassFeature(SharedOpaqueRenderer renderer)
    : renderer_(std::move(renderer)), pickPass_(*renderer_),
      prepass_(*renderer_), passes_{&pickPass_, &prepass_} {
  renderer_->onAttach();
}

OpaquePrepassFeature::~OpaquePrepassFeature() {
  if (renderer_) {
    renderer_->onDetach();
  }
}

Result<bool, std::string>
OpaquePrepassFeature::prepare(FrameBuildContext &ctx) {
  return renderer_->prepareOpaqueGraphPasses(ctx.frame);
}

Result<bool, std::string>
OpaquePrepassFeature::publishFrameData(FrameBuildContext &ctx) {
  renderer_->publishFrameData(ctx.frame);
  return Result<bool, std::string>::makeResult(true);
}

void OpaquePrepassFeature::onFrameSubmitted(
    const RenderFrameContext &frame) noexcept {
  renderer_->commitSubmittedFrame(frame.frameIndex);
}

void OpaquePrepassFeature::onFrameAbandoned(
    const RenderFrameContext &frame) noexcept {
  renderer_->abandonPreparedFrame(frame.frameIndex);
}

std::span<RenderFeaturePass *const> OpaquePrepassFeature::passes() noexcept {
  return passes_;
}

OpaqueMainFeature::OpaqueMainFeature(SharedOpaqueRenderer renderer)
    : renderer_(std::move(renderer)), mainLightingPass_(*renderer_),
      passes_{&mainLightingPass_} {}

std::span<RenderFeaturePass *const> OpaqueMainFeature::passes() noexcept {
  return passes_;
}

} // namespace nuri
