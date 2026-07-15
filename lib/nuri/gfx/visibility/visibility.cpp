#include "nuri/pch.h"

#include "nuri/gfx/visibility/visibility.h"

#include "nuri/gfx/frame/presentation_aa_plan.h"

#include <limits>

namespace nuri {
namespace {

VisibilityExecutionMode
resolveMainVisibilityMode(const RenderSettings &settings) noexcept {
  switch (sanitizeVisibilityCullingMode(settings.visibility.mainViewMode)) {
  case VisibilityCullingMode::Disabled:
    return VisibilityExecutionMode::Disabled;
  case VisibilityCullingMode::CpuCoarse:
    return VisibilityExecutionMode::Cpu;
  case VisibilityCullingMode::Hybrid:
    // Hybrid remains the authored compatibility name for the diagnostic path:
    // GPU owns rendering while the CPU independently checks the visible list.
    return settings.visibility.enableGpuInstanceCulling
               ? VisibilityExecutionMode::GpuWithValidation
               : VisibilityExecutionMode::Cpu;
  case VisibilityCullingMode::GpuDriven:
    return settings.visibility.enableGpuInstanceCulling
               ? VisibilityExecutionMode::Gpu
               : VisibilityExecutionMode::Disabled;
  }
  return VisibilityExecutionMode::Disabled;
}

} // namespace

VisibilityPassResult::VisibilityPassResult(std::pmr::memory_resource *memory)
    : visibleCandidateIndices(
          memory != nullptr ? memory : std::pmr::get_default_resource()) {}

void VisibilityPassResult::clear() {
  signature = {};
  visibleCandidateIndices.clear();
  cpuCandidates = 0;
  cpuVisibleCandidates = 0;
  cpuRejected = 0;
  uncertainVisible = 0;
}

VisibilityFrameState::VisibilityFrameState(std::pmr::memory_resource *memory)
    : memory_(memory != nullptr ? memory : std::pmr::get_default_resource()) {}

void VisibilityFrameState::clear() {}

VisibilityPassResult VisibilityFrameState::evaluateCpu(
    const VisibilityPassRequest &request,
    std::span<const VisibilityCandidate> candidates) {
  VisibilityPassResult result(memory_);
  result.signature = request.signature;
  result.cpuCandidates = static_cast<uint32_t>(
      std::min(candidates.size(),
               static_cast<size_t>(std::numeric_limits<uint32_t>::max())));
  result.visibleCandidateIndices.reserve(candidates.size());

  for (size_t i = 0; i < candidates.size(); ++i) {
    const VisibilityCandidate &candidate = candidates[i];
    bool visible = true;
    if (request.enableCpuFrustumCulling) {
      if ((candidate.flags & kVisibilityCandidateConservativeVisible) != 0u) {
        ++result.uncertainVisible;
      } else {
        const visibility_detail::VisibilityClassification classification =
            visibility_detail::classifyTransformedBounds(
                request.frustum, candidate.localBounds,
                candidate.worldFromLocal);
        visible = visibility_detail::isVisible(classification);
      }
    }

    if (!visible) {
      ++result.cpuRejected;
      continue;
    }
    if (i <= static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
      result.visibleCandidateIndices.push_back(static_cast<uint32_t>(i));
    } else {
      ++result.uncertainVisible;
    }
  }

  result.cpuVisibleCandidates = static_cast<uint32_t>(
      std::min(result.visibleCandidateIndices.size(),
               static_cast<size_t>(std::numeric_limits<uint32_t>::max())));
  return result;
}

VisibilityResolvedSettings
visibilitySettingsFromRenderSettings(const RenderSettings &settings) noexcept {
  const VisibilityExecutionMode mainViewMode =
      resolveMainVisibilityMode(settings);
  const VisibilityCullingMode shadowMode =
      sanitizeVisibilityCullingMode(settings.visibility.shadowMode);
  const VisibilityOcclusionMode occlusionMode =
      sanitizeVisibilityOcclusionMode(settings.visibility.occlusionMode);
  return VisibilityResolvedSettings{
      .mainViewMode = mainViewMode,
      .shadowMode = shadowMode,
      .occlusionMode = occlusionMode,
      .enableMeshletFrustumCulling =
          settings.opaque.enableMeshletFrustumCulling ||
          settings.visibility.enableMeshletFrustumCulling,
      .enableMeshletConeCulling = settings.opaque.enableMeshletConeCulling ||
                                  settings.visibility.enableMeshletConeCulling,
      .enableIndirectMeshDispatch =
          settings.visibility.enableIndirectMeshDispatch ||
          usesGpuMainVisibility(mainViewMode),
      .enableMeshletPreTaskCompaction =
          settings.visibility.enableMeshletPreTaskCompaction,
      .enableGpuIndirectDraw = settings.visibility.enableGpuIndirectDraw ||
                               usesGpuMainVisibility(mainViewMode),
      .enableOcclusionCulling =
          occlusionMode == VisibilityOcclusionMode::PreviousFrameHiZ ||
          occlusionMode == VisibilityOcclusionMode::CurrentFrameHiZExperimental,
      .visibleOnUncertain = settings.visibility.visibleOnUncertain,
      .forcedVisibleListCapacity =
          settings.visibility.debug.forcedVisibleListCapacity,
  };
}

VisibilityPassRequest makeMainViewVisibilityPassRequest(
    const CameraFrameState &camera,
    const VisibilityResolvedSettings &settings) noexcept {
  VisibilityPassRequest request{};
  request.kind = VisibilityPassKind::OpaqueMain;
  request.signature.kind = VisibilityPassKind::OpaqueMain;
  request.frustum = visibility_detail::buildCameraFrustumPlanes(camera);
  request.enableCpuFrustumCulling =
      runsCpuVisibilityEvaluation(settings.mainViewMode);
  return request;
}

VisibilityCandidateGpu
makeVisibilityCandidateGpu(const VisibilityCandidate &candidate) noexcept {
  const glm::vec4 sphere = visibility_detail::transformBoundingSphere(
      candidate.localBounds, candidate.worldFromLocal);
  const BoundingBox worldBounds =
      candidate.localBounds.getTransformed(candidate.worldFromLocal);

  VisibilityCandidateGpu out{};
  out.ids = glm::uvec4(candidate.renderableIndex, candidate.templateIndex,
                       candidate.submeshIndex, candidate.materialIndex);
  out.ranges = glm::uvec4(0u, 0u, candidate.flags, 0u);
  out.bounds = sphere;
  out.boundsMin = glm::vec4(worldBounds.min_, 0.0f);
  out.boundsMax = glm::vec4(worldBounds.max_, 0.0f);
  return out;
}

VisibilityPassGpuData makeMainViewVisibilityPassGpuData(
    const CameraFrameState &camera, const VisibilityPassRequest &request,
    uint32_t candidateCount, bool occlusionAvailable,
    uint32_t depthPyramidWidth, uint32_t depthPyramidHeight,
    uint32_t depthPyramidLevelCount, uint32_t depthPyramidSamplerId,
    std::span<const glm::uvec4> depthPyramidTexIds) noexcept {
  VisibilityPassGpuData out{};
  out.view = camera.view;
  out.proj = cameraCurrentUnjitteredProjection(camera);
  out.viewProj = cameraCurrentUnjitteredViewProjection(camera);
  out.previousViewProj = hasTemporalCameraContinuity(camera)
                             ? camera.previousUnjitteredViewProj
                             : out.viewProj;
  for (size_t i = 0; i < request.frustum.planes.size(); ++i) {
    out.planes[i] = request.frustum.planes[i];
  }
  out.cameraOrLightPos = camera.cameraPos;
  out.passInfo = glm::uvec4(static_cast<uint32_t>(request.kind),
                            request.signature.cascadeIndex,
                            request.signature.flags, candidateCount);
  out.depthPyramidInfo =
      occlusionAvailable
          ? glm::uvec4(depthPyramidWidth, depthPyramidHeight,
                       depthPyramidLevelCount, depthPyramidSamplerId)
          : glm::uvec4(0u);
  const size_t copyCount =
      std::min(depthPyramidTexIds.size(), out.depthPyramidTexIds.size());
  for (size_t i = 0; i < copyCount; ++i) {
    out.depthPyramidTexIds[i] = depthPyramidTexIds[i];
  }
  return out;
}

} // namespace nuri
