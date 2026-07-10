#include "nuri/tools/autotest/autotest_assertion.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>

namespace nuri::tools::autotest {
namespace {

constexpr uint64_t kInvalidFrameIndex = std::numeric_limits<uint64_t>::max();
constexpr double kEqualsEpsilon = 1.0e-9;

void addMetric(std::map<std::string, double> &out, std::string_view id,
               double value) {
  out[std::string(id)] = value;
}

void addBoolMetric(std::map<std::string, double> &out, std::string_view id,
                   bool value) {
  out[std::string(id)] = value ? 1.0 : 0.0;
}

void addBytesAsMiB(std::map<std::string, double> &out, std::string_view id,
                   uint64_t bytes) {
  out[std::string(id)] = static_cast<double>(bytes) / (1024.0 * 1024.0);
}

void applyGpuScope(std::map<uint64_t, std::map<std::string, double>> &frames,
                   const GpuTimingReport &report, GpuTimingScope scope,
                   uint64_t frameIndex, std::string_view metricId,
                   float timeMs) {
  if (frameIndex == kInvalidFrameIndex || !hasGpuTimingScope(report, scope)) {
    return;
  }
  frames[frameIndex][std::string(metricId)] = static_cast<double>(timeMs);
}

[[nodiscard]] bool
shouldIncludeGpuScopeInSum(const std::map<std::string, double> &metrics,
                           std::string_view id) {
  if (id == "gpu.scopes_sum_ms") {
    return false;
  }
  if (id.rfind("gpu.scopes.", 0u) != 0u) {
    return false;
  }
  if (metrics.find("gpu.scopes.shadow_ms") != metrics.end() &&
      (id == "gpu.scopes.shadow_depth_ms" ||
       id == "gpu.scopes.shadow_sdsm_ms")) {
    return false;
  }
  return true;
}

[[nodiscard]] double percentileR7(std::vector<double> sortedValues,
                                  double percentile) {
  if (sortedValues.empty()) {
    return 0.0;
  }
  std::sort(sortedValues.begin(), sortedValues.end());
  const double p = std::clamp(percentile, 0.0, 1.0);
  const double h = 1.0 + (static_cast<double>(sortedValues.size()) - 1.0) * p;
  const double floorH = std::floor(h);
  const size_t lowerIndex = static_cast<size_t>(floorH) - 1u;
  const size_t upperIndex = std::min(lowerIndex + 1u, sortedValues.size() - 1u);
  const double fraction = h - floorH;
  return sortedValues[lowerIndex] +
         fraction * (sortedValues[upperIndex] - sortedValues[lowerIndex]);
}

} // namespace

std::string autotestAssertionStatusName(AutotestAssertionStatus status) {
  switch (status) {
  case AutotestAssertionStatus::Pass:
    return "pass";
  case AutotestAssertionStatus::Warn:
    return "warn";
  case AutotestAssertionStatus::Fail:
    return "fail";
  case AutotestAssertionStatus::Unavailable:
    return "unavailable";
  case AutotestAssertionStatus::Invalid:
    return "invalid";
  }
  return "unknown";
}

void flattenAutotestRendererMetrics(std::map<std::string, double> &out,
                                    const RenderFrameMetrics &metrics) {
  addMetric(out, "renderer.frame_index",
            static_cast<double>(metrics.frameIndex));

  const OpaqueFrameMetrics &opaque = metrics.opaque;
  addMetric(out, "renderer.opaque.total_instances", opaque.totalInstances);
  addMetric(out, "renderer.opaque.visible_instances", opaque.visibleInstances);
  addMetric(out, "renderer.opaque.instanced_draws", opaque.instancedDraws);
  addMetric(out, "renderer.opaque.indirect_draw_calls",
            opaque.indirectDrawCalls);
  addMetric(out, "renderer.opaque.indirect_commands", opaque.indirectCommands);
  addMetric(out, "renderer.opaque.compute_dispatches",
            opaque.computeDispatches);
  addMetric(out, "renderer.opaque.depth_prepass_draws",
            opaque.depthPrepassDraws);
  addMetric(out, "renderer.opaque.depth_prepass_enabled",
            opaque.depthPrepassEnabled);
  addMetric(out, "renderer.opaque.depth_pyramid_levels",
            opaque.depthPyramidLevels);
  addMetric(out, "renderer.opaque.tessellated_draws", opaque.tessellatedDraws);
  addMetric(out, "renderer.opaque.meshlet_dispatches",
            opaque.meshletDispatches);
  addMetric(out, "renderer.opaque.meshlet_task_groups",
            opaque.meshletTaskGroups);
  addMetric(out, "renderer.opaque.meshlet_candidates",
            opaque.meshletCandidateCount);
  addMetric(out, "renderer.opaque.meshlet_mode_required",
            opaque.meshletModeRequired);
  addMetric(out, "renderer.opaque.meshlet_mode_active",
            opaque.meshletModeActive);
  addMetric(out, "renderer.opaque.meshlet_rejected_missing_feature",
            opaque.meshletRejectedMissingFeature);
  addMetric(out, "renderer.opaque.meshlet_rejected_missing_asset_data",
            opaque.meshletRejectedMissingAssetData);
  addMetric(out, "renderer.opaque.meshlet_rejected_incompatible_frame",
            opaque.meshletRejectedIncompatibleFrame);

  const VisibilityFrameMetrics &visibility = metrics.visibility;
  addMetric(out, "renderer.visibility.cpu_main_candidates",
            visibility.cpuMainCandidates);
  addMetric(out, "renderer.visibility.cpu_main_visible_candidates",
            visibility.cpuMainVisibleCandidates);
  addMetric(out, "renderer.visibility.cpu_main_rejected",
            visibility.cpuMainRejected);
  addMetric(out, "renderer.visibility.gpu_main_candidates",
            visibility.gpuMainCandidates);
  addMetric(out, "renderer.visibility.gpu_main_visible_candidates",
            visibility.gpuMainVisibleCandidates);
  addMetric(out, "renderer.visibility.gpu_main_rejected_frustum",
            visibility.gpuMainRejectedFrustum);
  addMetric(out, "renderer.visibility.gpu_main_rejected_occlusion",
            visibility.gpuMainRejectedOcclusion);
  addMetric(out, "renderer.visibility.gpu_output_overflow_count",
            visibility.gpuOutputOverflowCount);
  addMetric(out, "renderer.visibility.gpu_main_readback_available",
            visibility.gpuMainReadbackAvailable);
  addMetric(out, "renderer.visibility.gpu_main_readback_source_frame",
            visibility.gpuMainReadbackSourceFrame);
  addMetric(out, "renderer.visibility.gpu_main_readback_stale_frame_count",
            visibility.gpuMainReadbackStaleFrameCount);
  addMetric(out, "renderer.visibility.gpu_main_readback_error_count",
            visibility.gpuMainReadbackErrorCount);
  addMetric(out, "renderer.visibility.gpu_main_readback_visible_candidates",
            visibility.gpuMainReadbackVisibleCandidates);
  addMetric(out, "renderer.visibility.gpu_main_visible_list_mismatches",
            visibility.gpuMainVisibleListMismatches);
  addMetric(out, "renderer.visibility.gpu_indirect_draw_used",
            visibility.gpuIndirectDrawUsed);
  addMetric(out, "renderer.visibility.gpu_indirect_draw_fallback",
            visibility.gpuIndirectDrawFallback);
  addMetric(out, "renderer.visibility.gpu_indirect_draw_commands",
            visibility.gpuIndirectDrawCommands);
  addMetric(out, "renderer.visibility.gpu_indirect_draw_readback_commands",
            visibility.gpuIndirectDrawReadbackCommands);
  addMetric(out, "renderer.visibility.gpu_indirect_draw_readback_tombstoned",
            visibility.gpuIndirectDrawReadbackTombstoned);
  addMetric(out, "renderer.visibility.gpu_indirect_draw_readback_visible",
            visibility.gpuIndirectDrawReadbackVisible);
  addMetric(out, "renderer.visibility.indirect_mesh_dispatch_count",
            visibility.indirectMeshDispatchCount);
  addMetric(out, "renderer.visibility.meshlet_rejected_frustum",
            visibility.meshletRejectedFrustum);
  addMetric(out, "renderer.visibility.meshlet_rejected_cone",
            visibility.meshletRejectedCone);
  addMetric(out, "renderer.visibility.meshlet_rejected_occlusion",
            visibility.meshletRejectedOcclusion);
  addMetric(out, "renderer.visibility.meshlet_occlusion_available",
            visibility.meshletOcclusionAvailable);
  addMetric(out, "renderer.visibility.meshlet_payload_overflow_count",
            visibility.meshletPayloadOverflowCount);
  addMetric(out, "renderer.visibility.meshlet_readback_available",
            visibility.meshletReadbackAvailable);
  addMetric(out, "renderer.visibility.meshlet_readback_source_frame",
            visibility.meshletReadbackSourceFrame);
  addMetric(out, "renderer.visibility.meshlet_readback_stale_frame_count",
            visibility.meshletReadbackStaleFrameCount);
  addMetric(out, "renderer.visibility.meshlet_readback_error_count",
            visibility.meshletReadbackErrorCount);
  addMetric(out, "renderer.visibility.meshlet_emitted",
            visibility.meshletEmitted);
  addMetric(out, "renderer.visibility.meshlet_task_groups_executed",
            visibility.meshletTaskGroupsExecuted);
  addMetric(out, "renderer.visibility.uncertain_visible",
            visibility.uncertainVisible);
  addMetric(out, "renderer.visibility.shadow_cpu_candidates",
            visibility.shadowCpuCandidates);
  addMetric(out, "renderer.visibility.shadow_cpu_rejected",
            visibility.shadowCpuRejected);
  addMetric(out, "renderer.visibility.shadow_meshlet_candidates",
            visibility.shadowMeshletCandidates);
  addMetric(out, "renderer.visibility.shadow_meshlet_readback_available",
            visibility.shadowMeshletReadbackAvailable);
  addMetric(out, "renderer.visibility.shadow_meshlet_readback_source_frame",
            visibility.shadowMeshletReadbackSourceFrame);
  addMetric(out,
            "renderer.visibility.shadow_meshlet_readback_stale_frame_count",
            visibility.shadowMeshletReadbackStaleFrameCount);
  addMetric(out, "renderer.visibility.shadow_meshlet_readback_error_count",
            visibility.shadowMeshletReadbackErrorCount);
  addMetric(out, "renderer.visibility.shadow_meshlet_rejected_bounds",
            visibility.shadowMeshletRejectedBounds);
  addMetric(out, "renderer.visibility.occlusion_available",
            visibility.occlusionAvailable);

  const ShadowFrameMetrics &shadow = metrics.shadow;
  addMetric(out, "renderer.shadow.cascades", shadow.cascadeCount);
  addMetric(out, "renderer.shadow.total_draws", shadow.totalDraws);
  addMetric(out, "renderer.shadow.total_culled_draws", shadow.totalCulledDraws);
  addMetric(out, "renderer.shadow.static_caster_entries",
            shadow.staticCasterEntries);
  addMetric(out, "renderer.shadow.dynamic_caster_entries",
            shadow.dynamicCasterEntries);
  addMetric(out, "renderer.shadow.static_cache_reused",
            shadow.staticCacheReused);
  addMetric(out, "renderer.shadow.static_batch_templates",
            shadow.staticBatchTemplateCount);
  addMetric(out, "renderer.shadow.batch_entries", shadow.shadowBatchEntryCount);
  addMetric(out, "renderer.shadow.meshlet_dispatches",
            shadow.shadowMeshletDispatchCount);
  addMetric(out, "renderer.shadow.meshlet_task_groups",
            shadow.shadowMeshletTaskGroupCount);
  addMetric(out, "renderer.shadow.instance_remaps",
            shadow.shadowInstanceRemapCount);
  addMetric(out, "renderer.shadow.static_batch_full_emits",
            shadow.staticBatchFullEmitCount);
  addMetric(out, "renderer.shadow.static_light_grid_queries",
            shadow.staticLightGridQueryCount);
  addMetric(out, "renderer.shadow.static_light_grid_fallback_scans",
            shadow.staticLightGridFallbackScanCount);
  addMetric(out, "renderer.shadow.static_light_grid_query_cells",
            shadow.staticLightGridQueryCellCount);
  addMetric(out, "renderer.shadow.static_light_grid_candidates",
            shadow.staticLightGridCandidateCount);
  addMetric(out, "renderer.shadow.static_only_candidates",
            shadow.staticOnlyCandidateCount);
  addMetric(out, "renderer.shadow.static_only_reused_cascades",
            shadow.reusedStaticOnlyCascadeCount);
  addMetric(out, "renderer.shadow.static_only_miss_static_cache_rebuilt",
            shadow.staticOnlyReuseMissStaticCacheRebuiltCount);
  addMetric(out, "renderer.shadow.static_only_miss_dynamic_caster",
            shadow.staticOnlyReuseMissDynamicCasterCount);
  addMetric(out, "renderer.shadow.static_only_miss_no_previous",
            shadow.staticOnlyReuseMissNoPreviousCount);
  addMetric(out, "renderer.shadow.static_only_miss_raster_state_changed",
            shadow.staticOnlyReuseMissRasterStateChangedCount);
  addMetric(out, "renderer.shadow.static_only_miss_adaptive_refresh",
            shadow.staticOnlyReuseMissAdaptiveRefreshCount);
  addMetric(out, "renderer.shadow.static_only_scroll_candidates",
            shadow.staticOnlyScrollCandidateCount);
  addMetric(out, "renderer.shadow.static_only_scroll_compatible",
            shadow.staticOnlyScrollCompatibleCount);
  addMetric(out, "renderer.shadow.static_only_scroll_dirty_area_bp",
            shadow.staticOnlyScrollDirtyAreaBasisPoints);
  addMetric(out, "renderer.shadow.static_only_scroll_dirty_casters",
            shadow.staticOnlyScrollDirtyCasterEstimate);
  addMetric(out, "renderer.shadow.static_only_scroll_dirty_indices",
            shadow.staticOnlyScrollDirtyIndexEstimate);
  addMetric(out, "renderer.shadow.static_only_scroll_reject_anchor",
            shadow.staticOnlyScrollRejectAnchorCount);
  addMetric(out, "renderer.shadow.static_only_scroll_reject_depth",
            shadow.staticOnlyScrollRejectDepthCount);
  addMetric(out, "renderer.shadow.static_only_scroll_reject_extent",
            shadow.staticOnlyScrollRejectExtentCount);
  addMetric(out, "renderer.shadow.static_only_scroll_reject_shift",
            shadow.staticOnlyScrollRejectShiftCount);
  addMetric(out, "renderer.shadow.filter_sample_budget",
            shadow.filterSampleBudget);
  addMetric(out, "renderer.shadow.sdsm_compute_passes",
            shadow.sdsmComputePassCount);
  addBytesAsMiB(out, "gpu.memory.shadow.cascade_texture_mb",
                shadow.cascadeTextureBytes);

  const AntiAliasingFrameMetrics &aa = metrics.antiAliasing;
  addBoolMetric(out, "renderer.aa.history_valid", aa.historyValid);
  addBoolMetric(out, "renderer.aa.temporal_data_valid", aa.temporalDataValid);
  addMetric(out, "renderer.aa.history_reset_count", aa.historyResetCount);
  addMetric(out, "renderer.aa.frames_since_history_reset",
            aa.framesSinceHistoryReset);
  addMetric(out, "renderer.aa.camera_position_delta", aa.cameraPositionDelta);
  addMetric(out, "renderer.aa.jitter_delta_magnitude", aa.jitterDeltaMagnitude);
  addMetric(out, "renderer.aa.motion_vector_textures",
            aa.motionVectorTextureCount);
  addMetric(out, "renderer.aa.motion_vector_allocations",
            aa.motionVectorAllocationCount);
  addMetric(out, "renderer.aa.motion_vector_reallocations",
            aa.motionVectorReallocationCount);
  addMetric(out, "renderer.aa.motion_vector_depth_reprojection_passes",
            aa.motionVectorDepthReprojectionPassCount);
  addMetric(out, "renderer.aa.velocity_passes", aa.velocityPassCount);
  addMetric(out, "renderer.aa.velocity_draws", aa.velocityDrawCount);
  addMetric(out, "renderer.aa.velocity_instances", aa.velocityInstanceCount);
  addMetric(out, "renderer.aa.velocity_previous_transform_valid",
            aa.velocityPreviousTransformValidCount);
  addMetric(out, "renderer.aa.velocity_missing_previous_transform",
            aa.velocityMissingPreviousTransformCount);
  addMetric(out, "renderer.aa.velocity_animated_responsive",
            aa.velocityAnimatedResponsiveCount);
  addMetric(out, "renderer.aa.velocity_animated_previous_geometry",
            aa.velocityAnimatedPreviousGeometryCount);
  addMetric(out, "renderer.aa.velocity_average_object_motion",
            aa.velocityAverageObjectMotion);
  addMetric(out, "renderer.aa.velocity_max_object_motion",
            aa.velocityMaxObjectMotion);
  addMetric(out, "renderer.aa.velocity_estimated_average_magnitude",
            aa.velocityEstimatedAverageMagnitude);
  addMetric(out, "renderer.aa.velocity_estimated_max_magnitude",
            aa.velocityEstimatedMaxMagnitude);
  addMetric(out, "renderer.aa.velocity_missing_previous_ratio",
            aa.velocityMissingPreviousRatio);
  addMetric(out, "renderer.aa.velocity_edge_discontinuity_estimate",
            aa.velocityEdgeDiscontinuityEstimate);
  addMetric(out, "renderer.aa.reactive_mask_passes", aa.reactiveMaskPassCount);
  addMetric(out, "renderer.aa.reactive_mask_draws", aa.reactiveMaskDrawCount);
  addMetric(out, "renderer.aa.transparent_transmission_feedback_refreshes",
            aa.transparentTransmissionFeedbackRefreshCount);
  addMetric(out, "renderer.aa.transparent_transmission_blend_draws",
            aa.transparentTransmissionBlendDrawCount);
  addMetric(out, "renderer.aa.transparent_transmission_feedback_available",
            aa.transparentTransmissionFeedbackSourceAvailable);
  addMetric(out, "renderer.aa.taa_resolve_passes", aa.taaResolvePassCount);
  addMetric(out, "renderer.aa.taa_copy_back_passes", aa.taaCopyBackPassCount);
  addMetric(out, "renderer.aa.spatial_aa_passes", aa.spatialAAPassCount);
  addMetric(out, "renderer.aa.msaa_resolve_passes", aa.msaaResolvePassCount);
  addBytesAsMiB(out, "gpu.memory.aa.motion_vector_total_mb",
                aa.motionVectorTotalBytes);
  addBytesAsMiB(out, "gpu.memory.aa.reactive_mask_total_mb",
                aa.reactiveMaskTotalBytes);
  addBytesAsMiB(out, "gpu.memory.aa.spatial_aa_total_mb",
                aa.spatialAATotalBytes);
  addBytesAsMiB(out, "gpu.memory.aa.msaa_total_mb", aa.msaaTotalBytes);

  const AmbientOcclusionFrameMetrics &ao = metrics.ambientOcclusion;
  addBoolMetric(out, "renderer.ao.enabled", ao.enabled);
  addBoolMetric(out, "renderer.ao.active", ao.active);
  addMetric(out, "renderer.ao.normal_prepass_draws", ao.normalPrepassDraws);
  addMetric(out, "renderer.ao.depth_prefilter_passes",
            ao.depthPrefilterPassCount);
  addMetric(out, "renderer.ao.main_passes", ao.mainPassCount);
  addMetric(out, "renderer.ao.temporal_passes", ao.temporalPassCount);
  addMetric(out, "renderer.ao.texture_count", ao.textureCount);
  addBytesAsMiB(out, "gpu.memory.ao.total_texture_mb", ao.totalTextureBytes);

  const HDRPostProcessFrameMetrics &hdr = metrics.hdrPostProcess;
  addBoolMetric(out, "renderer.hdr.bloom_enabled", hdr.bloomEnabled);
  addBoolMetric(out, "renderer.hdr.bloom_active", hdr.bloomActive);
  addMetric(out, "renderer.hdr.bloom_passes", hdr.bloomPassCount);
  addMetric(out, "renderer.hdr.luminance_passes", hdr.luminancePassCount);
  addMetric(out, "renderer.hdr.adaptation_passes", hdr.adaptationPassCount);
  addMetric(out, "renderer.hdr.texture_count", hdr.textureCount);
  addBytesAsMiB(out, "gpu.memory.hdr.texture_mb", hdr.textureBytes);

  addMetric(out, "renderer.transparent.mesh_draws",
            metrics.transparent.meshDraws);
  addMetric(out, "renderer.transparent.contributor_sortable_draws",
            metrics.transparent.contributorSortableDraws);
  addMetric(out, "renderer.transparent.contributor_fixed_draws",
            metrics.transparent.contributorFixedDraws);
  addMetric(out, "renderer.transparent.pick_draws",
            metrics.transparent.pickDraws);
}

void applyAutotestGpuTimingReport(
    std::map<uint64_t, std::map<std::string, double>> &frames,
    const GpuTimingReport &report) {
  applyGpuScope(frames, report, GpuTimingScope::Shadow,
                report.shadowSourceFrameIndex, "gpu.scopes.shadow_ms",
                report.shadowTimeMs);
  applyGpuScope(frames, report, GpuTimingScope::ShadowDepth,
                report.shadowDepthSourceFrameIndex,
                "gpu.scopes.shadow_depth_ms", report.shadowDepthTimeMs);
  applyGpuScope(frames, report, GpuTimingScope::ShadowSdsm,
                report.shadowSdsmSourceFrameIndex, "gpu.scopes.shadow_sdsm_ms",
                report.shadowSdsmTimeMs);
  applyGpuScope(frames, report, GpuTimingScope::Opaque,
                report.opaqueSourceFrameIndex, "gpu.scopes.opaque_ms",
                report.opaqueTimeMs);
  applyGpuScope(frames, report, GpuTimingScope::GTAO,
                report.gtaoSourceFrameIndex, "gpu.scopes.gtao_ms",
                report.gtaoTimeMs);
  applyGpuScope(frames, report, GpuTimingScope::MsaaResolve,
                report.msaaResolveSourceFrameIndex,
                "gpu.scopes.msaa_resolve_ms", report.msaaResolveTimeMs);
  applyGpuScope(frames, report, GpuTimingScope::SceneColorDownsample,
                report.sceneColorDownsampleSourceFrameIndex,
                "gpu.scopes.scene_color_downsample_ms",
                report.sceneColorDownsampleTimeMs);
  applyGpuScope(frames, report, GpuTimingScope::TemporalAAResolve,
                report.temporalAAResolveSourceFrameIndex,
                "gpu.scopes.taa_resolve_ms", report.temporalAAResolveTimeMs);
  applyGpuScope(frames, report, GpuTimingScope::TemporalAADebug,
                report.temporalAADebugSourceFrameIndex,
                "gpu.scopes.taa_debug_ms", report.temporalAADebugTimeMs);
  applyGpuScope(frames, report, GpuTimingScope::SpatialAA,
                report.spatialAASourceFrameIndex, "gpu.scopes.spatial_aa_ms",
                report.spatialAATimeMs);
  applyGpuScope(frames, report, GpuTimingScope::Transmission,
                report.transmissionSourceFrameIndex,
                "gpu.scopes.transmission_ms", report.transmissionTimeMs);
  applyGpuScope(frames, report, GpuTimingScope::HDRPostProcess,
                report.hdrPostProcessSourceFrameIndex,
                "gpu.scopes.hdr_postprocess_ms", report.hdrPostProcessTimeMs);
  applyGpuScope(frames, report, GpuTimingScope::Skybox,
                report.skyboxSourceFrameIndex, "gpu.scopes.skybox_ms",
                report.skyboxTimeMs);

  for (auto &[frameIndex, metrics] : frames) {
    double sum = 0.0;
    bool any = false;
    for (const auto &[id, value] : metrics) {
      if (shouldIncludeGpuScopeInSum(metrics, id)) {
        sum += value;
        any = true;
      }
    }
    if (any) {
      metrics["gpu.scopes_sum_ms"] = sum;
    }
  }
}

AutotestAssertionResult
evaluateAutotestAssertion(const AutotestMetricAssertion &assertion,
                          const std::map<std::string, double> &measurements) {
  AutotestAssertionResult result{};
  result.id = assertion.id;
  result.metric = assertion.metric;
  const auto found = measurements.find(assertion.metric);
  if (found == measurements.end()) {
    result.status = assertion.optional ? AutotestAssertionStatus::Unavailable
                                       : AutotestAssertionStatus::Invalid;
    result.statusReason = assertion.optional ? "optional_metric_unavailable"
                                             : "required_metric_unavailable";
    return result;
  }

  const double actual = found->second;
  result.actual = actual;
  result.hasActual = true;
  result.sampleCount = 1u;
  bool passed = true;
  if (assertion.hasEquals) {
    passed = passed && std::abs(actual - assertion.equals) <= kEqualsEpsilon;
  }
  if (assertion.hasMin) {
    passed = passed && actual >= assertion.min;
  }
  if (assertion.hasMax) {
    passed = passed && actual <= assertion.max;
  }
  if (assertion.hasLessThan) {
    passed = passed && actual < assertion.lessThan;
  }
  if (assertion.hasLessOrEqual) {
    passed = passed && actual <= assertion.lessOrEqual;
  }
  if (assertion.hasGreaterThan) {
    passed = passed && actual > assertion.greaterThan;
  }
  if (assertion.hasGreaterOrEqual) {
    passed = passed && actual >= assertion.greaterOrEqual;
  }
  if (passed) {
    result.status = AutotestAssertionStatus::Pass;
    result.statusReason = "passed";
    return result;
  }
  result.status = assertion.severity == "warn" ? AutotestAssertionStatus::Warn
                                               : AutotestAssertionStatus::Fail;
  result.statusReason = "threshold_failed";
  return result;
}

std::vector<AutotestAssertionResult> evaluateAutotestAssertions(
    const std::vector<AutotestMetricAssertion> &assertions,
    const std::map<std::string, double> &measurements) {
  std::vector<AutotestAssertionResult> results;
  results.reserve(assertions.size());
  for (const AutotestMetricAssertion &assertion : assertions) {
    results.push_back(evaluateAutotestAssertion(assertion, measurements));
  }
  return results;
}

Result<AutotestMetricStats, std::string>
computeAutotestMetricStats(std::vector<double> values) {
  if (values.empty()) {
    return Result<AutotestMetricStats, std::string>::makeError(
        "computeAutotestMetricStats: no values");
  }
  for (const double value : values) {
    if (!std::isfinite(value)) {
      return Result<AutotestMetricStats, std::string>::makeError(
          "computeAutotestMetricStats: non-finite value");
    }
  }

  std::sort(values.begin(), values.end());
  const double sum = std::accumulate(values.begin(), values.end(), 0.0);
  const double mean = sum / static_cast<double>(values.size());
  AutotestMetricStats stats{};
  stats.count = static_cast<uint32_t>(values.size());
  stats.min = values.front();
  stats.max = values.back();
  stats.mean = mean;
  stats.median = percentileR7(values, 0.50);
  stats.p95 = percentileR7(values, 0.95);
  if (values.size() >= 2u) {
    double variance = 0.0;
    for (const double value : values) {
      const double diff = value - mean;
      variance += diff * diff;
    }
    stats.variance = variance / static_cast<double>(values.size() - 1u);
  }
  return Result<AutotestMetricStats, std::string>::makeResult(stats);
}

AutotestAssertionResult evaluateAutotestMetricWindowAssertion(
    const AutotestMetricWindowAssertion &assertion,
    const std::map<uint64_t, std::map<std::string, double>> &frames,
    uint32_t startFrame, uint32_t endFrame) {
  AutotestAssertionResult result{};
  result.id = assertion.id;
  result.metric = assertion.metric;
  if (startFrame > endFrame) {
    result.status = AutotestAssertionStatus::Invalid;
    result.statusReason = "invalid_metric_window_range";
    return result;
  }
  const uint64_t expectedSampleCount =
      static_cast<uint64_t>(endFrame) - startFrame + 1u;
  if (expectedSampleCount > UINT32_MAX) {
    result.status = AutotestAssertionStatus::Invalid;
    result.statusReason = "metric_window_sample_count_overflow";
    return result;
  }
  result.expectedSampleCount = static_cast<uint32_t>(expectedSampleCount);

  std::vector<double> samples;
  samples.reserve(static_cast<size_t>(endFrame - startFrame) + 1u);
  for (uint32_t frame = startFrame; frame <= endFrame; ++frame) {
    const auto frameIt = frames.find(frame);
    if (frameIt == frames.end()) {
      continue;
    }
    const auto metricIt = frameIt->second.find(assertion.metric);
    if (metricIt != frameIt->second.end()) {
      samples.push_back(metricIt->second);
    }
    if (frame == UINT32_MAX) {
      break;
    }
  }
  result.sampleCount = static_cast<uint32_t>(samples.size());
  if (samples.empty()) {
    result.status = assertion.optional ? AutotestAssertionStatus::Unavailable
                                       : AutotestAssertionStatus::Invalid;
    result.statusReason = assertion.optional ? "optional_metric_unavailable"
                                             : "required_metric_unavailable";
    return result;
  }
  if (samples.size() != expectedSampleCount) {
    result.status = assertion.optional ? AutotestAssertionStatus::Unavailable
                                       : AutotestAssertionStatus::Invalid;
    result.statusReason = assertion.optional
                              ? "optional_metric_window_incomplete"
                              : "required_metric_window_incomplete";
    return result;
  }

  auto statsResult = computeAutotestMetricStats(std::move(samples));
  if (statsResult.hasError()) {
    result.status = AutotestAssertionStatus::Invalid;
    result.statusReason = statsResult.error();
    return result;
  }
  const AutotestMetricStats &stats = statsResult.value();
  result.sampleCount = stats.count;

  auto recordActual = [&](std::string_view statistic, double value) {
    if (!result.hasActual) {
      result.statistic = statistic;
      result.actual = value;
      result.hasActual = true;
    }
  };
  auto fail = [&](std::string_view statistic, double value) {
    result.statistic = statistic;
    result.actual = value;
    result.hasActual = true;
    result.status = assertion.severity == "warn"
                        ? AutotestAssertionStatus::Warn
                        : AutotestAssertionStatus::Fail;
    result.statusReason = "threshold_failed";
  };

  if (assertion.hasEquals) {
    recordActual("all", stats.max);
    if (std::abs(stats.min - assertion.equals) > kEqualsEpsilon ||
        std::abs(stats.max - assertion.equals) > kEqualsEpsilon) {
      fail("all", stats.max);
    }
  }
  if (result.status == AutotestAssertionStatus::Pass && assertion.hasMin) {
    recordActual("min", stats.min);
    if (stats.min < assertion.min) {
      fail("min", stats.min);
    }
  }
  if (result.status == AutotestAssertionStatus::Pass && assertion.hasMax) {
    recordActual("max", stats.max);
    if (stats.max > assertion.max) {
      fail("max", stats.max);
    }
  }
  if (result.status == AutotestAssertionStatus::Pass &&
      assertion.hasMedianMin) {
    recordActual("median", stats.median);
    if (stats.median < assertion.medianMin) {
      fail("median", stats.median);
    }
  }
  if (result.status == AutotestAssertionStatus::Pass &&
      assertion.hasMedianMax) {
    recordActual("median", stats.median);
    if (stats.median > assertion.medianMax) {
      fail("median", stats.median);
    }
  }
  if (result.status == AutotestAssertionStatus::Pass && assertion.hasP95Min) {
    recordActual("p95", stats.p95);
    if (stats.p95 < assertion.p95Min) {
      fail("p95", stats.p95);
    }
  }
  if (result.status == AutotestAssertionStatus::Pass && assertion.hasP95Max) {
    recordActual("p95", stats.p95);
    if (stats.p95 > assertion.p95Max) {
      fail("p95", stats.p95);
    }
  }
  if (result.status == AutotestAssertionStatus::Pass &&
      assertion.hasVarianceMax) {
    recordActual("variance", stats.variance);
    if (stats.variance > assertion.varianceMax) {
      fail("variance", stats.variance);
    }
  }
  if (!result.hasActual) {
    result.statistic = "mean";
    result.actual = stats.mean;
    result.hasActual = true;
  }
  if (result.status == AutotestAssertionStatus::Pass) {
    result.statusReason = "passed";
  }
  return result;
}

std::vector<AutotestAssertionResult> evaluateAutotestMetricWindowAssertions(
    const AutotestMetricWindow &window,
    const std::map<uint64_t, std::map<std::string, double>> &frames) {
  std::vector<AutotestAssertionResult> results;
  results.reserve(window.assertions.size());
  for (const AutotestMetricWindowAssertion &assertion : window.assertions) {
    results.push_back(evaluateAutotestMetricWindowAssertion(
        assertion, frames, window.startFrame, window.endFrame));
  }
  return results;
}

bool autotestAssertionFailuresAreFatal(
    const std::vector<AutotestAssertionResult> &results) {
  for (const AutotestAssertionResult &result : results) {
    if (result.status == AutotestAssertionStatus::Fail ||
        result.status == AutotestAssertionStatus::Invalid) {
      return true;
    }
  }
  return false;
}

} // namespace nuri::tools::autotest
