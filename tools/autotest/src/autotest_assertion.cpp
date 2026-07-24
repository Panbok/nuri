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
  if (metrics.find("gpu.scopes.opaque_ms") != metrics.end() &&
      (id == "gpu.scopes.velocity_ms" || id == "gpu.scopes.reactive_mask_ms")) {
    return false;
  }
  if (metrics.find("gpu.scopes.taa_resolve_ms") != metrics.end() &&
      id == "gpu.scopes.taa_copy_back_ms") {
    return false;
  }
  if (metrics.find("gpu.scopes.gtao_ms") != metrics.end() &&
      id == "gpu.scopes.gtao_temporal_ms") {
    return false;
  }
  if (metrics.find("gpu.scopes.ray_tracing_scene_ms") != metrics.end() &&
      (id == "gpu.scopes.ray_tracing_blas_ms" ||
       id == "gpu.scopes.ray_tracing_tlas_ms")) {
    return false;
  }
  if (metrics.find("gpu.scopes.ddgi_ms") != metrics.end() &&
      (id == "gpu.scopes.ddgi_trace_ms" || id == "gpu.scopes.ddgi_update_ms" ||
       id == "gpu.scopes.ddgi_relocate_classify_ms")) {
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

  const RenderFrameMetrics::AssetStreamingFrameMetrics &assets = metrics.assets;
  addMetric(out, "renderer.assets.cpu_completions", assets.cpuCompletions);
  addMetric(out, "renderer.assets.cpu_workers", assets.cpuWorkers);
  addMetric(out, "renderer.assets.cpu_active_worker_limit",
            assets.cpuActiveWorkerLimit);
  addMetric(out, "renderer.assets.cpu_interactive_mode",
            assets.cpuInteractiveMode);
  addMetric(out, "renderer.assets.cpu_queued_jobs", assets.cpuQueuedJobs);
  addMetric(out, "renderer.assets.cpu_running_jobs", assets.cpuRunningJobs);
  addMetric(out, "renderer.assets.cpu_running_io", assets.cpuRunningIo);
  addMetric(out, "renderer.assets.cpu_running_decode", assets.cpuRunningDecode);
  addMetric(out, "renderer.assets.cpu_running_cook", assets.cpuRunningCook);
  addMetric(out, "renderer.assets.cpu_running_transcode",
            assets.cpuRunningTranscode);
  addMetric(out, "renderer.assets.cpu_running_metadata",
            assets.cpuRunningMetadata);
  addMetric(out, "renderer.assets.dedicated_copy_queue",
            assets.dedicatedCopyQueue);
  addMetric(out, "renderer.assets.gpu_materialized", assets.gpuMaterialized);
  addMetric(out, "renderer.assets.published", assets.published);
  addMetric(out, "renderer.assets.cancelled", assets.cancelled);
  addMetric(out, "renderer.assets.failed", assets.failed);
  addMetric(out, "renderer.assets.scene_patches", assets.scenePatches);
  addMetric(out, "renderer.assets.scene_commits", assets.sceneCommits);
  addMetric(out, "renderer.assets.deferred_cpu_completions",
            assets.deferredCpuCompletions);
  addMetric(out, "renderer.assets.publication_deadline_exceeded",
            assets.publicationDeadlineExceeded);
  addMetric(out, "renderer.assets.publication_main_thread_ms",
            assets.publicationMainThreadMilliseconds);
  addMetric(out, "renderer.assets.publication_max_operation_ms",
            assets.publicationMaxOperationMilliseconds);
  addMetric(out, "renderer.assets.cpu_in_flight_bytes",
            static_cast<double>(assets.cpuInFlightBytes));
  addMetric(out, "renderer.assets.upload_bytes",
            static_cast<double>(assets.uploadBytes));
  addMetric(out, "renderer.assets.submitted_jobs",
            static_cast<double>(assets.submittedJobs));
  addMetric(out, "renderer.assets.completed_jobs",
            static_cast<double>(assets.completedJobs));
  addMetric(out, "renderer.assets.cancelled_jobs",
            static_cast<double>(assets.cancelledJobs));
  addMetric(out, "renderer.assets.rejected_jobs",
            static_cast<double>(assets.rejectedJobs));

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
  addMetric(out, "renderer.opaque.meshlet_hybrid_active",
            opaque.meshletHybridActive);
  addMetric(out, "renderer.opaque.meshlet_hybrid_classic_batches",
            opaque.meshletHybridClassicBatches);
  addMetric(out, "renderer.opaque.meshlet_hybrid_classic_instances",
            opaque.meshletHybridClassicInstances);
  addMetric(out, "renderer.opaque.meshlet_hybrid_coverage_classic_batches",
            opaque.meshletHybridCoverageClassicBatches);
  addMetric(out, "renderer.opaque.meshlet_hybrid_coverage_classic_instances",
            opaque.meshletHybridCoverageClassicInstances);
  addMetric(out, "renderer.opaque.meshlet_hybrid_meshlet_batches",
            opaque.meshletHybridMeshletBatches);
  addMetric(out, "renderer.opaque.meshlet_hybrid_meshlet_instances",
            opaque.meshletHybridMeshletInstances);
  addMetric(out, "renderer.opaque.auto_lod_active", opaque.autoLodActive);
  addMetric(out, "renderer.opaque.auto_lod_history_reset",
            opaque.autoLodHistoryReset);
  addMetric(out, "renderer.opaque.auto_lod_transitions",
            opaque.autoLodTransitions);
  addMetric(out, "renderer.opaque.auto_lod_lod0_instances",
            opaque.autoLodLod0Instances);
  addMetric(out, "renderer.opaque.auto_lod_lod1_instances",
            opaque.autoLodLod1Instances);
  addMetric(out, "renderer.opaque.meshlet_rejected_missing_feature",
            opaque.meshletRejectedMissingFeature);
  addMetric(out, "renderer.opaque.meshlet_rejected_missing_asset_data",
            opaque.meshletRejectedMissingAssetData);
  addMetric(out, "renderer.opaque.meshlet_rejected_incompatible_frame",
            opaque.meshletRejectedIncompatibleFrame);
  addMetric(out, "renderer.opaque.classic_main_draws", opaque.classicMainDraws);
  addMetric(out, "renderer.opaque.classic_alpha_masked_main_draws",
            opaque.classicAlphaMaskedMainDraws);
  addMetric(out, "renderer.opaque.meshlet_main_dispatches",
            opaque.meshletMainDispatches);
  addMetric(out, "renderer.opaque.meshlet_main_represented_items",
            opaque.meshletMainRepresentedItems);
  addMetric(out, "renderer.opaque.meshlet_alpha_masked_main_dispatches",
            opaque.meshletAlphaMaskedMainDispatches);
  addMetric(out, "renderer.opaque.meshlet_alpha_masked_main_items",
            opaque.meshletAlphaMaskedMainItems);
  addMetric(out, "renderer.opaque.msaa_depth_prepass_draws",
            opaque.msaaDepthPrepassDraws);
  addMetric(out, "renderer.opaque.msaa_depth_prepass_dispatches",
            opaque.msaaDepthPrepassDispatches);
  addMetric(out, "renderer.opaque.gtao_auxiliary_prepass_draws",
            opaque.gtaoAuxiliaryPrepassDraws);
  addMetric(out, "renderer.opaque.gtao_auxiliary_prepass_dispatches",
            opaque.gtaoAuxiliaryPrepassDispatches);
  addMetric(out, "renderer.opaque.gtao_auxiliary_writes_single_sample_depth",
            opaque.gtaoAuxiliaryWritesSingleSampleDepth);
  addMetric(out, "renderer.opaque.main_equal_readonly_draws",
            opaque.mainEqualReadOnlyDraws);
  addMetric(out, "renderer.opaque.main_equal_readonly_dispatches",
            opaque.mainEqualReadOnlyDispatches);
  addMetric(out, "renderer.opaque.main_less_write_draws",
            opaque.mainLessWriteDraws);
  addMetric(out, "renderer.opaque.main_less_write_dispatches",
            opaque.mainLessWriteDispatches);
  addMetric(out, "renderer.opaque.depth_pyramid_requested",
            opaque.depthPyramidRequested);
  addMetric(out, "renderer.opaque.depth_pyramid_active",
            opaque.depthPyramidActive);
  addMetric(out, "renderer.visibility.hiz_requested", opaque.hiZRequested);
  addMetric(out, "renderer.visibility.hiz_active", opaque.hiZActive);
  addMetric(out, "renderer.visibility.hiz_source_frame_policy",
            static_cast<uint32_t>(opaque.hiZSourceFramePolicy));

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
  addMetric(out, "renderer.visibility.meshlet_occlusion_mode",
            visibility.meshletOcclusionMode);
  addMetric(out, "renderer.visibility.meshlet_occlusion_source_frame",
            visibility.meshletOcclusionSourceFrame);
  addMetric(out, "renderer.visibility.meshlet_occlusion_source_age",
            visibility.meshletOcclusionSourceAge);
  addMetric(out, "renderer.visibility.current_frame_hiz_active",
            visibility.currentFrameHiZActive);
  addMetric(out, "renderer.visibility.meshlet_pre_task_compaction_active",
            visibility.meshletPreTaskCompactionActive);
  addMetric(out, "renderer.visibility.meshlet_pre_task_candidates_input",
            visibility.meshletPreTaskCandidatesInput);
  addMetric(out, "renderer.visibility.meshlet_pre_task_candidates_output",
            visibility.meshletPreTaskCandidatesOutput);
  addMetric(out, "renderer.visibility.meshlet_pre_task_task_groups_input",
            visibility.meshletPreTaskTaskGroupsInput);
  addMetric(out, "renderer.visibility.meshlet_pre_task_task_groups_output",
            visibility.meshletPreTaskTaskGroupsOutput);
  addMetric(out, "renderer.visibility.meshlet_pre_task_task_groups_saved",
            visibility.meshletPreTaskTaskGroupsSaved);
  addMetric(out, "renderer.visibility.meshlet_pre_task_overflow_count",
            visibility.meshletPreTaskOverflowCount);
  addMetric(out, "renderer.visibility.meshlet_pre_task_mismatch_count",
            visibility.meshletPreTaskMismatchCount);
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
  addMetric(out, "renderer.shadow.instance_remaps",
            shadow.shadowInstanceRemapCount);
  addMetric(out, "renderer.shadow.submitted_draw_items",
            shadow.submittedDrawItemCount);
  addMetric(out, "renderer.shadow.indirect_commands",
            shadow.indirectCommandCount);
  addMetric(out, "renderer.shadow.draw_packet_bytes", shadow.drawPacketBytes);
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
  addMetric(out, "renderer.shadow.filter_sample_budget",
            shadow.filterSampleBudget);
  addMetric(out, "renderer.shadow.frame_gpu_bytes", shadow.frameGpuBytes);
  addMetric(out, "renderer.shadow.sdsm_compute_passes",
            shadow.sdsmComputePassCount);
  addBytesAsMiB(out, "gpu.memory.shadow.cascade_texture_mb",
                shadow.cascadeTextureBytes);

  const RayTracingSceneFrameMetrics &rayTracing = metrics.rayTracingScene;
  addMetric(out, "renderer.ray_tracing.static_instances",
            rayTracing.staticInstances);
  addMetric(out, "renderer.ray_tracing.dynamic_instances",
            rayTracing.dynamicInstances);
  addMetric(out, "renderer.ray_tracing.excluded_dynamic_instances",
            rayTracing.excludedDynamicInstances);
  addMetric(out, "renderer.ray_tracing.static_blas_count",
            rayTracing.staticBlasCount);
  addMetric(out, "renderer.ray_tracing.dynamic_blas_count",
            rayTracing.dynamicBlasCount);
  addMetric(out, "renderer.ray_tracing.tlas_count", rayTracing.tlasCount);
  addMetric(out, "renderer.ray_tracing.unique_static_geometry",
            rayTracing.uniqueStaticGeometry);
  addMetric(out, "renderer.ray_tracing.geometry_records",
            rayTracing.geometryRecords);
  addMetric(out, "renderer.ray_tracing.triangles", rayTracing.triangles);
  addMetric(out, "renderer.ray_tracing.queued_blas_builds",
            rayTracing.queuedBlasBuilds);
  addMetric(out, "renderer.ray_tracing.decoded_vertices",
            rayTracing.decodedVertices);
  addMetric(out, "renderer.ray_tracing.decode_dispatches",
            rayTracing.decodeDispatches);
  addMetric(out, "renderer.ray_tracing.blas_builds", rayTracing.blasBuilds);
  addMetric(out, "renderer.ray_tracing.tlas_builds", rayTracing.tlasBuilds);
  addMetric(out, "renderer.ray_tracing.tlas_updates", rayTracing.tlasUpdates);
  addMetric(out, "renderer.ray_tracing.dynamic_blas_updates",
            rayTracing.dynamicBlasUpdates);
  addMetric(out, "renderer.ray_tracing.dynamic_vertex_dispatches",
            rayTracing.dynamicVertexDispatches);
  addMetric(out, "renderer.ray_tracing.readiness",
            static_cast<uint32_t>(rayTracing.readiness));
  addMetric(out, "renderer.ray_tracing.consumed_rebuild_epoch",
            rayTracing.consumedRebuildEpoch);
  addBytesAsMiB(out, "gpu.memory.ray_tracing.decoded_positions_mb",
                rayTracing.decodedPositionBytes);
  addBytesAsMiB(out, "gpu.memory.ray_tracing.tables_mb", rayTracing.tableBytes);
  addBytesAsMiB(out, "gpu.memory.ray_tracing.blas_mb",
                rayTracing.blasAllocationBytes);
  addBytesAsMiB(out, "gpu.memory.ray_tracing.tlas_mb",
                rayTracing.tlasAllocationBytes);
  addBytesAsMiB(out, "gpu.memory.ray_tracing.as_scratch_high_water_mb",
                rayTracing.asScratchHighWaterBytes);
  addMetric(out, "renderer.ray_tracing.direct_binding_pool_high_water",
            rayTracing.directBindingPoolHighWater);
  addMetric(out, "gpu.scopes.ray_tracing_scene_ms", rayTracing.gpuTimeMs);
  addMetric(out, "gpu.scopes.ray_tracing_blas_ms", rayTracing.blasGpuTimeMs);
  addMetric(out, "gpu.scopes.ray_tracing_tlas_ms", rayTracing.tlasGpuTimeMs);
  addMetric(out, "renderer.ray_tracing.gpu_timing_available",
            rayTracing.gpuTimingAvailable);

  const DDGIFrameMetrics &ddgi = metrics.ddgi;
  addMetric(out, "renderer.ddgi.requested", ddgi.requested);
  addMetric(out, "renderer.ddgi.active", ddgi.active);
  addMetric(out, "renderer.ddgi.active_volumes", ddgi.activeVolumes);
  addMetric(out, "renderer.ddgi.ready_volumes", ddgi.readyVolumes);
  addMetric(out, "renderer.ddgi.total_probes", ddgi.totalProbes);
  addMetric(out, "renderer.ddgi.vigilant_probes", ddgi.vigilantProbes);
  addMetric(out, "renderer.ddgi.uninitialized_probes",
            ddgi.uninitializedProbes);
  addMetric(out, "renderer.ddgi.off_probes", ddgi.offProbes);
  addMetric(out, "renderer.ddgi.sleeping_probes", ddgi.sleepingProbes);
  addMetric(out, "renderer.ddgi.newly_awake_probes", ddgi.newlyAwakeProbes);
  addMetric(out, "renderer.ddgi.awake_probes", ddgi.awakeProbes);
  addMetric(out, "renderer.ddgi.newly_vigilant_probes",
            ddgi.newlyVigilantProbes);
  addMetric(out, "renderer.ddgi.relocated_probes", ddgi.relocatedProbes);
  addMetric(out, "renderer.ddgi.probe_state_readback_available",
            ddgi.probeStateReadbackAvailable);
  addMetric(out, "renderer.ddgi.probe_state_readback_source_frame",
            ddgi.probeStateReadbackSourceFrame);
  addMetric(out, "renderer.ddgi.probe_state_readback_stale_frames",
            ddgi.probeStateReadbackStaleFrames);
  addMetric(out, "renderer.ddgi.max_relocation", ddgi.maxRelocation);
  addMetric(out, "renderer.ddgi.updated_probes", ddgi.updatedProbes);
  addMetric(out, "renderer.ddgi.primary_queries", ddgi.primaryQueries);
  addMetric(out, "renderer.ddgi.classification_probe_updates",
            ddgi.classificationProbeUpdates);
  addMetric(out, "renderer.ddgi.classification_primary_queries",
            ddgi.classificationPrimaryQueries);
  addMetric(out, "renderer.ddgi.irradiance_primary_queries",
            ddgi.irradiancePrimaryQueries);
  addMetric(out, "renderer.ddgi.primary_queries_issued",
            ddgi.primaryQueriesIssued);
  addMetric(out, "renderer.ddgi.trace_counter_source_frame",
            ddgi.traceCounterSourceFrame);
  addMetric(out, "renderer.ddgi.trace_counter_stale_frames",
            ddgi.traceCounterStaleFrames);
  addMetric(out, "renderer.ddgi.readback_waits", ddgi.readbackWaits);
  addMetric(out, "renderer.ddgi.readback_pending_slots",
            ddgi.readbackPendingSlots);
  addMetric(out, "renderer.ddgi.readback_dropped_samples",
            ddgi.readbackDroppedSamples);
  addMetric(out, "renderer.ddgi.readback_oldest_pending_age",
            ddgi.readbackOldestPendingAge);
  addMetric(out, "renderer.ddgi.readback_blocking_fallbacks",
            ddgi.readbackBlockingFallbacks);
  addMetric(out, "renderer.ddgi.readback_generation_mismatches",
            ddgi.readbackGenerationMismatches);
  addMetric(out, "renderer.ddgi.readback_early_reuse_attempts",
            ddgi.readbackEarlyReuseAttempts);
  addMetric(out, "renderer.ddgi.readback_copy_bytes", ddgi.readbackCopyBytes);
  addMetric(out, "renderer.ddgi.readback_per_slot_bytes",
            ddgi.readbackPerSlotBytes);
  addMetric(out, "renderer.ddgi.readback_ring_bytes", ddgi.readbackRingBytes);
  addMetric(out, "renderer.ddgi.secondary_queries_reserved",
            ddgi.secondaryQueriesReserved);
  addMetric(out, "renderer.ddgi.secondary_queries_unused",
            ddgi.secondaryQueriesUnused);
  addMetric(out, "renderer.ddgi.secondary_query_budget_overflows",
            ddgi.secondaryQueryBudgetOverflows);
  addMetric(out, "renderer.ddgi.secondary_queries", ddgi.secondaryQueries);
  addMetric(out, "renderer.ddgi.directional_secondary_queries",
            ddgi.directionalSecondaryQueries);
  addMetric(out, "renderer.ddgi.local_secondary_queries",
            ddgi.localSecondaryQueries);
  addMetric(out, "renderer.ddgi.total_queries_issued",
            static_cast<uint64_t>(ddgi.primaryQueriesIssued) +
                ddgi.secondaryQueries);
  addMetric(out, "renderer.ddgi.primary_candidate_intersections",
            ddgi.primaryCandidateIntersections);
  addMetric(out, "renderer.ddgi.secondary_candidate_intersections",
            ddgi.secondaryCandidateIntersections);
  addMetric(out, "renderer.ddgi.alpha_candidate_rejections",
            ddgi.alphaCandidateRejections);
  addMetric(out, "renderer.ddgi.backface_candidate_rejections",
            ddgi.backfaceCandidateRejections);
  addMetric(out, "renderer.ddgi.candidate_overflows", ddgi.candidateOverflows);
  addMetric(out, "renderer.ddgi.local_light_truncations",
            ddgi.localLightTruncations);
  addMetric(out, "renderer.ddgi.non_finite_radiance_rejects",
            ddgi.nonFiniteRadianceRejects);
  addMetric(out, "renderer.ddgi.emissive_radiance_clamps",
            ddgi.emissiveRadianceClamps);
  addMetric(out, "renderer.ddgi.direct_radiance_clamps",
            ddgi.directRadianceClamps);
  addMetric(out, "renderer.ddgi.sky_radiance_clamps", ddgi.skyRadianceClamps);
  addMetric(out, "renderer.ddgi.multi_bounce_radiance_clamps",
            ddgi.multiBounceRadianceClamps);
  addMetric(out, "renderer.ddgi.final_radiance_clamps",
            ddgi.finalRadianceClamps);
  addMetric(out, "renderer.ddgi.diagnostic_counters_enabled",
            ddgi.diagnosticCountersEnabled);
  addMetric(out, "renderer.ddgi.surface_gather_architecture",
            static_cast<uint32_t>(ddgi.surfaceGatherArchitecture));
  addMetric(out, "renderer.ddgi.surface_gather_width", ddgi.surfaceGatherWidth);
  addMetric(out, "renderer.ddgi.surface_gather_height",
            ddgi.surfaceGatherHeight);
  addMetric(out, "renderer.ddgi.surface_gather_max_candidate_volumes",
            ddgi.surfaceGatherMaxCandidateVolumes);
  addMetric(out, "renderer.ddgi.surface_gather_max_sampled_volumes",
            ddgi.surfaceGatherMaxSampledVolumes);
  addMetric(out, "renderer.ddgi.surface_gather_max_state_loads_per_pixel",
            ddgi.surfaceGatherMaxStateLoadsPerPixel);
  addMetric(out, "renderer.ddgi.surface_gather_max_atlas_samples_per_pixel",
            ddgi.surfaceGatherMaxAtlasSamplesPerPixel);
  addMetric(out, "renderer.ddgi.ray_query_capacity", ddgi.rayQueryCapacity);
  addMetric(out, "renderer.ddgi.probe_update_capacity",
            ddgi.probeUpdateCapacity);
  addMetric(out, "renderer.ddgi.requested_probe_update_capacity",
            ddgi.requestedProbeUpdateCapacity);
  addMetric(out, "renderer.ddgi.effective_probe_update_capacity",
            ddgi.effectiveProbeUpdateCapacity);
  addMetric(out, "renderer.ddgi.startup_phase",
            static_cast<uint32_t>(ddgi.startupPhase));
  addMetric(out, "renderer.ddgi.sky_remainder_over_threshold_percentage",
            ddgi.skyRemainderOverThresholdPercentage);
  addMetric(out, "renderer.ddgi.reset_count", ddgi.resetCount);
  addMetric(out, "renderer.ddgi.scroll_count", ddgi.scrollCount);
  addMetric(out, "renderer.ddgi.invalidated_probes", ddgi.invalidatedProbes);
  addMetric(out, "renderer.ddgi.failed_volumes", ddgi.failedVolumes);
  addMetric(out, "renderer.ddgi.effective_volumes", ddgi.effectiveVolumes);
  addMetric(out, "renderer.ddgi.authored_volumes", ddgi.authoredVolumes);
  addMetric(out, "renderer.ddgi.generated_volumes", ddgi.generatedVolumes);
  addMetric(out, "renderer.ddgi.redundant_authored_volumes",
            ddgi.redundantAuthoredVolumes);
  addMetric(out, "renderer.ddgi.redundant_authored_probes",
            ddgi.redundantAuthoredProbes);
  addBytesAsMiB(out, "gpu.memory.ddgi.redundant_authored_mb",
                ddgi.redundantAuthoredBytes);
  addMetric(out, "renderer.ddgi.coverage_mode", ddgi.coverageMode);
  addMetric(out, "renderer.ddgi.coverage_status",
            static_cast<uint32_t>(ddgi.coverageStatus));
  addMetric(out, "renderer.ddgi.coverage_error",
            static_cast<uint32_t>(ddgi.coverageError));
  addMetric(out, "renderer.ddgi.limiting_constraint",
            static_cast<uint32_t>(ddgi.limitingConstraint));
  addMetric(out, "renderer.ddgi.requested_half_extent_x",
            ddgi.requestedCoverageHalfExtents.x);
  addMetric(out, "renderer.ddgi.requested_half_extent_y",
            ddgi.requestedCoverageHalfExtents.y);
  addMetric(out, "renderer.ddgi.requested_half_extent_z",
            ddgi.requestedCoverageHalfExtents.z);
  addMetric(out, "renderer.ddgi.achieved_half_extent_x",
            ddgi.achievedCoverageHalfExtents.x);
  addMetric(out, "renderer.ddgi.achieved_half_extent_y",
            ddgi.achievedCoverageHalfExtents.y);
  addMetric(out, "renderer.ddgi.achieved_half_extent_z",
            ddgi.achievedCoverageHalfExtents.z);
  addMetric(out, "renderer.ddgi.scene_coverage_ratio", ddgi.sceneCoverageRatio);
  addMetric(out, "renderer.ddgi.coverage_resolve_cpu_ms",
            ddgi.coverageResolveCpuTimeMs);
  addMetric(out, "renderer.ddgi.diagnostic_sample_count",
            ddgi.diagnosticSampleCount);
  addMetric(out, "renderer.ddgi.uncovered_diagnostic_samples",
            ddgi.uncoveredDiagnosticSamples);
  addMetric(out, "renderer.ddgi.sky_remainder_samples",
            ddgi.skyRemainderSamples);
  addMetric(out, "renderer.ddgi.diagnostic_samples_available",
            ddgi.diagnosticSamplesAvailable);
  addMetric(out, "renderer.ddgi.dirty_regions_produced",
            ddgi.dirtyRegionsProduced);
  addMetric(out, "renderer.ddgi.dirty_regions_merged", ddgi.dirtyRegionsMerged);
  addMetric(out, "renderer.ddgi.dirty_regions_overflowed",
            ddgi.dirtyRegionsOverflowed);
  addMetric(out, "renderer.ddgi.dirty_regions_pending",
            ddgi.dirtyRegionsPending);
  addMetric(out, "renderer.ddgi.dirty_probes_affected",
            ddgi.dirtyProbesAffected);
  addMetric(out, "renderer.ddgi.classification_fallbacks",
            ddgi.classificationFallbacks);
  addMetric(out, "renderer.ddgi.classification_overflows",
            ddgi.classificationOverflows);
  addMetric(out, "renderer.ddgi.volume_failure_reason",
            static_cast<uint32_t>(ddgi.volumeFailureReason));
  addMetric(out, "renderer.ddgi.history_ready", ddgi.historyReady);
  addMetric(out, "renderer.ddgi.irradiance_response_remaining",
            ddgi.irradianceResponseRemaining);
  addMetric(out, "renderer.ddgi.distance_response_remaining",
            ddgi.distanceResponseRemaining);
  addMetric(out, "renderer.ddgi.inspection_available",
            ddgi.inspectionAvailable);
  addMetric(out, "renderer.ddgi.inspection_valid", ddgi.inspectionValid);
  addMetric(out, "renderer.ddgi.inspection_ray_count", ddgi.inspectionRayCount);
  addMetric(out, "renderer.ddgi.inspection_hit_count", ddgi.inspectionHitCount);
  addMetric(out, "renderer.ddgi.inspection_miss_count",
            ddgi.inspectionMissCount);
  addMetric(out, "renderer.ddgi.inspection_candidate_overflows",
            ddgi.inspectionCandidateOverflows);
  addMetric(out, "renderer.ddgi.inspection_event_overflows",
            ddgi.inspectionEventOverflows);
  addMetric(out, "renderer.ddgi.sky_fallback_active", ddgi.skyFallbackActive);
  addMetric(out, "renderer.ddgi.submitted_sequence", ddgi.submittedSequence);
  addMetric(out, "renderer.ddgi.layout_generation", ddgi.layoutGeneration);
  addMetric(out, "renderer.ddgi.resource_generation", ddgi.resourceGeneration);
  addMetric(out, "renderer.ddgi.device_epoch", ddgi.deviceEpoch);
  addMetric(out, "renderer.ddgi.consumed_reset_epoch", ddgi.consumedResetEpoch);
  addMetric(out, "renderer.ddgi.consumed_force_update_epoch",
            ddgi.consumedForceUpdateEpoch);
  addMetric(out, "renderer.ddgi.fallback_reason",
            static_cast<uint32_t>(ddgi.fallbackReason));
  addMetric(out, "renderer.ddgi.debug_view",
            static_cast<uint32_t>(ddgi.debugView));
  addMetric(out, "gpu.scopes.ddgi_ms", ddgi.gpuTimeMs);
  addMetric(out, "gpu.scopes.ddgi_trace_ms", ddgi.traceGpuTimeMs);
  addMetric(out, "gpu.scopes.ddgi_update_ms", ddgi.updateGpuTimeMs);
  addMetric(out, "gpu.scopes.ddgi_relocate_classify_ms",
            ddgi.relocateClassifyGpuTimeMs);
  addMetric(out, "renderer.ddgi.gpu_timing_available", ddgi.gpuTimingAvailable);
  addBytesAsMiB(out, "gpu.memory.ddgi.persistent_mb", ddgi.persistentBytes);
  addBytesAsMiB(out, "gpu.memory.ddgi.frame_batch_mb", ddgi.frameBatchBytes);
  addBytesAsMiB(out, "gpu.memory.ddgi.committed_atlas_mb",
                ddgi.committedAtlasBytes);
  addBytesAsMiB(out, "gpu.memory.ddgi.pending_atlas_mb",
                ddgi.pendingAtlasBytes);
  addBytesAsMiB(out, "gpu.memory.ddgi.peak_atlas_mb", ddgi.peakAtlasBytes);
  for (size_t volumeIndex = 0u; volumeIndex < ddgi.volumes.size();
       ++volumeIndex) {
    const DDGIVolumeFrameMetrics &volume = ddgi.volumes[volumeIndex];
    const std::string prefix =
        "renderer.ddgi.volume" + std::to_string(volumeIndex) + ".";
    const auto addVolumeMetric = [&](std::string_view suffix, double value) {
      out[prefix + std::string(suffix)] = value;
    };
    addVolumeMetric("active", volume.active);
    addVolumeMetric("effective_key_hash",
                    static_cast<double>(volume.effectiveKeyHash));
    addVolumeMetric("effective_kind", volume.effectiveKind);
    addVolumeMetric("tier", volume.tier);
    addVolumeMetric("cascade_index", volume.cascadeIndex);
    addVolumeMetric("total_probes", volume.totalProbes);
    addVolumeMetric("initialized_probes", volume.initializedProbes);
    addVolumeMetric("shading_enabled_probes", volume.shadingEnabledProbes);
    addVolumeMetric("invalid_probes", volume.invalidProbes);
    addVolumeMetric("newly_exposed_probes", volume.newlyExposedProbes);
    addVolumeMetric("updates", volume.updates);
    addVolumeMetric("primary_queries", volume.primaryQueries);
    addVolumeMetric("primary_queries_issued", volume.primaryQueriesIssued);
    addVolumeMetric("secondary_queries", volume.secondaryQueries);
    addVolumeMetric("update_age_median", volume.updateAgeMedian);
    addVolumeMetric("update_age_p95", volume.updateAgeP95);
    addVolumeMetric("update_age_maximum", volume.updateAgeMaximum);
    addVolumeMetric("scheduled_quota", volume.scheduledQuota);
    addVolumeMetric("used_quota", volume.usedQuota);
    addVolumeMetric("deficit", static_cast<double>(volume.deficit));
    addVolumeMetric("starvation_frames", volume.starvationFrames);
    addVolumeMetric("estimated_full_refresh_frames",
                    volume.estimatedFullRefreshFrames);
    addVolumeMetric("persistent_mb",
                    static_cast<double>(volume.persistentBytes) /
                        (1024.0 * 1024.0));
    addVolumeMetric("unique_coverage_percentage",
                    volume.uniqueCoveragePercentage);
    addVolumeMetric("redundant_coverage", volume.redundantCoverage);
    addVolumeMetric("interior_half_extent_x", volume.interiorHalfExtents.x);
    addVolumeMetric("interior_half_extent_y", volume.interiorHalfExtents.y);
    addVolumeMetric("interior_half_extent_z", volume.interiorHalfExtents.z);
    addVolumeMetric("fade_start_half_extent_x", volume.fadeStartHalfExtents.x);
    addVolumeMetric("fade_start_half_extent_y", volume.fadeStartHalfExtents.y);
    addVolumeMetric("fade_start_half_extent_z", volume.fadeStartHalfExtents.z);
    addVolumeMetric("fade_end_half_extent_x", volume.fadeEndHalfExtents.x);
    addVolumeMetric("fade_end_half_extent_y", volume.fadeEndHalfExtents.y);
    addVolumeMetric("fade_end_half_extent_z", volume.fadeEndHalfExtents.z);
    addVolumeMetric("camera_cell_x", volume.cameraCell.x);
    addVolumeMetric("camera_cell_y", volume.cameraCell.y);
    addVolumeMetric("camera_cell_z", volume.cameraCell.z);
    addVolumeMetric("history_ready_percentage", volume.historyReadyPercentage);
    addVolumeMetric("coverage_ready_percentage",
                    volume.coverageReadyPercentage);
    addVolumeMetric("confidence", volume.confidence);
  }

  const AntiAliasingFrameMetrics &aa = metrics.antiAliasing;
  const PostAAPlan &postAAPlan = aa.postAAPlan;
  const PostAAFrameFacts &postAA = aa.postAA;
  addBoolMetric(out, "renderer.aa.post_aa_requested", postAAPlan.requested);
  addBoolMetric(out, "renderer.aa.post_aa_resolved_active", postAAPlan.active);
  addMetric(out, "renderer.aa.post_aa_inactive_reason",
            static_cast<uint32_t>(postAAPlan.inactiveReason));
  addMetric(out, "renderer.aa.post_aa_specular_algorithm",
            static_cast<uint32_t>(postAAPlan.specular));
  addMetric(out, "renderer.aa.post_aa_spatial_algorithm",
            static_cast<uint32_t>(postAAPlan.spatial));
  addMetric(out, "renderer.aa.post_aa_material_variance_scale",
            postAAPlan.materialVarianceScale);
  addMetric(out, "renderer.aa.post_aa_geometric_variance_scale",
            postAAPlan.geometricVarianceScale);
  addMetric(out, "renderer.aa.post_aa_max_slope_variance",
            postAAPlan.maxSlopeVariance);
  addBoolMetric(out, "renderer.aa.post_aa_specular_selected",
                postAA.specularSelected);
  addBoolMetric(out, "renderer.aa.post_aa_smaa_planned", postAA.smaaPlanned);
  addBoolMetric(out, "renderer.aa.post_aa_smaa_submitted",
                postAA.smaaSubmitted);
  addMetric(out, "renderer.aa.post_aa_smaa_submitted_passes",
            postAA.smaaSubmittedPassCount);
  addBoolMetric(out, "renderer.aa.post_aa_smaa_completed",
                postAA.smaaCompleted);
  if (postAA.smaaCompletedSourceFrameIndex != kInvalidFrameIndex) {
    addMetric(out, "renderer.aa.post_aa_smaa_completed_source_frame",
              postAA.smaaCompletedSourceFrameIndex);
  }
  addMetric(out, "renderer.aa.post_aa_degradation_mask",
            static_cast<uint32_t>(postAA.degradation));
  addMetric(out, "renderer.aa.resolved_material_specular_aa",
            static_cast<uint32_t>(postAAPlan.resolvedMaterialSpecularAA));
  addMetric(out, "renderer.aa.debug_view",
            static_cast<uint32_t>(postAAPlan.debugView));
  addMetric(out, "renderer.aa.specular_aa_debug_override",
            static_cast<uint32_t>(postAAPlan.specularAADebugOverride));
  addMetric(out, "renderer.aa.normal_variance_contract_materials_live",
            aa.normalVarianceContractMaterialsLive);
  addMetric(out, "renderer.aa.normal_variance_contract_textures_live",
            aa.normalVarianceContractTexturesLive);
  addMetric(out, "renderer.aa.normal_variance_unavailable_slots_live",
            aa.normalVarianceUnavailableSlotsLive);
  addBytesAsMiB(out, "gpu.memory.aa.normal_variance_contract_textures_mb",
                aa.normalVarianceContractTextureBytesLive);
  addBoolMetric(out, "renderer.aa.history_valid", aa.historyValid);
  addBoolMetric(out, "renderer.aa.temporal_data_valid", aa.temporalDataValid);
  addMetric(out, "renderer.aa.history_reset_count", aa.historyResetCount);
  addMetric(out, "renderer.aa.frames_since_history_reset",
            aa.framesSinceHistoryReset);
  addMetric(out, "renderer.aa.camera_position_delta", aa.cameraPositionDelta);
  addMetric(out, "renderer.aa.camera_direction_delta", aa.cameraDirectionDelta);
  addMetric(out, "renderer.aa.jitter_delta_magnitude", aa.jitterDeltaMagnitude);
  addMetric(out, "renderer.aa.motion_vector_textures",
            aa.motionVectorTextureCount);
  addMetric(out, "renderer.aa.motion_class_textures",
            aa.motionClassTextureCount);
  addMetric(out, "renderer.aa.history_color_textures",
            aa.historyColorTextureCount);
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
  addMetric(out, "renderer.aa.msaa_color_resolves", aa.msaaColorResolveCount);
  addMetric(out, "renderer.aa.msaa_depth_resolves", aa.msaaDepthResolveCount);
  addMetric(out, "renderer.aa.msaa_resolved_sample_count",
            aa.msaaResolvedSampleCount);
  addMetric(out, "renderer.aa.msaa_sample_count", aa.msaaSampleCount);
  addBoolMetric(out, "renderer.aa.msaa_sample4_color_supported",
                aa.msaaSample4ColorSupported);
  addBoolMetric(out, "renderer.aa.msaa_sample4_depth_supported",
                aa.msaaSample4DepthSupported);
  addBoolMetric(out, "renderer.aa.msaa_sample8_color_supported",
                aa.msaaSample8ColorSupported);
  addBoolMetric(out, "renderer.aa.msaa_sample8_depth_supported",
                aa.msaaSample8DepthSupported);
  addBoolMetric(out, "renderer.aa.msaa_depth_resolve_min_supported",
                aa.msaaDepthResolveMinSupported);
  addBoolMetric(out, "renderer.aa.msaa_alpha_to_coverage_supported",
                aa.msaaAlphaToCoverageSupported);
  addBoolMetric(out, "renderer.aa.msaa_sample_rate_shading_supported",
                aa.msaaSampleRateShadingSupported);
  addBoolMetric(out, "renderer.aa.msaa_alpha_to_coverage_active",
                aa.msaaAlphaToCoverageEnabled);
  addBoolMetric(out, "renderer.aa.msaa_sample_shading_active",
                aa.msaaSampleShadingEnabled);
  addBoolMetric(out, "renderer.aa.msaa_alpha_coverage_requested",
                aa.msaaAlphaCoverageRequested);
  addBoolMetric(out, "renderer.aa.msaa_spatial_cleanup_requested",
                aa.msaaSpatialCleanupRequested);
  addBoolMetric(out, "renderer.aa.msaa_spatial_cleanup_active",
                aa.msaaSpatialCleanupActive);
  addMetric(out, "renderer.aa.msaa_unsupported_reason",
            static_cast<uint32_t>(aa.msaaUnsupportedReason));
  addMetric(out, "renderer.aa.msaa_alpha_coverage_policy",
            static_cast<uint32_t>(aa.msaaAlphaCoveragePolicy));
  addMetric(out, "renderer.aa.msaa_transparency_policy",
            static_cast<uint32_t>(aa.msaaTransparencyPolicy));
  addMetric(out, "renderer.aa.msaa_resolve_placement",
            static_cast<uint32_t>(aa.msaaResolvePlacement));
  addMetric(out, "renderer.aa.msaa_main_color_format",
            static_cast<uint32_t>(aa.msaaMainColorFormat));
  addMetric(out, "renderer.aa.msaa_main_depth_format",
            static_cast<uint32_t>(aa.msaaMainDepthFormat));
  addMetric(out, "renderer.aa.msaa_main_attachment_sample_count",
            aa.msaaMainAttachmentSampleCount);
  addMetric(out, "renderer.aa.msaa_extent_width", aa.msaaExtentWidth);
  addMetric(out, "renderer.aa.msaa_extent_height", aa.msaaExtentHeight);
  addMetric(out, "renderer.aa.msaa_color_texel_bytes", aa.msaaColorTexelBytes);
  addMetric(out, "renderer.aa.msaa_depth_texel_bytes", aa.msaaDepthTexelBytes);
  addMetric(out, "renderer.aa.msaa_traffic_formula_version",
            aa.msaaTrafficFormulaVersion);
  addBytesAsMiB(out, "gpu.memory.aa.motion_vector_total_mb",
                aa.motionVectorTotalBytes);
  addBytesAsMiB(out, "gpu.memory.aa.motion_class_total_mb",
                aa.motionClassTotalBytes);
  addBytesAsMiB(out, "gpu.memory.aa.history_color_total_mb",
                aa.historyColorTotalBytes);
  addBytesAsMiB(out, "gpu.memory.aa.reactive_mask_total_mb",
                aa.reactiveMaskTotalBytes);
  addBytesAsMiB(out, "gpu.memory.aa.spatial_aa_total_mb",
                aa.spatialAATotalBytes);
  addBytesAsMiB(out, "gpu.memory.aa.msaa_total_mb", aa.msaaTotalBytes);
  addBytesAsMiB(out, "gpu.memory.aa.msaa_active_color_mb",
                aa.msaaColorTextureBytes);
  addBytesAsMiB(out, "gpu.memory.aa.msaa_active_depth_mb",
                aa.msaaDepthTextureBytes);
  addBytesAsMiB(out, "gpu.memory.aa.msaa_ring_color_mb", aa.msaaRingColorBytes);
  addBytesAsMiB(out, "gpu.memory.aa.msaa_ring_depth_mb", aa.msaaRingDepthBytes);
  addBytesAsMiB(out, "gpu.traffic.aa.msaa_resolve_read_estimated_mb",
                aa.msaaResolveReadEstimateBytes);
  addBytesAsMiB(out, "gpu.traffic.aa.msaa_resolve_write_estimated_mb",
                aa.msaaResolveWriteEstimateBytes);
  addMetric(out, "renderer.aa.msaa_color_textures", aa.msaaColorTextureCount);
  addMetric(out, "renderer.aa.msaa_depth_textures", aa.msaaDepthTextureCount);
  addMetric(out, "renderer.aa.msaa_ring_slots", aa.msaaRingSlots);
  addMetric(out, "renderer.aa.msaa_color_allocations",
            aa.msaaColorAllocationCount);
  addMetric(out, "renderer.aa.msaa_color_reallocations",
            aa.msaaColorReallocationCount);
  addMetric(out, "renderer.aa.msaa_depth_allocations",
            aa.msaaDepthAllocationCount);
  addMetric(out, "renderer.aa.msaa_depth_reallocations",
            aa.msaaDepthReallocationCount);
  addMetric(out, "renderer.aa.spatial_aa_allocations",
            aa.spatialAAAllocationCount);
  addMetric(out, "renderer.aa.spatial_aa_reallocations",
            aa.spatialAAReallocationCount);

  const AmbientOcclusionFrameMetrics &ao = metrics.ambientOcclusion;
  addBoolMetric(out, "renderer.ao.enabled", ao.enabled);
  addBoolMetric(out, "renderer.ao.active", ao.active);
  addMetric(out, "renderer.ao.normal_prepass_draws", ao.normalPrepassDraws);
  addMetric(out, "renderer.ao.depth_prefilter_passes",
            ao.depthPrefilterPassCount);
  addMetric(out, "renderer.ao.main_passes", ao.mainPassCount);
  addMetric(out, "renderer.ao.temporal_passes", ao.temporalPassCount);
  addBoolMetric(out, "renderer.ao.temporal_motion_class_consumed",
                ao.temporalMotionClassConsumed);
  addBoolMetric(out, "renderer.ao.temporal_previous_depth_consumed",
                ao.temporalPreviousDepthConsumed);
  addMetric(out, "renderer.ao.texture_count", ao.textureCount);
  addBytesAsMiB(out, "gpu.memory.ao.total_texture_mb", ao.totalTextureBytes);

  const HDRPostProcessFrameMetrics &hdr = metrics.hdrPostProcess;
  addBoolMetric(out, "renderer.hdr.bloom_enabled", hdr.bloomEnabled);
  addBoolMetric(out, "renderer.hdr.bloom_active", hdr.bloomActive);
  addMetric(out, "renderer.hdr.bloom_passes", hdr.bloomPassCount);
  addMetric(out, "renderer.hdr.luminance_passes", hdr.luminancePassCount);
  addMetric(out, "renderer.hdr.adaptation_passes", hdr.adaptationPassCount);
  addBoolMetric(out, "renderer.hdr.adaptation_enabled", hdr.adaptationEnabled);
  addBoolMetric(out, "renderer.hdr.adaptation_active", hdr.adaptationActive);
  addMetric(out, "renderer.hdr.texture_count", hdr.textureCount);
  addMetric(out, "renderer.hdr.adapted_exposure_ev", hdr.adaptedExposureEv);
  addMetric(out, "renderer.hdr.automatic_exposure_ev", hdr.automaticExposureEv);
  addMetric(out, "renderer.hdr.exposure_target_ev", hdr.exposureTargetEv);
  addMetric(out, "renderer.hdr.exposure_metered_luminance",
            hdr.exposureMeteredLuminance);
  addMetric(out, "renderer.hdr.effective_exposure_ev", hdr.effectiveExposureEv);
  addMetric(out, "renderer.hdr.exposure_invalid_sample_fraction",
            hdr.exposureInvalidSampleFraction);
  addMetric(out, "renderer.hdr.exposure_telemetry_available",
            hdr.exposureTelemetryAvailable);
  addMetric(out, "renderer.hdr.exposure_telemetry_source_frame",
            hdr.exposureTelemetrySourceFrameIndex);
  addMetric(out, "renderer.hdr.exposure_telemetry_stale_frames",
            hdr.exposureTelemetryStaleFrames);
  addMetric(out, "renderer.hdr.exposure_telemetry_pending_slots",
            hdr.exposureTelemetryPendingSlots);
  addMetric(out, "renderer.hdr.exposure_telemetry_dropped_samples",
            hdr.exposureTelemetryDroppedSamples);
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
  applyGpuScope(frames, report, GpuTimingScope::WholeFrame,
                report.wholeFrameSourceFrameIndex, "gpu.frame_ms",
                report.wholeFrameTimeMs);
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
  applyGpuScope(frames, report, GpuTimingScope::OpaqueDepth,
                report.opaqueDepthSourceFrameIndex,
                "gpu.scopes.opaque_depth_ms", report.opaqueDepthTimeMs);
  applyGpuScope(frames, report, GpuTimingScope::OpaqueNormal,
                report.opaqueNormalSourceFrameIndex,
                "gpu.scopes.opaque_normal_ms", report.opaqueNormalTimeMs);
  applyGpuScope(frames, report, GpuTimingScope::OpaqueMain,
                report.opaqueMainSourceFrameIndex, "gpu.scopes.opaque_main_ms",
                report.opaqueMainTimeMs);
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
  applyGpuScope(frames, report, GpuTimingScope::Velocity,
                report.velocitySourceFrameIndex, "gpu.scopes.velocity_ms",
                report.velocityTimeMs);
  applyGpuScope(frames, report, GpuTimingScope::ReactiveMask,
                report.reactiveMaskSourceFrameIndex,
                "gpu.scopes.reactive_mask_ms", report.reactiveMaskTimeMs);
  applyGpuScope(frames, report, GpuTimingScope::TemporalAACopyBack,
                report.temporalAACopyBackSourceFrameIndex,
                "gpu.scopes.taa_copy_back_ms", report.temporalAACopyBackTimeMs);
  applyGpuScope(frames, report, GpuTimingScope::GTAOTemporal,
                report.gtaoTemporalSourceFrameIndex,
                "gpu.scopes.gtao_temporal_ms", report.gtaoTemporalTimeMs);
  applyGpuScope(frames, report, GpuTimingScope::RayTracingScene,
                report.rayTracingSceneSourceFrameIndex,
                "gpu.scopes.ray_tracing_scene_ms",
                report.rayTracingSceneTimeMs);
  applyGpuScope(frames, report, GpuTimingScope::RayTracingBLAS,
                report.rayTracingBlasSourceFrameIndex,
                "gpu.scopes.ray_tracing_blas_ms", report.rayTracingBlasTimeMs);
  applyGpuScope(frames, report, GpuTimingScope::RayTracingTLAS,
                report.rayTracingTlasSourceFrameIndex,
                "gpu.scopes.ray_tracing_tlas_ms", report.rayTracingTlasTimeMs);
  applyGpuScope(frames, report, GpuTimingScope::DDGI,
                report.ddgiSourceFrameIndex, "gpu.scopes.ddgi_ms",
                report.ddgiTimeMs);
  applyGpuScope(frames, report, GpuTimingScope::DDGITrace,
                report.ddgiTraceSourceFrameIndex, "gpu.scopes.ddgi_trace_ms",
                report.ddgiTraceTimeMs);
  applyGpuScope(frames, report, GpuTimingScope::DDGIUpdate,
                report.ddgiUpdateSourceFrameIndex, "gpu.scopes.ddgi_update_ms",
                report.ddgiUpdateTimeMs);
  applyGpuScope(frames, report, GpuTimingScope::DDGIRelocateClassify,
                report.ddgiRelocateClassifySourceFrameIndex,
                "gpu.scopes.ddgi_relocate_classify_ms",
                report.ddgiRelocateClassifyTimeMs);
  if (auto frame = frames.find(report.wholeFrameSourceFrameIndex);
      frame != frames.end()) {
    const auto addAbsent = [&](GpuTimingScope scope, std::string_view id) {
      if (!hasGpuTimingScope(report, scope)) {
        frame->second[std::string(id)] = 0.0;
      }
    };
    addAbsent(GpuTimingScope::OpaqueDepth, "gpu.scopes.opaque_depth_ms");
    addAbsent(GpuTimingScope::OpaqueNormal, "gpu.scopes.opaque_normal_ms");
    addAbsent(GpuTimingScope::OpaqueMain, "gpu.scopes.opaque_main_ms");
  }
  if (auto frame = frames.find(report.opaquePipelineStatisticsSourceFrameIndex);
      frame != frames.end()) {
    frame->second["renderer.opaque.pipeline_statistics_requested"] =
        report.opaquePipelineStatisticsRequested ? 1.0 : 0.0;
    frame->second["renderer.opaque.pipeline_statistics_available"] =
        report.opaquePipelineStatisticsAvailable ? 1.0 : 0.0;
    if (report.opaquePipelineStatisticsAvailable) {
      frame->second
          ["renderer.opaque.pipeline_statistics_input_assembly_vertices"] =
          static_cast<double>(report.opaqueInputAssemblyVertices);
      frame->second
          ["renderer.opaque.pipeline_statistics_input_assembly_primitives"] =
          static_cast<double>(report.opaqueInputAssemblyPrimitives);
      frame
          ->second["renderer.opaque.pipeline_statistics_clipping_invocations"] =
          static_cast<double>(report.opaqueClippingInvocations);
      frame->second["renderer.opaque.pipeline_statistics_clipping_primitives"] =
          static_cast<double>(report.opaqueClippingPrimitives);
      frame->second
          ["renderer.opaque.pipeline_statistics_fragment_shader_invocations"] =
          static_cast<double>(report.opaqueFragmentShaderInvocations);
    }
  }

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
