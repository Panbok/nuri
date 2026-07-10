#include "nuri/tools/benchmark/benchmark_metric_registry.h"

#include <algorithm>
#include <array>
#include <string>

namespace nuri::tools::benchmark {
namespace {

using Rule = BenchmarkMetricIdRule;
using Unit = BenchmarkMetricUnit;
using Numeric = BenchmarkMetricNumericType;
using Direction = BenchmarkMetricDirection;
using Aggregation = BenchmarkMetricAggregation;
using Availability = BenchmarkMetricAvailability;
using Phase = BenchmarkMetricSamplingPhase;
using Gate = BenchmarkMetricGateRole;

constexpr Aggregation kFrameDistribution =
    Aggregation::MedianAndP95AcrossMeasuredFrames;

#define NURI_EXACT_METRIC(id, unit, numeric, direction, availability, phase, role) \
  {id, Rule::Exact, unit, numeric, direction, kFrameDistribution, availability, \
   phase, role}

#define NURI_CPU_TIMING(id, role)                                                \
  NURI_EXACT_METRIC(id, Unit::Milliseconds, Numeric::Float64,                    \
                    Direction::LowerIsBetter, Availability::EveryMeasuredFrame, \
                    Phase::CpuMeasuredRegion, role)

#define NURI_GPU_TIMING(id, role)                                                \
  NURI_EXACT_METRIC(id, Unit::Milliseconds, Numeric::Float64,                    \
                    Direction::LowerIsBetter,                                    \
                    Availability::WhenGpuTimingAvailable,                        \
                    Phase::DelayedGpuReadback, role)

#define NURI_MEMORY(id, availability)                                            \
  NURI_EXACT_METRIC(id, Unit::Mebibytes, Numeric::Float64,                       \
                    Direction::LowerIsBetter, availability,                      \
                    Phase::PostRenderMeasuredFrame, Gate::Diagnostic)

#define NURI_COUNTER(id)                                                         \
  NURI_EXACT_METRIC(id, Unit::Count, Numeric::Uint64EncodedAsFloat64,            \
                    Direction::Informational, Availability::EveryMeasuredFrame, \
                    Phase::PostRenderMeasuredFrame,                              \
                    Gate::WorkloadCharacterization)

#define NURI_RENDERGRAPH_COUNTER(id)                                             \
  NURI_EXACT_METRIC(id, Unit::Count, Numeric::Uint64EncodedAsFloat64,            \
                    Direction::Informational,                                    \
                    Availability::WhenRenderGraphTelemetryAvailable,             \
                    Phase::PostRenderMeasuredFrame,                              \
                    Gate::WorkloadCharacterization)

constexpr BenchmarkMetricDescriptor kDescriptors[] = {
    NURI_CPU_TIMING("cpu.total_ms", Gate::Eligible),
    NURI_CPU_TIMING("cpu.scene_commit_ms", Gate::Eligible),
    NURI_CPU_TIMING("cpu.render_submit_ms", Gate::Primary),

    NURI_GPU_TIMING("gpu.scopes_sum_ms", Gate::Primary),
    NURI_GPU_TIMING("gpu.scopes.shadow_ms", Gate::Eligible),
    NURI_GPU_TIMING("gpu.scopes.shadow_depth_ms", Gate::Eligible),
    NURI_GPU_TIMING("gpu.scopes.shadow_sdsm_ms", Gate::Eligible),
    NURI_GPU_TIMING("gpu.scopes.opaque_ms", Gate::Eligible),
    NURI_GPU_TIMING("gpu.scopes.gtao_ms", Gate::Eligible),
    NURI_GPU_TIMING("gpu.scopes.msaa_resolve_ms", Gate::Eligible),
    NURI_GPU_TIMING("gpu.scopes.scene_color_downsample_ms", Gate::Eligible),
    NURI_GPU_TIMING("gpu.scopes.taa_resolve_ms", Gate::Eligible),
    NURI_GPU_TIMING("gpu.scopes.taa_debug_ms", Gate::Eligible),
    NURI_GPU_TIMING("gpu.scopes.spatial_aa_ms", Gate::Eligible),
    NURI_GPU_TIMING("gpu.scopes.transmission_ms", Gate::Eligible),
    NURI_GPU_TIMING("gpu.scopes.hdr_postprocess_ms", Gate::Eligible),
    NURI_GPU_TIMING("gpu.scopes.skybox_ms", Gate::Eligible),

    NURI_EXACT_METRIC("benchmark.camera.position_delta", Unit::WorldUnits,
                      Numeric::Float64, Direction::Informational,
                      Availability::EveryMeasuredFrame,
                      Phase::PostRenderMeasuredFrame,
                      Gate::WorkloadCharacterization),
    NURI_EXACT_METRIC("benchmark.camera.direction_delta",
                      Unit::NormalizedVectorDelta, Numeric::Float64,
                      Direction::Informational,
                      Availability::EveryMeasuredFrame,
                      Phase::PostRenderMeasuredFrame,
                      Gate::WorkloadCharacterization),

    NURI_MEMORY("memory.process.working_set_mb",
                Availability::WhenPlatformMemoryAvailable),
    NURI_MEMORY("memory.process.peak_working_set_mb",
                Availability::WhenPlatformMemoryAvailable),
    NURI_MEMORY("memory.process.private_usage_mb",
                Availability::WhenPlatformMemoryAvailable),
    NURI_MEMORY("memory.process.pagefile_usage_mb",
                Availability::WhenPlatformMemoryAvailable),
    NURI_MEMORY("memory.process.peak_pagefile_usage_mb",
                Availability::WhenPlatformMemoryAvailable),
    NURI_MEMORY("memory.pmr.renderer_current_mb",
                Availability::EveryMeasuredFrame),
    NURI_MEMORY("memory.pmr.renderer_peak_mb", Availability::EveryMeasuredFrame),
    NURI_MEMORY("memory.pmr.pipeline_current_mb",
                Availability::EveryMeasuredFrame),
    NURI_MEMORY("memory.pmr.pipeline_peak_mb", Availability::EveryMeasuredFrame),
    NURI_MEMORY("memory.pmr.scene_current_mb", Availability::EveryMeasuredFrame),
    NURI_MEMORY("memory.pmr.scene_peak_mb", Availability::EveryMeasuredFrame),
    NURI_MEMORY("gpu.memory.shadow.cascade_texture_mb",
                Availability::EveryMeasuredFrame),
    NURI_MEMORY("gpu.memory.aa.motion_vector_total_mb",
                Availability::EveryMeasuredFrame),
    NURI_MEMORY("gpu.memory.aa.reactive_mask_total_mb",
                Availability::EveryMeasuredFrame),
    NURI_MEMORY("gpu.memory.aa.spatial_aa_total_mb",
                Availability::EveryMeasuredFrame),
    NURI_MEMORY("gpu.memory.aa.msaa_total_mb", Availability::EveryMeasuredFrame),
    NURI_MEMORY("gpu.memory.ao.total_texture_mb",
                Availability::EveryMeasuredFrame),
    NURI_MEMORY("gpu.memory.hdr.texture_mb", Availability::EveryMeasuredFrame),
    NURI_MEMORY("gpu.memory.frame_textures_estimated_mb",
                Availability::EveryMeasuredFrame),

    NURI_COUNTER("renderer.opaque.total_instances"),
    NURI_COUNTER("renderer.opaque.visible_instances"),
    NURI_COUNTER("renderer.opaque.instanced_draws"),
    NURI_COUNTER("renderer.opaque.indirect_draw_calls"),
    NURI_COUNTER("renderer.opaque.indirect_commands"),
    NURI_COUNTER("renderer.opaque.compute_dispatches"),
    NURI_COUNTER("renderer.opaque.depth_prepass_draws"),
    NURI_COUNTER("renderer.opaque.tessellated_draws"),
    NURI_COUNTER("renderer.visibility.cpu_main_candidates"),
    NURI_COUNTER("renderer.visibility.cpu_main_visible_candidates"),
    NURI_COUNTER("renderer.visibility.cpu_main_rejected"),
    NURI_COUNTER("renderer.visibility.gpu_main_candidates"),
    NURI_COUNTER("renderer.visibility.gpu_main_visible_candidates"),
    NURI_COUNTER("renderer.visibility.gpu_main_rejected_frustum"),
    NURI_COUNTER("renderer.visibility.gpu_main_rejected_occlusion"),
    NURI_COUNTER("renderer.visibility.gpu_output_overflow_count"),
    NURI_COUNTER("renderer.visibility.gpu_main_readback_available"),
    NURI_COUNTER("renderer.visibility.gpu_main_readback_source_frame"),
    NURI_COUNTER("renderer.visibility.gpu_main_readback_stale_frame_count"),
    NURI_COUNTER("renderer.visibility.gpu_main_readback_error_count"),
    NURI_COUNTER("renderer.visibility.gpu_main_readback_visible_candidates"),
    NURI_COUNTER("renderer.visibility.gpu_main_visible_list_mismatches"),
    NURI_COUNTER("renderer.visibility.gpu_indirect_draw_used"),
    NURI_COUNTER("renderer.visibility.gpu_indirect_draw_fallback"),
    NURI_COUNTER("renderer.visibility.gpu_indirect_draw_commands"),
    NURI_COUNTER("renderer.visibility.gpu_indirect_draw_readback_commands"),
    NURI_COUNTER("renderer.visibility.gpu_indirect_draw_readback_tombstoned"),
    NURI_COUNTER("renderer.visibility.gpu_indirect_draw_readback_visible"),
    NURI_COUNTER("renderer.visibility.indirect_mesh_dispatch_count"),
    NURI_COUNTER("renderer.visibility.meshlet_rejected_frustum"),
    NURI_COUNTER("renderer.visibility.meshlet_rejected_cone"),
    NURI_COUNTER("renderer.visibility.meshlet_rejected_occlusion"),
    NURI_COUNTER("renderer.visibility.meshlet_occlusion_available"),
    NURI_COUNTER("renderer.visibility.meshlet_payload_overflow_count"),
    NURI_COUNTER("renderer.visibility.meshlet_readback_available"),
    NURI_COUNTER("renderer.visibility.meshlet_readback_source_frame"),
    NURI_COUNTER("renderer.visibility.meshlet_readback_stale_frame_count"),
    NURI_COUNTER("renderer.visibility.meshlet_readback_error_count"),
    NURI_COUNTER("renderer.visibility.meshlet_emitted"),
    NURI_COUNTER("renderer.visibility.meshlet_task_groups_executed"),
    NURI_COUNTER("renderer.visibility.uncertain_visible"),
    NURI_COUNTER("renderer.visibility.shadow_cpu_candidates"),
    NURI_COUNTER("renderer.visibility.shadow_cpu_rejected"),
    NURI_COUNTER("renderer.visibility.shadow_meshlet_candidates"),
    NURI_COUNTER("renderer.visibility.shadow_meshlet_readback_available"),
    NURI_COUNTER("renderer.visibility.shadow_meshlet_readback_source_frame"),
    NURI_COUNTER("renderer.visibility.shadow_meshlet_readback_stale_frame_count"),
    NURI_COUNTER("renderer.visibility.shadow_meshlet_readback_error_count"),
    NURI_COUNTER("renderer.visibility.shadow_meshlet_rejected_bounds"),
    NURI_COUNTER("renderer.visibility.occlusion_available"),
    NURI_COUNTER("renderer.shadow.cascades"),
    NURI_COUNTER("renderer.shadow.total_draws"),
    NURI_COUNTER("renderer.shadow.total_culled_draws"),
    NURI_COUNTER("renderer.shadow.static_caster_entries"),
    NURI_COUNTER("renderer.shadow.dynamic_caster_entries"),
    NURI_COUNTER("renderer.shadow.filter_sample_budget"),
    NURI_COUNTER("renderer.shadow.sdsm_compute_passes"),
    NURI_COUNTER("renderer.aa.motion_vector_textures"),
    NURI_COUNTER("renderer.aa.motion_vector_allocations"),
    NURI_COUNTER("renderer.aa.motion_vector_reallocations"),
    NURI_COUNTER("renderer.aa.motion_vector_depth_reprojection_passes"),
    NURI_COUNTER("renderer.aa.velocity_passes"),
    NURI_COUNTER("renderer.aa.velocity_draws"),
    NURI_COUNTER("renderer.aa.velocity_instances"),
    NURI_COUNTER("renderer.aa.reactive_mask_passes"),
    NURI_COUNTER("renderer.aa.reactive_mask_draws"),
    NURI_COUNTER("renderer.aa.taa_resolve_passes"),
    NURI_COUNTER("renderer.aa.taa_copy_back_passes"),
    NURI_COUNTER("renderer.aa.spatial_aa_passes"),
    NURI_COUNTER("renderer.aa.msaa_resolve_passes"),
    NURI_COUNTER("renderer.ao.normal_prepass_draws"),
    NURI_COUNTER("renderer.ao.depth_prefilter_passes"),
    NURI_COUNTER("renderer.ao.main_passes"),
    NURI_COUNTER("renderer.ao.temporal_passes"),
    NURI_COUNTER("renderer.ao.texture_count"),
    NURI_COUNTER("renderer.hdr.bloom_passes"),
    NURI_COUNTER("renderer.hdr.luminance_passes"),
    NURI_COUNTER("renderer.hdr.adaptation_passes"),
    NURI_COUNTER("renderer.hdr.texture_count"),
    NURI_COUNTER("renderer.transparent.mesh_draws"),
    NURI_COUNTER("renderer.transparent.contributor_sortable_draws"),
    NURI_COUNTER("renderer.transparent.contributor_fixed_draws"),
    NURI_COUNTER("renderer.transparent.pick_draws"),

    NURI_RENDERGRAPH_COUNTER("rendergraph.summary.declared_pass_count"),
    NURI_RENDERGRAPH_COUNTER("rendergraph.summary.culled_pass_count"),
    NURI_RENDERGRAPH_COUNTER("rendergraph.summary.root_pass_count"),
    NURI_RENDERGRAPH_COUNTER("rendergraph.summary.pass_count"),
    NURI_RENDERGRAPH_COUNTER("rendergraph.summary.edge_count"),
    NURI_RENDERGRAPH_COUNTER("rendergraph.summary.recorded_graphics_pass_count"),
    NURI_RENDERGRAPH_COUNTER("rendergraph.summary.pass_barrier_plan_count"),
    NURI_RENDERGRAPH_COUNTER("rendergraph.summary.final_barrier_record_count"),
    NURI_RENDERGRAPH_COUNTER("rendergraph.summary.pass_barrier_record_count"),
    NURI_RENDERGRAPH_COUNTER("rendergraph.summary.recorded_command_buffer_count"),
    NURI_RENDERGRAPH_COUNTER("rendergraph.summary.submit_batch_count"),
    NURI_RENDERGRAPH_COUNTER("rendergraph.summary.pass_range_count"),
    NURI_RENDERGRAPH_COUNTER("rendergraph.summary.pass_timing_count"),
    NURI_RENDERGRAPH_COUNTER("rendergraph.summary.imported_texture_count"),
    NURI_RENDERGRAPH_COUNTER("rendergraph.summary.transient_texture_count"),
    NURI_RENDERGRAPH_COUNTER("rendergraph.summary.imported_buffer_count"),
    NURI_RENDERGRAPH_COUNTER("rendergraph.summary.transient_buffer_count"),
    NURI_RENDERGRAPH_COUNTER(
        "rendergraph.summary.transient_texture_lifetime_count"),
    NURI_RENDERGRAPH_COUNTER(
        "rendergraph.summary.transient_buffer_lifetime_count"),
    NURI_RENDERGRAPH_COUNTER(
        "rendergraph.summary.transient_texture_physical_count"),
    NURI_RENDERGRAPH_COUNTER(
        "rendergraph.summary.transient_buffer_physical_count"),
    NURI_RENDERGRAPH_COUNTER(
        "rendergraph.summary.transient_texture_allocation_map_size"),
    NURI_RENDERGRAPH_COUNTER(
        "rendergraph.summary.transient_buffer_allocation_map_size"),
    NURI_RENDERGRAPH_COUNTER(
        "rendergraph.summary.transient_texture_physical_allocation_count"),
    NURI_RENDERGRAPH_COUNTER(
        "rendergraph.summary.transient_buffer_physical_allocation_count"),
    NURI_RENDERGRAPH_COUNTER(
        "rendergraph.summary.unresolved_texture_binding_count"),
    NURI_RENDERGRAPH_COUNTER(
        "rendergraph.summary.resolved_dependency_buffer_slot_count"),
    NURI_RENDERGRAPH_COUNTER(
        "rendergraph.summary.unresolved_dependency_buffer_binding_count"),
    NURI_RENDERGRAPH_COUNTER("rendergraph.summary.owned_pre_dispatch_count"),
    NURI_RENDERGRAPH_COUNTER("rendergraph.summary.owned_draw_item_count"),
    NURI_RENDERGRAPH_COUNTER(
        "rendergraph.summary.owned_mesh_dispatch_item_count"),

    {"rendergraph.pass.<index>.<pass>.cpu_ms",
     Rule::RenderGraphPassCpuTiming,
     Unit::Milliseconds,
     Numeric::Float64,
     Direction::LowerIsBetter,
     kFrameDistribution,
     Availability::WhenRenderGraphTelemetryAvailable,
     Phase::PostRenderMeasuredFrame,
     Gate::Eligible},
    {"rendergraph.pass.<index>.<pass>.gpu_ms",
     Rule::RenderGraphPassGpuTiming,
     Unit::Milliseconds,
     Numeric::Float64,
     Direction::LowerIsBetter,
     kFrameDistribution,
     Availability::WhenGpuTimingAvailable,
     Phase::DelayedGpuReadback,
     Gate::Eligible},
};

#undef NURI_RENDERGRAPH_COUNTER
#undef NURI_COUNTER
#undef NURI_MEMORY
#undef NURI_GPU_TIMING
#undef NURI_CPU_TIMING
#undef NURI_EXACT_METRIC

[[nodiscard]] bool isLowerSlug(std::string_view value) noexcept {
  if (value.empty() || value.size() > 64u || value.front() == '_' ||
      value.back() == '_') {
    return false;
  }
  return std::all_of(value.begin(), value.end(), [](char character) {
    return (character >= 'a' && character <= 'z') ||
           (character >= '0' && character <= '9') || character == '_';
  });
}

[[nodiscard]] bool matchesRenderGraphPassTiming(std::string_view metricId,
                                                std::string_view suffix) noexcept {
  constexpr std::string_view prefix = "rendergraph.pass.";
  if (!metricId.starts_with(prefix) || !metricId.ends_with(suffix) ||
      metricId.size() > 160u) {
    return false;
  }
  const std::string_view body = metricId.substr(
      prefix.size(), metricId.size() - prefix.size() - suffix.size());
  const size_t separator = body.find('.');
  if (separator < 3u || separator > 6u || separator + 1u >= body.size()) {
    return false;
  }
  const std::string_view index = body.substr(0u, separator);
  const std::string_view pass = body.substr(separator + 1u);
  return std::all_of(index.begin(), index.end(), [](char character) {
           return character >= '0' && character <= '9';
         }) &&
         isLowerSlug(pass);
}

} // namespace

std::span<const BenchmarkMetricDescriptor>
benchmarkMetricDescriptors() noexcept {
  return std::span{kDescriptors};
}

const BenchmarkMetricDescriptor *
findBenchmarkMetricDescriptor(std::string_view metricId) noexcept {
  for (const BenchmarkMetricDescriptor &descriptor : kDescriptors) {
    switch (descriptor.idRule) {
    case Rule::Exact:
      if (descriptor.idOrRule == metricId) {
        return &descriptor;
      }
      break;
    case Rule::RenderGraphPassCpuTiming:
      if (matchesRenderGraphPassTiming(metricId, ".cpu_ms")) {
        return &descriptor;
      }
      break;
    case Rule::RenderGraphPassGpuTiming:
      if (matchesRenderGraphPassTiming(metricId, ".gpu_ms")) {
        return &descriptor;
      }
      break;
    }
  }
  return nullptr;
}

std::optional<BenchmarkMetricIndex>
findExactBenchmarkMetricIndex(std::string_view metricId) noexcept {
  for (size_t index = 0u; index < std::size(kDescriptors); ++index) {
    const BenchmarkMetricDescriptor &descriptor = kDescriptors[index];
    if (descriptor.idRule == Rule::Exact && descriptor.idOrRule == metricId) {
      return BenchmarkMetricIndex{static_cast<uint16_t>(index)};
    }
  }
  return std::nullopt;
}

const BenchmarkMetricDescriptor *
benchmarkMetricDescriptor(BenchmarkMetricIndex index) noexcept {
  if (!index.valid() || index.value >= std::size(kDescriptors)) {
    return nullptr;
  }
  const BenchmarkMetricDescriptor &descriptor = kDescriptors[index.value];
  return descriptor.idRule == Rule::Exact ? &descriptor : nullptr;
}

size_t exactBenchmarkMetricCount() noexcept {
  static const size_t count = static_cast<size_t>(std::count_if(
      std::begin(kDescriptors), std::end(kDescriptors),
      [](const BenchmarkMetricDescriptor &descriptor) {
        return descriptor.idRule == Rule::Exact;
      }));
  return count;
}

Result<const BenchmarkMetricDescriptor *, std::string>
requireBenchmarkMetricDescriptor(std::string_view metricId,
                                 std::string_view field) {
  const BenchmarkMetricDescriptor *descriptor =
      findBenchmarkMetricDescriptor(metricId);
  if (descriptor == nullptr) {
    return Result<const BenchmarkMetricDescriptor *, std::string>::makeError(
        std::string(field) + " contains unregistered benchmark metric '" +
        std::string(metricId) + "'");
  }
  return Result<const BenchmarkMetricDescriptor *, std::string>::makeResult(
      descriptor);
}

std::string_view benchmarkMetricIdRuleName(BenchmarkMetricIdRule rule) noexcept {
  switch (rule) {
  case Rule::Exact:
    return "exact";
  case Rule::RenderGraphPassCpuTiming:
    return "rendergraph-pass-cpu-timing";
  case Rule::RenderGraphPassGpuTiming:
    return "rendergraph-pass-gpu-timing";
  }
  return "exact";
}

std::string_view benchmarkMetricUnitName(BenchmarkMetricUnit unit) noexcept {
  switch (unit) {
  case Unit::Milliseconds:
    return "ms";
  case Unit::Mebibytes:
    return "MiB";
  case Unit::Count:
    return "count";
  case Unit::WorldUnits:
    return "world-units";
  case Unit::NormalizedVectorDelta:
    return "normalized-vector-delta";
  }
  return "count";
}

std::string_view
benchmarkMetricNumericTypeName(BenchmarkMetricNumericType type) noexcept {
  switch (type) {
  case Numeric::Float64:
    return "float64";
  case Numeric::Uint64EncodedAsFloat64:
    return "uint64-as-float64";
  }
  return "float64";
}

std::string_view
benchmarkMetricDirectionName(BenchmarkMetricDirection direction) noexcept {
  switch (direction) {
  case Direction::LowerIsBetter:
    return "lower-is-better";
  case Direction::Informational:
    return "informational";
  }
  return "informational";
}

std::string_view benchmarkMetricAggregationName(
    BenchmarkMetricAggregation aggregation) noexcept {
  switch (aggregation) {
  case Aggregation::MedianAndP95AcrossMeasuredFrames:
    return "median-and-p95-across-measured-frames";
  }
  return "median-and-p95-across-measured-frames";
}

std::string_view benchmarkMetricAvailabilityName(
    BenchmarkMetricAvailability availability) noexcept {
  switch (availability) {
  case Availability::EveryMeasuredFrame:
    return "every-measured-frame";
  case Availability::WhenGpuTimingAvailable:
    return "when-gpu-timing-available";
  case Availability::WhenRenderGraphTelemetryAvailable:
    return "when-rendergraph-telemetry-available";
  case Availability::WhenPlatformMemoryAvailable:
    return "when-platform-memory-available";
  }
  return "every-measured-frame";
}

std::string_view benchmarkMetricSamplingPhaseName(
    BenchmarkMetricSamplingPhase phase) noexcept {
  switch (phase) {
  case Phase::CpuMeasuredRegion:
    return "cpu-measured-region";
  case Phase::PostRenderMeasuredFrame:
    return "post-render-measured-frame";
  case Phase::DelayedGpuReadback:
    return "delayed-gpu-readback";
  }
  return "post-render-measured-frame";
}

std::string_view
benchmarkMetricGateRoleName(BenchmarkMetricGateRole role) noexcept {
  switch (role) {
  case Gate::Primary:
    return "primary";
  case Gate::Eligible:
    return "eligible";
  case Gate::WorkloadCharacterization:
    return "workload-characterization";
  case Gate::Diagnostic:
    return "diagnostic";
  }
  return "diagnostic";
}

} // namespace nuri::tools::benchmark
