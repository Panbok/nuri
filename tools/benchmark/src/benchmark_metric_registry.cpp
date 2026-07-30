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
using Evidence = BenchmarkMetricEvidenceClass;

constexpr Aggregation kFrameDistribution =
    Aggregation::MedianAndP95AcrossMeasuredFrames;

#define NURI_EXACT_METRIC(id, unit, numeric, direction, availability, phase,   \
                          role)                                                \
  {id,           Rule::Exact, unit, numeric, direction, kFrameDistribution,    \
   availability, phase,       role}

#define NURI_CPU_TIMING(id, role)                                              \
  NURI_EXACT_METRIC(                                                           \
      id, Unit::Milliseconds, Numeric::Float64, Direction::LowerIsBetter,      \
      Availability::EveryMeasuredFrame, Phase::CpuMeasuredRegion, role)

#define NURI_GPU_TIMING(id, role)                                              \
  NURI_EXACT_METRIC(                                                           \
      id, Unit::Milliseconds, Numeric::Float64, Direction::LowerIsBetter,      \
      Availability::WhenGpuTimingAvailable, Phase::DelayedGpuReadback, role)

#define NURI_MEMORY(id, availability)                                          \
  NURI_EXACT_METRIC(id, Unit::Mebibytes, Numeric::Float64,                     \
                    Direction::LowerIsBetter, availability,                    \
                    Phase::PostRenderMeasuredFrame, Gate::Diagnostic)

#define NURI_COUNTER(id)                                                       \
  NURI_EXACT_METRIC(                                                           \
      id, Unit::Count, Numeric::Uint64EncodedAsFloat64,                        \
      Direction::Informational, Availability::EveryMeasuredFrame,              \
      Phase::PostRenderMeasuredFrame, Gate::WorkloadCharacterization)

#define NURI_DIAGNOSTIC_COUNTER(id)                                            \
  {id,                                                                         \
   Rule::Exact,                                                                \
   Unit::Count,                                                                \
   Numeric::Uint64EncodedAsFloat64,                                            \
   Direction::Informational,                                                   \
   kFrameDistribution,                                                         \
   Availability::EveryMeasuredFrame,                                           \
   Phase::DelayedGpuReadback,                                                  \
   Gate::Diagnostic,                                                           \
   Evidence::DiagnosticOnly}

#define NURI_DIAGNOSTIC_CPU_COUNTER(id)                                        \
  {id,                                                                         \
   Rule::Exact,                                                                \
   Unit::Count,                                                                \
   Numeric::Uint64EncodedAsFloat64,                                            \
   Direction::Informational,                                                   \
   kFrameDistribution,                                                         \
   Availability::EveryMeasuredFrame,                                           \
   Phase::PostRenderMeasuredFrame,                                             \
   Gate::Diagnostic,                                                           \
   Evidence::DiagnosticOnly}

#define NURI_RATIO(id)                                                         \
  NURI_EXACT_METRIC(id, Unit::Ratio, Numeric::Float64,                         \
                    Direction::Informational,                                  \
                    Availability::EveryMeasuredFrame,                          \
                    Phase::PostRenderMeasuredFrame, Gate::Diagnostic)

#define NURI_WORLD(id)                                                         \
  NURI_EXACT_METRIC(id, Unit::WorldUnits, Numeric::Float64,                    \
                    Direction::Informational,                                  \
                    Availability::EveryMeasuredFrame,                          \
                    Phase::PostRenderMeasuredFrame, Gate::Diagnostic)

#define NURI_POST_CPU_TIMING(id)                                               \
  NURI_EXACT_METRIC(id, Unit::Milliseconds, Numeric::Float64,                  \
                    Direction::LowerIsBetter,                                  \
                    Availability::EveryMeasuredFrame,                          \
                    Phase::PostRenderMeasuredFrame, Gate::Diagnostic)

#define NURI_DDGI_VOLUME_METRICS(slot)                                         \
  NURI_COUNTER("renderer.ddgi.volume" #slot ".active"),                        \
      NURI_COUNTER("renderer.ddgi.volume" #slot ".effective_kind"),            \
      NURI_COUNTER("renderer.ddgi.volume" #slot ".tier"),                      \
      NURI_COUNTER("renderer.ddgi.volume" #slot ".cascade_index"),             \
      NURI_COUNTER("renderer.ddgi.volume" #slot ".total_probes"),              \
      NURI_COUNTER("renderer.ddgi.volume" #slot ".initialized_probes"),        \
      NURI_COUNTER("renderer.ddgi.volume" #slot ".shading_enabled_probes"),    \
      NURI_COUNTER("renderer.ddgi.volume" #slot ".invalid_probes"),            \
      NURI_COUNTER("renderer.ddgi.volume" #slot ".newly_exposed_probes"),      \
      NURI_COUNTER("renderer.ddgi.volume" #slot ".updates"),                   \
      NURI_COUNTER("renderer.ddgi.volume" #slot ".primary_queries"),           \
      NURI_COUNTER("renderer.ddgi.volume" #slot ".primary_queries_issued"),    \
      NURI_COUNTER("renderer.ddgi.volume" #slot ".secondary_queries"),         \
      NURI_COUNTER("renderer.ddgi.volume" #slot ".update_age_median"),         \
      NURI_COUNTER("renderer.ddgi.volume" #slot ".update_age_p95"),            \
      NURI_COUNTER("renderer.ddgi.volume" #slot ".update_age_maximum"),        \
      NURI_COUNTER("renderer.ddgi.volume" #slot ".scheduled_quota"),           \
      NURI_COUNTER("renderer.ddgi.volume" #slot ".used_quota"),                \
      NURI_EXACT_METRIC("renderer.ddgi.volume" #slot ".deficit", Unit::Count,  \
                        Numeric::Float64, Direction::Informational,            \
                        Availability::EveryMeasuredFrame,                      \
                        Phase::PostRenderMeasuredFrame, Gate::Diagnostic),     \
      NURI_COUNTER("renderer.ddgi.volume" #slot ".starvation_frames"),         \
      NURI_COUNTER("renderer.ddgi.volume" #slot                                \
                   ".estimated_full_refresh_frames"),                          \
      NURI_MEMORY("renderer.ddgi.volume" #slot ".persistent_mb",               \
                  Availability::EveryMeasuredFrame),                           \
      NURI_RATIO("renderer.ddgi.volume" #slot ".unique_coverage_percentage"),  \
      NURI_COUNTER("renderer.ddgi.volume" #slot ".redundant_coverage"),        \
      NURI_RATIO("renderer.ddgi.volume" #slot ".history_ready_percentage"),    \
      NURI_RATIO("renderer.ddgi.volume" #slot ".coverage_ready_percentage"),   \
      NURI_RATIO("renderer.ddgi.volume" #slot ".confidence")

#define NURI_RENDERGRAPH_COUNTER(id)                                           \
  NURI_EXACT_METRIC(id, Unit::Count, Numeric::Uint64EncodedAsFloat64,          \
                    Direction::Informational,                                  \
                    Availability::WhenRenderGraphTelemetryAvailable,           \
                    Phase::PostRenderMeasuredFrame,                            \
                    Gate::WorkloadCharacterization)

#define NURI_ASSET_TIMING(id)                                                  \
  NURI_EXACT_METRIC(id, Unit::Milliseconds, Numeric::Float64,                  \
                    Direction::LowerIsBetter,                                  \
                    Availability::EveryMeasuredFrame,                          \
                    Phase::PostRenderMeasuredFrame, Gate::Eligible)

constexpr BenchmarkMetricDescriptor kDescriptors[] = {
    NURI_CPU_TIMING("cpu.total_ms", Gate::Eligible),
    NURI_CPU_TIMING("cpu.scene_commit_ms", Gate::Eligible),
    NURI_CPU_TIMING("cpu.render_submit_ms", Gate::Primary),
    NURI_CPU_TIMING("cpu.scene_resource_prepare_ms", Gate::Primary),
    NURI_POST_CPU_TIMING("cpu.ray_tracing.prepare_ms"),
    NURI_POST_CPU_TIMING("cpu.ray_tracing.topology_prepare_ms"),
    NURI_POST_CPU_TIMING("cpu.ray_tracing.transform_prepare_ms"),
    NURI_POST_CPU_TIMING("cpu.ray_tracing.deformation_prepare_ms"),
    NURI_POST_CPU_TIMING("cpu.ray_tracing.tlas_prepare_ms"),
    NURI_CPU_TIMING("texture.artifact_build_ms", Gate::Eligible),
    NURI_CPU_TIMING("texture.normal_variance_artifact_build_ms",
                    Gate::Eligible),
    NURI_CPU_TIMING("texture.dds_read_ms", Gate::Eligible),
    NURI_CPU_TIMING("texture.pack.build_ms", Gate::Eligible),
    NURI_CPU_TIMING("texture.pack.open_ms", Gate::Eligible),
    NURI_CPU_TIMING("texture.pack.read_ms", Gate::Eligible),

    NURI_EXACT_METRIC("gpu.frame_ms", Unit::Milliseconds, Numeric::Float64,
                      Direction::LowerIsBetter,
                      Availability::WhenWholeFrameGpuTimingAvailable,
                      Phase::DelayedGpuReadback, Gate::Primary),
    NURI_GPU_TIMING("gpu.scopes_sum_ms", Gate::Primary),
    NURI_GPU_TIMING("gpu.scopes.shadow_ms", Gate::Eligible),
    NURI_GPU_TIMING("gpu.scopes.shadow_depth_ms", Gate::Eligible),
    NURI_GPU_TIMING("gpu.scopes.shadow_sdsm_ms", Gate::Eligible),
    NURI_GPU_TIMING("gpu.scopes.opaque_ms", Gate::Eligible),
    NURI_GPU_TIMING("gpu.scopes.opaque_depth_ms", Gate::Eligible),
    NURI_GPU_TIMING("gpu.scopes.opaque_normal_ms", Gate::Eligible),
    NURI_GPU_TIMING("gpu.scopes.opaque_main_ms", Gate::Eligible),
    NURI_GPU_TIMING("gpu.scopes.gtao_ms", Gate::Eligible),
    NURI_GPU_TIMING("renderer.ao.input_ms", Gate::Eligible),
    NURI_GPU_TIMING("renderer.ao.prefilter_edges_ms", Gate::Eligible),
    NURI_GPU_TIMING("renderer.ao.main_ms", Gate::Eligible),
    NURI_GPU_TIMING("renderer.ao.denoise_ms", Gate::Eligible),
    NURI_GPU_TIMING("renderer.ao.upscale_ms", Gate::Eligible),
    NURI_GPU_TIMING("renderer.ao.temporal_ms", Gate::Eligible),
    NURI_GPU_TIMING("gpu.scopes.msaa_resolve_ms", Gate::Eligible),
    NURI_GPU_TIMING("gpu.scopes.scene_color_downsample_ms", Gate::Eligible),
    NURI_GPU_TIMING("gpu.scopes.taa_resolve_ms", Gate::Eligible),
    NURI_GPU_TIMING("gpu.scopes.taa_debug_ms", Gate::Eligible),
    NURI_GPU_TIMING("gpu.scopes.spatial_aa_ms", Gate::Eligible),
    NURI_GPU_TIMING("gpu.scopes.transmission_ms", Gate::Eligible),
    NURI_GPU_TIMING("gpu.scopes.hdr_postprocess_ms", Gate::Eligible),
    NURI_GPU_TIMING("gpu.scopes.skybox_ms", Gate::Eligible),
    NURI_GPU_TIMING("gpu.scopes.velocity_ms", Gate::Eligible),
    NURI_GPU_TIMING("gpu.scopes.reactive_mask_ms", Gate::Eligible),
    NURI_GPU_TIMING("gpu.scopes.taa_copy_back_ms", Gate::Eligible),
    NURI_GPU_TIMING("gpu.scopes.gtao_temporal_ms", Gate::Eligible),
    NURI_GPU_TIMING("gpu.scopes.ray_tracing_scene_ms", Gate::Eligible),
    NURI_GPU_TIMING("gpu.scopes.ray_tracing_blas_ms", Gate::Eligible),
    NURI_GPU_TIMING("gpu.scopes.ray_tracing_tlas_ms", Gate::Eligible),
    NURI_GPU_TIMING("gpu.scopes.ddgi_ms", Gate::Eligible),
    NURI_GPU_TIMING("gpu.scopes.ddgi_trace_ms", Gate::Eligible),
    NURI_GPU_TIMING("gpu.scopes.ddgi_update_ms", Gate::Eligible),
    NURI_GPU_TIMING("gpu.scopes.ddgi_irradiance_update_ms", Gate::Eligible),
    NURI_GPU_TIMING("gpu.scopes.ddgi_distance_update_ms", Gate::Eligible),
    NURI_GPU_TIMING("gpu.scopes.ddgi_relocate_classify_ms", Gate::Eligible),
    NURI_GPU_TIMING("gpu.scopes.ddgi_readback_ms", Gate::Eligible),
    NURI_GPU_TIMING("gpu.scopes.ddgi_surface_cache_ms", Gate::Eligible),

    NURI_EXACT_METRIC(
        "benchmark.camera.position_delta", Unit::WorldUnits, Numeric::Float64,
        Direction::Informational, Availability::EveryMeasuredFrame,
        Phase::PostRenderMeasuredFrame, Gate::WorkloadCharacterization),
    NURI_EXACT_METRIC(
        "benchmark.camera.direction_delta", Unit::NormalizedVectorDelta,
        Numeric::Float64, Direction::Informational,
        Availability::EveryMeasuredFrame, Phase::PostRenderMeasuredFrame,
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
    NURI_MEMORY("memory.pmr.renderer_peak_mb",
                Availability::EveryMeasuredFrame),
    NURI_MEMORY("memory.pmr.pipeline_current_mb",
                Availability::EveryMeasuredFrame),
    NURI_MEMORY("memory.pmr.pipeline_peak_mb",
                Availability::EveryMeasuredFrame),
    NURI_MEMORY("memory.pmr.scene_current_mb",
                Availability::EveryMeasuredFrame),
    NURI_MEMORY("memory.pmr.scene_peak_mb", Availability::EveryMeasuredFrame),
    NURI_MEMORY("gpu.memory.shadow.cascade_texture_mb",
                Availability::EveryMeasuredFrame),
    NURI_MEMORY("gpu.memory.aa.motion_vector_total_mb",
                Availability::EveryMeasuredFrame),
    NURI_MEMORY("gpu.memory.aa.reactive_mask_total_mb",
                Availability::EveryMeasuredFrame),
    NURI_MEMORY("gpu.memory.aa.motion_class_total_mb",
                Availability::EveryMeasuredFrame),
    NURI_MEMORY("gpu.memory.aa.history_color_total_mb",
                Availability::EveryMeasuredFrame),
    NURI_MEMORY("gpu.memory.aa.spatial_aa_total_mb",
                Availability::EveryMeasuredFrame),
    NURI_MEMORY("gpu.memory.aa.normal_variance_contract_textures_mb",
                Availability::EveryMeasuredFrame),
    NURI_MEMORY("gpu.memory.aa.msaa_total_mb",
                Availability::EveryMeasuredFrame),
    NURI_MEMORY("gpu.memory.aa.msaa_active_color_mb",
                Availability::EveryMeasuredFrame),
    NURI_MEMORY("gpu.memory.aa.msaa_active_depth_mb",
                Availability::EveryMeasuredFrame),
    NURI_MEMORY("gpu.memory.aa.msaa_ring_color_mb",
                Availability::EveryMeasuredFrame),
    NURI_MEMORY("gpu.memory.aa.msaa_ring_depth_mb",
                Availability::EveryMeasuredFrame),
    NURI_MEMORY("gpu.traffic.aa.msaa_resolve_read_estimated_mb",
                Availability::EveryMeasuredFrame),
    NURI_MEMORY("gpu.traffic.aa.msaa_resolve_write_estimated_mb",
                Availability::EveryMeasuredFrame),
    NURI_MEMORY("gpu.memory.ao.total_texture_mb",
                Availability::EveryMeasuredFrame),
    NURI_MEMORY("gpu.memory.hdr.texture_mb", Availability::EveryMeasuredFrame),
    NURI_MEMORY("gpu.memory.frame_textures_estimated_mb",
                Availability::EveryMeasuredFrame),
    NURI_MEMORY("texture.io.authored_source_read_mb",
                Availability::EveryMeasuredFrame),
    NURI_MEMORY("texture.io.native_artifact_read_mb",
                Availability::EveryMeasuredFrame),
    NURI_MEMORY("texture.io.normal_variance_artifact_write_mb",
                Availability::EveryMeasuredFrame),
    NURI_MEMORY("texture.io.dds_source_read_mb",
                Availability::EveryMeasuredFrame),
    NURI_MEMORY("texture.io.pack_bytes_served_mb",
                Availability::EveryMeasuredFrame),
    NURI_MEMORY("texture.io.pack_build_source_read_mb",
                Availability::EveryMeasuredFrame),
    NURI_MEMORY("texture.upload.recorded_mb", Availability::EveryMeasuredFrame),

    NURI_COUNTER("texture.cache.native_hits"),
    NURI_COUNTER("texture.cache.native_misses"),
    NURI_COUNTER("texture.cache.native_stale"),
    NURI_COUNTER("texture.cache.native_corrupt"),
    NURI_COUNTER("texture.cache.native_writes"),
    NURI_COUNTER("texture.cache.native_write_failures"),
    NURI_COUNTER("texture.cache.artifact_builds"),
    NURI_COUNTER("texture.cache.normal_variance_artifact_builds"),
    NURI_COUNTER("texture.cache.normal_variance_clean_texels"),
    NURI_COUNTER("texture.cache.normal_variance_toksvig_fallback_texels"),
    NURI_COUNTER("texture.cache.normal_variance_contract_rejections"),
    NURI_COUNTER("texture.pack.hits"),
    NURI_COUNTER("texture.pack.misses"),
    NURI_COUNTER("texture.pack.stale"),
    NURI_COUNTER("texture.pack.corrupt"),
    NURI_COUNTER("texture.pack.builds"),
    NURI_COUNTER("texture.pack.build_failures"),
    NURI_COUNTER("texture.pack.read_failures"),
    NURI_COUNTER("texture.pack.entries_served"),
    NURI_COUNTER("texture.upload.textures_recorded"),
    NURI_COUNTER("texture.upload.batches_submitted"),
    NURI_COUNTER("texture.upload.bounded_batch_flushes"),
    NURI_COUNTER("texture.upload.completion_waits"),

    NURI_COUNTER("renderer.assets.cpu_completions"),
    NURI_COUNTER("renderer.assets.cpu_workers"),
    NURI_COUNTER("renderer.assets.cpu_active_worker_limit"),
    NURI_COUNTER("renderer.assets.cpu_interactive_mode"),
    NURI_COUNTER("renderer.assets.cpu_queued_jobs"),
    NURI_COUNTER("renderer.assets.cpu_running_jobs"),
    NURI_COUNTER("renderer.assets.cpu_running_io"),
    NURI_COUNTER("renderer.assets.cpu_running_decode"),
    NURI_COUNTER("renderer.assets.cpu_running_cook"),
    NURI_COUNTER("renderer.assets.cpu_running_transcode"),
    NURI_COUNTER("renderer.assets.cpu_running_metadata"),
    NURI_COUNTER("renderer.assets.dedicated_copy_queue"),
    NURI_COUNTER("renderer.assets.gpu_materialized"),
    NURI_COUNTER("renderer.assets.published"),
    NURI_COUNTER("renderer.assets.cancelled"),
    NURI_COUNTER("renderer.assets.failed"),
    NURI_COUNTER("renderer.assets.scene_patches"),
    NURI_COUNTER("renderer.assets.scene_commits"),
    NURI_COUNTER("renderer.assets.deferred_cpu_completions"),
    NURI_COUNTER("renderer.assets.publication_deadline_exceeded"),
    NURI_ASSET_TIMING("renderer.assets.publication_main_thread_ms"),
    NURI_ASSET_TIMING("renderer.assets.publication_max_operation_ms"),
    NURI_COUNTER("renderer.assets.cpu_in_flight_bytes"),
    NURI_COUNTER("renderer.assets.upload_bytes"),
    NURI_COUNTER("renderer.assets.submitted_jobs"),
    NURI_COUNTER("renderer.assets.completed_jobs"),
    NURI_COUNTER("renderer.assets.cancelled_jobs"),
    NURI_COUNTER("renderer.assets.rejected_jobs"),

    NURI_COUNTER("renderer.opaque.total_instances"),
    NURI_COUNTER("renderer.opaque.visible_instances"),
    NURI_COUNTER("renderer.opaque.instanced_draws"),
    NURI_COUNTER("renderer.opaque.indirect_draw_calls"),
    NURI_COUNTER("renderer.opaque.indirect_commands"),
    NURI_COUNTER("renderer.opaque.compute_dispatches"),
    NURI_COUNTER("renderer.opaque.depth_prepass_draws"),
    NURI_COUNTER("renderer.opaque.tessellated_draws"),
    NURI_COUNTER("renderer.opaque.meshlet_dispatches"),
    NURI_COUNTER("renderer.opaque.meshlet_task_groups"),
    NURI_COUNTER("renderer.opaque.meshlet_candidates"),
    NURI_COUNTER("renderer.opaque.meshlet_mode_required"),
    NURI_COUNTER("renderer.opaque.meshlet_mode_active"),
    NURI_COUNTER("renderer.opaque.meshlet_rejected_missing_feature"),
    NURI_COUNTER("renderer.opaque.meshlet_rejected_missing_asset_data"),
    NURI_COUNTER("renderer.opaque.meshlet_rejected_incompatible_frame"),
    NURI_COUNTER("renderer.opaque.meshlet_hybrid_active"),
    NURI_COUNTER("renderer.opaque.meshlet_hybrid_classic_batches"),
    NURI_COUNTER("renderer.opaque.meshlet_hybrid_classic_instances"),
    NURI_COUNTER("renderer.opaque.meshlet_hybrid_coverage_classic_batches"),
    NURI_COUNTER("renderer.opaque.meshlet_hybrid_coverage_classic_instances"),
    NURI_COUNTER("renderer.opaque.meshlet_hybrid_meshlet_batches"),
    NURI_COUNTER("renderer.opaque.meshlet_hybrid_meshlet_instances"),
    NURI_COUNTER("renderer.opaque.auto_lod_active"),
    NURI_COUNTER("renderer.opaque.auto_lod_history_reset"),
    NURI_COUNTER("renderer.opaque.auto_lod_transitions"),
    NURI_COUNTER("renderer.opaque.auto_lod_lod0_instances"),
    NURI_COUNTER("renderer.opaque.auto_lod_lod1_instances"),
    NURI_COUNTER("renderer.opaque.classic_main_draws"),
    NURI_COUNTER("renderer.opaque.classic_alpha_masked_main_draws"),
    NURI_COUNTER("renderer.opaque.meshlet_main_dispatches"),
    NURI_COUNTER("renderer.opaque.meshlet_main_represented_items"),
    NURI_COUNTER("renderer.opaque.meshlet_alpha_masked_main_dispatches"),
    NURI_COUNTER("renderer.opaque.meshlet_alpha_masked_main_items"),
    NURI_COUNTER("renderer.opaque.msaa_depth_prepass_draws"),
    NURI_COUNTER("renderer.opaque.msaa_depth_prepass_dispatches"),
    NURI_COUNTER("renderer.opaque.gtao_auxiliary_prepass_draws"),
    NURI_COUNTER("renderer.opaque.gtao_auxiliary_prepass_dispatches"),
    NURI_COUNTER("renderer.opaque.gtao_auxiliary_writes_single_sample_depth"),
    NURI_COUNTER("renderer.opaque.main_equal_readonly_draws"),
    NURI_COUNTER("renderer.opaque.main_equal_readonly_dispatches"),
    NURI_COUNTER("renderer.opaque.main_less_write_draws"),
    NURI_COUNTER("renderer.opaque.main_less_write_dispatches"),
    NURI_COUNTER("renderer.opaque.pipeline_statistics_requested"),
    NURI_COUNTER("renderer.opaque.pipeline_statistics_available"),
    NURI_EXACT_METRIC(
        "renderer.opaque.pipeline_statistics_input_assembly_vertices",
        Unit::Count, Numeric::Uint64EncodedAsFloat64, Direction::Informational,
        Availability::WhenOpaquePipelineStatisticsAvailable,
        Phase::DelayedGpuReadback, Gate::Diagnostic),
    NURI_COUNTER("renderer.backend.render_pipeline_creations"),
    NURI_COUNTER("renderer.backend.compute_pipeline_creations"),
    NURI_COUNTER("renderer.backend.meshlet_pipeline_creations"),
    NURI_COUNTER("renderer.backend.framebuffer_creations"),
    NURI_EXACT_METRIC(
        "renderer.opaque.pipeline_statistics_input_assembly_primitives",
        Unit::Count, Numeric::Uint64EncodedAsFloat64, Direction::Informational,
        Availability::WhenOpaquePipelineStatisticsAvailable,
        Phase::DelayedGpuReadback, Gate::Diagnostic),
    NURI_EXACT_METRIC(
        "renderer.opaque.pipeline_statistics_clipping_invocations", Unit::Count,
        Numeric::Uint64EncodedAsFloat64, Direction::Informational,
        Availability::WhenOpaquePipelineStatisticsAvailable,
        Phase::DelayedGpuReadback, Gate::Diagnostic),
    NURI_EXACT_METRIC("renderer.opaque.pipeline_statistics_clipping_primitives",
                      Unit::Count, Numeric::Uint64EncodedAsFloat64,
                      Direction::Informational,
                      Availability::WhenOpaquePipelineStatisticsAvailable,
                      Phase::DelayedGpuReadback, Gate::Diagnostic),
    NURI_EXACT_METRIC(
        "renderer.opaque.pipeline_statistics_fragment_shader_invocations",
        Unit::Count, Numeric::Uint64EncodedAsFloat64, Direction::Informational,
        Availability::WhenOpaquePipelineStatisticsAvailable,
        Phase::DelayedGpuReadback, Gate::Diagnostic),
    NURI_COUNTER("renderer.opaque.depth_pyramid_requested"),
    NURI_COUNTER("renderer.opaque.depth_pyramid_active"),
    NURI_COUNTER("renderer.visibility.hiz_requested"),
    NURI_COUNTER("renderer.visibility.hiz_active"),
    NURI_COUNTER("renderer.visibility.hiz_source_frame_policy"),
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
    NURI_COUNTER("renderer.visibility.meshlet_occlusion_mode"),
    NURI_COUNTER("renderer.visibility.meshlet_occlusion_source_frame"),
    NURI_COUNTER("renderer.visibility.meshlet_occlusion_source_age"),
    NURI_COUNTER("renderer.visibility.current_frame_hiz_active"),
    NURI_COUNTER("renderer.visibility.meshlet_pre_task_compaction_active"),
    NURI_COUNTER("renderer.visibility.meshlet_pre_task_candidates_input"),
    NURI_COUNTER("renderer.visibility.meshlet_pre_task_candidates_output"),
    NURI_COUNTER("renderer.visibility.meshlet_pre_task_task_groups_input"),
    NURI_COUNTER("renderer.visibility.meshlet_pre_task_task_groups_output"),
    NURI_COUNTER("renderer.visibility.meshlet_pre_task_task_groups_saved"),
    NURI_COUNTER("renderer.visibility.meshlet_pre_task_overflow_count"),
    NURI_COUNTER("renderer.visibility.meshlet_pre_task_mismatch_count"),
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
    NURI_COUNTER("renderer.visibility.occlusion_available"),
    NURI_COUNTER("renderer.shadow.cascades"),
    NURI_COUNTER("renderer.shadow.total_draws"),
    NURI_COUNTER("renderer.shadow.total_culled_draws"),
    NURI_COUNTER("renderer.shadow.static_caster_entries"),
    NURI_COUNTER("renderer.shadow.dynamic_caster_entries"),
    NURI_COUNTER("renderer.shadow.static_batch_templates"),
    NURI_COUNTER("renderer.shadow.batch_entries"),
    NURI_COUNTER("renderer.shadow.instance_remaps"),
    NURI_COUNTER("renderer.shadow.total_index_count_estimate"),
    NURI_COUNTER("renderer.shadow.submitted_draw_items"),
    NURI_COUNTER("renderer.shadow.indirect_commands"),
    NURI_COUNTER("renderer.shadow.draw_packet_bytes"),
    NURI_COUNTER("renderer.shadow.filter_sample_budget"),
    NURI_COUNTER("renderer.shadow.frame_gpu_bytes"),
    NURI_COUNTER("renderer.shadow.sdsm_compute_passes"),
    NURI_COUNTER("renderer.aa.post_aa_requested"),
    NURI_COUNTER("renderer.aa.post_aa_resolved_active"),
    NURI_COUNTER("renderer.aa.post_aa_inactive_reason"),
    NURI_COUNTER("renderer.aa.post_aa_specular_algorithm"),
    NURI_COUNTER("renderer.aa.post_aa_spatial_algorithm"),
    NURI_RATIO("renderer.aa.post_aa_material_variance_scale"),
    NURI_RATIO("renderer.aa.post_aa_geometric_variance_scale"),
    NURI_RATIO("renderer.aa.post_aa_max_slope_variance"),
    NURI_COUNTER("renderer.aa.post_aa_specular_selected"),
    NURI_COUNTER("renderer.aa.post_aa_smaa_planned"),
    NURI_COUNTER("renderer.aa.post_aa_smaa_submitted"),
    NURI_COUNTER("renderer.aa.post_aa_smaa_submitted_passes"),
    NURI_COUNTER("renderer.aa.post_aa_smaa_completed"),
    NURI_EXACT_METRIC(
        "renderer.aa.post_aa_smaa_completed_source_frame", Unit::Count,
        Numeric::Uint64EncodedAsFloat64, Direction::Informational,
        Availability::WhenGpuTimingAvailable, Phase::DelayedGpuReadback,
        Gate::WorkloadCharacterization),
    NURI_COUNTER("renderer.aa.post_aa_degradation_mask"),
    NURI_COUNTER("renderer.aa.resolved_material_specular_aa"),
    NURI_COUNTER("renderer.aa.debug_view"),
    NURI_COUNTER("renderer.aa.specular_aa_debug_override"),
    NURI_COUNTER("renderer.aa.normal_variance_contract_materials_live"),
    NURI_COUNTER("renderer.aa.normal_variance_contract_textures_live"),
    NURI_COUNTER("renderer.aa.normal_variance_unavailable_slots_live"),
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
    NURI_COUNTER("renderer.aa.msaa_color_resolves"),
    NURI_COUNTER("renderer.aa.msaa_depth_resolves"),
    NURI_COUNTER("renderer.aa.msaa_resolved_sample_count"),
    NURI_COUNTER("renderer.aa.msaa_sample_count"),
    NURI_COUNTER("renderer.aa.msaa_color_textures"),
    NURI_COUNTER("renderer.aa.msaa_depth_textures"),
    NURI_COUNTER("renderer.aa.msaa_ring_slots"),
    NURI_COUNTER("renderer.aa.msaa_color_allocations"),
    NURI_COUNTER("renderer.aa.msaa_color_reallocations"),
    NURI_COUNTER("renderer.aa.msaa_depth_allocations"),
    NURI_COUNTER("renderer.aa.msaa_depth_reallocations"),
    NURI_COUNTER("renderer.aa.spatial_aa_allocations"),
    NURI_COUNTER("renderer.aa.spatial_aa_reallocations"),
    NURI_COUNTER("renderer.aa.msaa_sample4_color_supported"),
    NURI_COUNTER("renderer.aa.msaa_sample4_depth_supported"),
    NURI_COUNTER("renderer.aa.msaa_sample8_color_supported"),
    NURI_COUNTER("renderer.aa.msaa_sample8_depth_supported"),
    NURI_COUNTER("renderer.aa.msaa_depth_resolve_min_supported"),
    NURI_COUNTER("renderer.aa.msaa_alpha_to_coverage_supported"),
    NURI_COUNTER("renderer.aa.msaa_sample_rate_shading_supported"),
    NURI_COUNTER("renderer.aa.msaa_alpha_to_coverage_active"),
    NURI_COUNTER("renderer.aa.msaa_sample_shading_active"),
    NURI_COUNTER("renderer.aa.msaa_alpha_coverage_requested"),
    NURI_COUNTER("renderer.aa.msaa_spatial_cleanup_requested"),
    NURI_COUNTER("renderer.aa.msaa_spatial_cleanup_active"),
    NURI_COUNTER("renderer.aa.msaa_unsupported_reason"),
    NURI_COUNTER("renderer.aa.msaa_alpha_coverage_policy"),
    NURI_COUNTER("renderer.aa.msaa_transparency_policy"),
    NURI_COUNTER("renderer.aa.msaa_resolve_placement"),
    NURI_COUNTER("renderer.aa.msaa_main_color_format"),
    NURI_COUNTER("renderer.aa.msaa_main_depth_format"),
    NURI_COUNTER("renderer.aa.msaa_main_attachment_sample_count"),
    NURI_COUNTER("renderer.aa.msaa_extent_width"),
    NURI_COUNTER("renderer.aa.msaa_extent_height"),
    NURI_COUNTER("renderer.aa.msaa_color_texel_bytes"),
    NURI_COUNTER("renderer.aa.msaa_depth_texel_bytes"),
    NURI_COUNTER("renderer.aa.msaa_traffic_formula_version"),
    NURI_COUNTER("renderer.aa.motion_class_textures"),
    NURI_COUNTER("renderer.aa.motion_class_coverage_available"),
    NURI_EXACT_METRIC("renderer.aa.motion_class_invalid_pixels", Unit::Count,
                      Numeric::Uint64EncodedAsFloat64, Direction::Informational,
                      Availability::WhenMotionClassCoverageAvailable,
                      Phase::DelayedGpuReadback,
                      Gate::WorkloadCharacterization),
    NURI_EXACT_METRIC(
        "renderer.aa.motion_class_static_camera_only_pixels", Unit::Count,
        Numeric::Uint64EncodedAsFloat64, Direction::Informational,
        Availability::WhenMotionClassCoverageAvailable,
        Phase::DelayedGpuReadback, Gate::WorkloadCharacterization),
    NURI_EXACT_METRIC("renderer.aa.motion_class_full_pixels", Unit::Count,
                      Numeric::Uint64EncodedAsFloat64, Direction::Informational,
                      Availability::WhenMotionClassCoverageAvailable,
                      Phase::DelayedGpuReadback,
                      Gate::WorkloadCharacterization),
    NURI_EXACT_METRIC(
        "renderer.aa.motion_class_background_rotation_pixels", Unit::Count,
        Numeric::Uint64EncodedAsFloat64, Direction::Informational,
        Availability::WhenMotionClassCoverageAvailable,
        Phase::DelayedGpuReadback, Gate::WorkloadCharacterization),
    NURI_EXACT_METRIC("renderer.aa.motion_class_invalid_ratio", Unit::Ratio,
                      Numeric::Float64, Direction::Informational,
                      Availability::WhenMotionClassCoverageAvailable,
                      Phase::DelayedGpuReadback,
                      Gate::WorkloadCharacterization),
    NURI_EXACT_METRIC("renderer.aa.motion_class_static_camera_only_ratio",
                      Unit::Ratio, Numeric::Float64, Direction::Informational,
                      Availability::WhenMotionClassCoverageAvailable,
                      Phase::DelayedGpuReadback,
                      Gate::WorkloadCharacterization),
    NURI_EXACT_METRIC("renderer.aa.motion_class_full_ratio", Unit::Ratio,
                      Numeric::Float64, Direction::Informational,
                      Availability::WhenMotionClassCoverageAvailable,
                      Phase::DelayedGpuReadback,
                      Gate::WorkloadCharacterization),
    NURI_EXACT_METRIC("renderer.aa.motion_class_background_rotation_ratio",
                      Unit::Ratio, Numeric::Float64, Direction::Informational,
                      Availability::WhenMotionClassCoverageAvailable,
                      Phase::DelayedGpuReadback,
                      Gate::WorkloadCharacterization),
    NURI_COUNTER("renderer.aa.history_color_textures"),
    NURI_COUNTER("renderer.aa.transparent_transmission_blend_draws"),
    NURI_COUNTER("renderer.aa.transparent_transmission_feedback_refreshes"),
    NURI_COUNTER("renderer.aa.transparent_transmission_feedback_available"),
    NURI_COUNTER("renderer.ao.normal_texture_count"),
    NURI_COUNTER("renderer.ao.history_texture_count"),
    NURI_COUNTER("renderer.ao.input_mode"),
    NURI_COUNTER("renderer.ao.working_resolution"),
    NURI_COUNTER("renderer.ao.output_width"),
    NURI_COUNTER("renderer.ao.output_height"),
    NURI_COUNTER("renderer.ao.working_width"),
    NURI_COUNTER("renderer.ao.working_height"),
    NURI_COUNTER("renderer.ao.working_pixel_count"),
    NURI_COUNTER("renderer.ao.input_pass_draws"),
    NURI_COUNTER("renderer.ao.normal_prepass_draws"),
    NURI_COUNTER("renderer.ao.depth_prefilter_passes"),
    NURI_COUNTER("renderer.ao.main_passes"),
    NURI_COUNTER("renderer.ao.temporal_passes"),
    NURI_COUNTER("renderer.ao.temporal_motion_class_consumed"),
    NURI_COUNTER("renderer.ao.temporal_reactive_mask_consumed"),
    NURI_COUNTER("renderer.ao.temporal_previous_depth_consumed"),
    NURI_COUNTER("renderer.ao.texture_count"),
    NURI_MEMORY("gpu.memory.ao.scratch_texture_mb",
                Availability::EveryMeasuredFrame),
    NURI_MEMORY("renderer.ao.allocated_texture_mb",
                Availability::EveryMeasuredFrame),
    NURI_MEMORY("renderer.ao.logical_active_texture_mb",
                Availability::EveryMeasuredFrame),
    NURI_MEMORY("renderer.ao.provider_texture_mb",
                Availability::EveryMeasuredFrame),
    NURI_MEMORY("renderer.ao.feature_texture_mb",
                Availability::EveryMeasuredFrame),
    NURI_COUNTER("renderer.hdr.bloom_passes"),
    NURI_COUNTER("renderer.hdr.luminance_passes"),
    NURI_COUNTER("renderer.hdr.adaptation_passes"),
    NURI_COUNTER("renderer.hdr.texture_count"),
    NURI_COUNTER("renderer.hdr.exposure_telemetry_available"),
    NURI_COUNTER("renderer.hdr.exposure_telemetry_source_frame"),
    NURI_COUNTER("renderer.hdr.exposure_telemetry_stale_frames"),
    NURI_COUNTER("renderer.hdr.exposure_telemetry_pending_slots"),
    NURI_COUNTER("renderer.hdr.exposure_telemetry_dropped_samples"),
    NURI_WORLD("renderer.hdr.automatic_exposure_ev"),
    NURI_WORLD("renderer.hdr.exposure_target_ev"),
    NURI_WORLD("renderer.hdr.exposure_metered_luminance"),
    NURI_WORLD("renderer.hdr.effective_exposure_ev"),
    NURI_RATIO("renderer.hdr.exposure_invalid_sample_fraction"),
    NURI_COUNTER("renderer.transparent.mesh_draws"),
    NURI_COUNTER("renderer.transparent.contributor_sortable_draws"),
    NURI_COUNTER("renderer.transparent.contributor_fixed_draws"),
    NURI_COUNTER("renderer.transparent.pick_draws"),

    NURI_COUNTER("renderer.ray_tracing.static_instances"),
    NURI_COUNTER("renderer.ray_tracing.dynamic_instances"),
    NURI_COUNTER("renderer.ray_tracing.excluded_dynamic_instances"),
    NURI_COUNTER("renderer.ray_tracing.static_blas_count"),
    NURI_COUNTER("renderer.ray_tracing.dynamic_blas_count"),
    NURI_COUNTER("renderer.ray_tracing.tlas_count"),
    NURI_COUNTER("renderer.ray_tracing.unique_static_geometry"),
    NURI_COUNTER("renderer.ray_tracing.geometry_records"),
    NURI_COUNTER("renderer.ray_tracing.triangles"),
    NURI_COUNTER("renderer.ray_tracing.queued_blas_builds"),
    NURI_COUNTER("renderer.ray_tracing.decoded_vertices"),
    NURI_COUNTER("renderer.ray_tracing.decode_dispatches"),
    NURI_COUNTER("renderer.ray_tracing.blas_builds"),
    NURI_COUNTER("renderer.ray_tracing.tlas_builds"),
    NURI_COUNTER("renderer.ray_tracing.tlas_updates"),
    NURI_COUNTER("renderer.ray_tracing.dynamic_blas_updates"),
    NURI_COUNTER("renderer.ray_tracing.dynamic_vertex_dispatches"),
    NURI_COUNTER("renderer.ray_tracing.no_as_work_frame"),
    NURI_COUNTER("renderer.ray_tracing.indirect_submission_references"),
    NURI_COUNTER("renderer.ray_tracing.indirect_texture_references"),
    NURI_COUNTER(
        "renderer.ray_tracing.graph_acceleration_structure_dependencies"),
    NURI_COUNTER("renderer.ray_tracing.readiness"),
    NURI_COUNTER("renderer.ray_tracing.consumed_rebuild_epoch"),
    NURI_MEMORY("gpu.memory.ray_tracing.decoded_positions_mb",
                Availability::EveryMeasuredFrame),
    NURI_MEMORY("gpu.memory.ray_tracing.tables_mb",
                Availability::EveryMeasuredFrame),
    NURI_MEMORY("gpu.memory.ray_tracing.blas_mb",
                Availability::EveryMeasuredFrame),
    NURI_MEMORY("gpu.memory.ray_tracing.tlas_mb",
                Availability::EveryMeasuredFrame),
    NURI_MEMORY("gpu.memory.ray_tracing.as_scratch_high_water_mb",
                Availability::EveryMeasuredFrame),
    NURI_MEMORY("gpu.memory.ray_tracing.as_scratch_current_mb",
                Availability::EveryMeasuredFrame),
    NURI_COUNTER("renderer.ray_tracing.direct_binding_pool_high_water"),
    NURI_COUNTER("renderer.ddgi.requested"),
    NURI_COUNTER("renderer.ddgi.active"),
    NURI_COUNTER("renderer.ddgi.active_volumes"),
    NURI_COUNTER("renderer.ddgi.ready_volumes"),
    NURI_COUNTER("renderer.ddgi.total_probes"),
    NURI_COUNTER("renderer.ddgi.vigilant_probes"),
    NURI_COUNTER("renderer.ddgi.uninitialized_probes"),
    NURI_COUNTER("renderer.ddgi.off_probes"),
    NURI_COUNTER("renderer.ddgi.sleeping_probes"),
    NURI_COUNTER("renderer.ddgi.newly_awake_probes"),
    NURI_COUNTER("renderer.ddgi.awake_probes"),
    NURI_COUNTER("renderer.ddgi.newly_vigilant_probes"),
    NURI_COUNTER("renderer.ddgi.relocated_probes"),
    NURI_COUNTER("renderer.ddgi.probe_state_readback_available"),
    NURI_COUNTER("renderer.ddgi.probe_state_readback_source_frame"),
    NURI_COUNTER("renderer.ddgi.probe_state_readback_stale_frames"),
    NURI_COUNTER("renderer.ddgi.max_relocation"),
    NURI_COUNTER("renderer.ddgi.updated_probes"),
    NURI_COUNTER("renderer.ddgi.primary_queries"),
    NURI_COUNTER("renderer.ddgi.classification_probe_updates"),
    NURI_COUNTER("renderer.ddgi.classification_primary_queries"),
    NURI_COUNTER("renderer.ddgi.irradiance_primary_queries"),
    NURI_COUNTER("renderer.ddgi.primary_queries_issued"),
    NURI_COUNTER("renderer.ddgi.trace_counters_available"),
    NURI_COUNTER("renderer.ddgi.trace_counter_source_frame"),
    NURI_COUNTER("renderer.ddgi.trace_counter_stale_frames"),
    NURI_COUNTER("renderer.ddgi.readback_waits"),
    NURI_COUNTER("renderer.ddgi.readback_copy_bytes"),
    NURI_COUNTER("renderer.ddgi.readback_pending_slots"),
    NURI_COUNTER("renderer.ddgi.readback_dropped_samples"),
    NURI_COUNTER("renderer.ddgi.readback_oldest_pending_age"),
    NURI_COUNTER("renderer.ddgi.readback_blocking_fallbacks"),
    NURI_COUNTER("renderer.ddgi.readback_generation_mismatches"),
    NURI_COUNTER("renderer.ddgi.readback_early_reuse_attempts"),
    NURI_COUNTER("renderer.ddgi.secondary_queries_reserved"),
    NURI_COUNTER("renderer.ddgi.secondary_queries_unused"),
    NURI_COUNTER("renderer.ddgi.secondary_query_budget_overflows"),
    NURI_COUNTER("renderer.ddgi.secondary_queries"),
    NURI_COUNTER("renderer.ddgi.directional_secondary_queries"),
    NURI_COUNTER("renderer.ddgi.local_secondary_queries"),
    NURI_COUNTER("renderer.ddgi.total_queries_issued"),
    NURI_DIAGNOSTIC_COUNTER("renderer.ddgi.primary_candidate_intersections"),
    NURI_DIAGNOSTIC_COUNTER("renderer.ddgi.secondary_candidate_intersections"),
    NURI_DIAGNOSTIC_COUNTER("renderer.ddgi.alpha_candidate_rejections"),
    NURI_DIAGNOSTIC_COUNTER("renderer.ddgi.backface_candidate_rejections"),
    NURI_DIAGNOSTIC_COUNTER("renderer.ddgi.candidate_overflows"),
    NURI_DIAGNOSTIC_COUNTER("renderer.ddgi.local_light_truncations"),
    NURI_DIAGNOSTIC_COUNTER("renderer.ddgi.non_finite_radiance_rejects"),
    NURI_DIAGNOSTIC_COUNTER("renderer.ddgi.emissive_radiance_clamps"),
    NURI_DIAGNOSTIC_COUNTER("renderer.ddgi.direct_radiance_clamps"),
    NURI_DIAGNOSTIC_COUNTER("renderer.ddgi.sky_radiance_clamps"),
    NURI_DIAGNOSTIC_COUNTER("renderer.ddgi.multi_bounce_radiance_clamps"),
    NURI_DIAGNOSTIC_COUNTER("renderer.ddgi.final_radiance_clamps"),
    NURI_COUNTER("renderer.ddgi.diagnostic_counters_enabled"),
    NURI_COUNTER("renderer.ddgi.quality_schema"),
    NURI_COUNTER("renderer.ddgi.requested_quality_preset"),
    NURI_COUNTER("renderer.ddgi.quality_preset"),
    NURI_COUNTER("renderer.ddgi.coverage_preset_schema"),
    NURI_COUNTER("renderer.ddgi.requested_coverage_preset"),
    NURI_COUNTER("renderer.ddgi.coverage_preset"),
    NURI_COUNTER("renderer.ddgi.product_profile_schema"),
    NURI_COUNTER("renderer.ddgi.gather_identity_schema"),
    NURI_COUNTER("renderer.ddgi.opaque_gather_architecture"),
    NURI_COUNTER("renderer.ddgi.opaque_gather_variant"),
    NURI_COUNTER("renderer.ddgi.transmission_gather_architecture"),
    NURI_COUNTER("renderer.ddgi.transmission_gather_variant"),
    NURI_COUNTER("renderer.ddgi.trace_multibounce_gather_architecture"),
    NURI_COUNTER("renderer.ddgi.trace_multibounce_gather_variant"),
    NURI_COUNTER("renderer.ddgi.surface_gather_architecture"),
    NURI_COUNTER("renderer.ddgi.surface_gather_width"),
    NURI_COUNTER("renderer.ddgi.surface_gather_height"),
    NURI_COUNTER("renderer.ddgi.surface_gather_max_candidate_volumes"),
    NURI_COUNTER("renderer.ddgi.surface_gather_max_sampled_volumes"),
    NURI_COUNTER("renderer.ddgi.surface_gather_max_state_loads_per_pixel"),
    NURI_COUNTER("renderer.ddgi.surface_gather_max_atlas_samples_per_pixel"),
    NURI_COUNTER("renderer.ddgi.surface_cache_format"),
    NURI_COUNTER("renderer.ddgi.surface_cache_bytes"),
    NURI_COUNTER("renderer.ddgi.ray_query_capacity"),
    NURI_COUNTER("renderer.ddgi.probe_update_capacity"),
    NURI_COUNTER("renderer.ddgi.requested_probe_update_capacity"),
    NURI_COUNTER("renderer.ddgi.effective_probe_update_capacity"),
    NURI_COUNTER("renderer.ddgi.requested_maintenance_probe_update_capacity"),
    NURI_COUNTER("renderer.ddgi.effective_maintenance_probe_update_capacity"),
    NURI_COUNTER("renderer.ddgi.maintenance_probe_updates"),
    NURI_COUNTER("renderer.ddgi.primary_result_capacity"),
    NURI_COUNTER("renderer.ddgi.trace_dispatches"),
    NURI_COUNTER("renderer.ddgi.trace_launched_lanes"),
    NURI_COUNTER("renderer.ddgi.trace_useful_lanes"),
    NURI_COUNTER("renderer.ddgi.classification_launched_lanes"),
    NURI_COUNTER("renderer.ddgi.classification_useful_lanes"),
    NURI_COUNTER("renderer.ddgi.irradiance_atlas_dispatches"),
    NURI_COUNTER("renderer.ddgi.distance_atlas_dispatches"),
    NURI_COUNTER("renderer.ddgi.irradiance_result_visits"),
    NURI_COUNTER("renderer.ddgi.distance_result_visits"),
    NURI_COUNTER("renderer.ddgi.irradiance_texel_writes"),
    NURI_COUNTER("renderer.ddgi.distance_texel_writes"),
    NURI_COUNTER("renderer.ddgi.update_reason_bits"),
    NURI_COUNTER("renderer.ddgi.reason.bootstrap"),
    NURI_COUNTER("renderer.ddgi.reason.scroll"),
    NURI_COUNTER("renderer.ddgi.reason.dirty_geometry"),
    NURI_COUNTER("renderer.ddgi.reason.radiometric_response"),
    NURI_COUNTER("renderer.ddgi.reason.maintenance"),
    NURI_COUNTER("renderer.ddgi.reason.force"),
    NURI_COUNTER("renderer.ddgi.reason.wake"),
    NURI_COUNTER("renderer.ddgi.reason.reclassification"),
    NURI_COUNTER("renderer.ddgi.startup_phase"),
    NURI_RATIO("renderer.ddgi.sky_remainder_over_threshold_percentage"),
    NURI_COUNTER("renderer.ddgi.reset_count"),
    NURI_COUNTER("renderer.ddgi.scroll_count"),
    NURI_COUNTER("renderer.ddgi.invalidated_probes"),
    NURI_COUNTER("renderer.ddgi.failed_volumes"),
    NURI_COUNTER("renderer.ddgi.effective_volumes"),
    NURI_COUNTER("renderer.ddgi.authored_volumes"),
    NURI_COUNTER("renderer.ddgi.generated_volumes"),
    NURI_COUNTER("renderer.ddgi.redundant_authored_volumes"),
    NURI_COUNTER("renderer.ddgi.redundant_authored_probes"),
    NURI_COUNTER("renderer.ddgi.coverage_mode"),
    NURI_COUNTER("renderer.ddgi.coverage_solve_executions"),
    NURI_COUNTER("renderer.ddgi.coverage_plan_cache_hits"),
    NURI_COUNTER("renderer.ddgi.state_history_scan_count"),
    NURI_DIAGNOSTIC_CPU_COUNTER("renderer.ddgi.age_sample_count"),
    NURI_DIAGNOSTIC_CPU_COUNTER("renderer.ddgi.age_selection_count"),
    NURI_DIAGNOSTIC_CPU_COUNTER("renderer.ddgi.coverage_lattice_evaluations"),
    NURI_COUNTER("renderer.ddgi.upload_submission_count"),
    NURI_DIAGNOSTIC_CPU_COUNTER("renderer.ddgi.light_difference_comparisons"),
    NURI_COUNTER("renderer.ddgi.coverage_status"),
    NURI_COUNTER("renderer.ddgi.coverage_error"),
    NURI_COUNTER("renderer.ddgi.limiting_constraint"),
    NURI_WORLD("renderer.ddgi.requested_half_extent_x"),
    NURI_WORLD("renderer.ddgi.requested_half_extent_y"),
    NURI_WORLD("renderer.ddgi.requested_half_extent_z"),
    NURI_WORLD("renderer.ddgi.achieved_half_extent_x"),
    NURI_WORLD("renderer.ddgi.achieved_half_extent_y"),
    NURI_WORLD("renderer.ddgi.achieved_half_extent_z"),
    NURI_RATIO("renderer.ddgi.scene_coverage_ratio"),
    NURI_POST_CPU_TIMING("renderer.ddgi.coverage_resolve_cpu_ms"),
    NURI_POST_CPU_TIMING("cpu.ddgi.prepare_ms"),
    NURI_POST_CPU_TIMING("cpu.ddgi.schedule_ms"),
    NURI_POST_CPU_TIMING("cpu.ddgi.graph_build_ms"),
    NURI_POST_CPU_TIMING("cpu.ddgi.readback_poll_ms"),
    NURI_COUNTER("renderer.ddgi.diagnostic_sample_count"),
    NURI_COUNTER("renderer.ddgi.uncovered_diagnostic_samples"),
    NURI_COUNTER("renderer.ddgi.sky_remainder_samples"),
    NURI_COUNTER("renderer.ddgi.diagnostic_samples_available"),
    NURI_COUNTER("renderer.ddgi.dirty_regions_produced"),
    NURI_COUNTER("renderer.ddgi.dirty_regions_merged"),
    NURI_COUNTER("renderer.ddgi.dirty_regions_overflowed"),
    NURI_COUNTER("renderer.ddgi.dirty_regions_pending"),
    NURI_COUNTER("renderer.ddgi.dirty_probes_affected"),
    NURI_COUNTER("renderer.ddgi.classification_fallbacks"),
    NURI_COUNTER("renderer.ddgi.classification_overflows"),
    NURI_COUNTER("renderer.ddgi.volume_failure_reason"),
    NURI_COUNTER("renderer.ddgi.history_ready"),
    NURI_COUNTER("renderer.ddgi.irradiance_response_remaining"),
    NURI_COUNTER("renderer.ddgi.distance_response_remaining"),
    NURI_COUNTER("renderer.ddgi.inspection_available"),
    NURI_COUNTER("renderer.ddgi.inspection_valid"),
    NURI_COUNTER("renderer.ddgi.inspection_ray_count"),
    NURI_COUNTER("renderer.ddgi.inspection_hit_count"),
    NURI_COUNTER("renderer.ddgi.inspection_miss_count"),
    NURI_COUNTER("renderer.ddgi.inspection_candidate_overflows"),
    NURI_COUNTER("renderer.ddgi.inspection_event_overflows"),
    NURI_COUNTER("renderer.ddgi.sky_fallback_active"),
    NURI_COUNTER("renderer.ddgi.fallback_reason"),
    NURI_COUNTER("renderer.ddgi.submitted_sequence"),
    NURI_COUNTER("renderer.ddgi.layout_generation"),
    NURI_COUNTER("renderer.ddgi.resource_generation"),
    NURI_COUNTER("renderer.ddgi.device_epoch"),
    NURI_COUNTER("renderer.ddgi.consumed_reset_epoch"),
    NURI_COUNTER("renderer.ddgi.consumed_force_update_epoch"),
    NURI_MEMORY("gpu.memory.ddgi.persistent_mb",
                Availability::EveryMeasuredFrame),
    NURI_MEMORY("gpu.memory.ddgi.frame_batch_mb",
                Availability::EveryMeasuredFrame),
    NURI_MEMORY("gpu.memory.ddgi.frame_ring_device_mb",
                Availability::EveryMeasuredFrame),
    NURI_MEMORY("gpu.memory.ddgi.frame_ring_readback_mb",
                Availability::EveryMeasuredFrame),
    NURI_MEMORY("gpu.memory.ddgi.frame_ring_total_mb",
                Availability::EveryMeasuredFrame),
    NURI_COUNTER("renderer.ddgi.frame_slot_count"),
    NURI_MEMORY("gpu.memory.ddgi.committed_atlas_mb",
                Availability::EveryMeasuredFrame),
    NURI_MEMORY("gpu.memory.ddgi.pending_atlas_mb",
                Availability::EveryMeasuredFrame),
    NURI_MEMORY("gpu.memory.ddgi.peak_atlas_mb",
                Availability::EveryMeasuredFrame),
    NURI_MEMORY("gpu.memory.ddgi.redundant_authored_mb",
                Availability::EveryMeasuredFrame),
    NURI_DDGI_VOLUME_METRICS(0),
    NURI_DDGI_VOLUME_METRICS(1),
    NURI_DDGI_VOLUME_METRICS(2),
    NURI_DDGI_VOLUME_METRICS(3),
    NURI_DDGI_VOLUME_METRICS(4),
    NURI_DDGI_VOLUME_METRICS(5),
    NURI_DDGI_VOLUME_METRICS(6),
    NURI_DDGI_VOLUME_METRICS(7),

    NURI_RENDERGRAPH_COUNTER("rendergraph.summary.declared_pass_count"),
    NURI_RENDERGRAPH_COUNTER("rendergraph.summary.culled_pass_count"),
    NURI_RENDERGRAPH_COUNTER("rendergraph.summary.root_pass_count"),
    NURI_RENDERGRAPH_COUNTER("rendergraph.summary.pass_count"),
    NURI_RENDERGRAPH_COUNTER("rendergraph.summary.edge_count"),
    NURI_RENDERGRAPH_COUNTER(
        "rendergraph.summary.recorded_graphics_pass_count"),
    NURI_RENDERGRAPH_COUNTER("rendergraph.summary.pass_barrier_plan_count"),
    NURI_RENDERGRAPH_COUNTER("rendergraph.summary.final_barrier_record_count"),
    NURI_RENDERGRAPH_COUNTER("rendergraph.summary.pass_barrier_record_count"),
    NURI_RENDERGRAPH_COUNTER(
        "rendergraph.summary.recorded_command_buffer_count"),
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
        "rendergraph.summary.command_resource_patch_count"),
    NURI_RENDERGRAPH_COUNTER("rendergraph.summary.owned_pre_dispatch_count"),
    NURI_RENDERGRAPH_COUNTER("rendergraph.summary.owned_draw_item_count"),
    NURI_RENDERGRAPH_COUNTER(
        "rendergraph.summary.owned_mesh_dispatch_item_count"),

    {"rendergraph.pass.<index>.<pass>.cpu_ms", Rule::RenderGraphPassCpuTiming,
     Unit::Milliseconds, Numeric::Float64, Direction::LowerIsBetter,
     kFrameDistribution, Availability::WhenRenderGraphTelemetryAvailable,
     Phase::PostRenderMeasuredFrame, Gate::Eligible},
    {"rendergraph.pass.<index>.<pass>.gpu_ms", Rule::RenderGraphPassGpuTiming,
     Unit::Milliseconds, Numeric::Float64, Direction::LowerIsBetter,
     kFrameDistribution, Availability::WhenGpuTimingAvailable,
     Phase::DelayedGpuReadback, Gate::Eligible},
};

#undef NURI_RENDERGRAPH_COUNTER
#undef NURI_ASSET_TIMING
#undef NURI_COUNTER
#undef NURI_DDGI_VOLUME_METRICS
#undef NURI_POST_CPU_TIMING
#undef NURI_WORLD
#undef NURI_RATIO
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

[[nodiscard]] bool
matchesRenderGraphPassTiming(std::string_view metricId,
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
  return std::all_of(index.begin(), index.end(),
                     [](char character) {
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
  static const size_t count = static_cast<size_t>(
      std::count_if(std::begin(kDescriptors), std::end(kDescriptors),
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

std::string_view
benchmarkMetricIdRuleName(BenchmarkMetricIdRule rule) noexcept {
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
  case Unit::Ratio:
    return "ratio";
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
    return "median-p95-p99-across-measured-frames";
  }
  return "median-p95-p99-across-measured-frames";
}

std::string_view benchmarkMetricAvailabilityName(
    BenchmarkMetricAvailability availability) noexcept {
  switch (availability) {
  case Availability::EveryMeasuredFrame:
    return "every-measured-frame";
  case Availability::WhenGpuTimingAvailable:
    return "when-gpu-timing-available";
  case Availability::WhenWholeFrameGpuTimingAvailable:
    return "when-whole-frame-gpu-timing-available";
  case Availability::WhenRenderGraphTelemetryAvailable:
    return "when-rendergraph-telemetry-available";
  case Availability::WhenPlatformMemoryAvailable:
    return "when-platform-memory-available";
  case Availability::WhenMotionClassCoverageAvailable:
    return "when-motion-class-coverage-available";
  case Availability::WhenOpaquePipelineStatisticsAvailable:
    return "when-opaque-pipeline-statistics-available";
  }
  return "every-measured-frame";
}

std::string_view
benchmarkMetricSamplingPhaseName(BenchmarkMetricSamplingPhase phase) noexcept {
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

std::string_view benchmarkMetricEvidenceClassName(
    BenchmarkMetricEvidenceClass evidenceClass) noexcept {
  switch (evidenceClass) {
  case Evidence::ProductSafe:
    return "product-safe";
  case Evidence::DiagnosticOnly:
    return "diagnostic-only";
  case Evidence::Derived:
    return "derived";
  }
  return "product-safe";
}

} // namespace nuri::tools::benchmark
