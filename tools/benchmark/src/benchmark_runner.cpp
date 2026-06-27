#include "nuri/tools/benchmark/benchmark_runner.h"

#include "nuri/core/log.h"
#include "nuri/core/runtime_config.h"
#include "nuri/core/window.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/pipeline/default_render_pipeline.h"
#include "nuri/gfx/pipeline/render_pipeline.h"
#include "nuri/gfx/renderer.h"
#include "nuri/resources/gpu/resource_manager.h"
#include "nuri/resources/scene_importer.h"
#include "nuri/scene/camera.h"
#include "nuri/scene/render_scene.h"
#include "nuri/tools/benchmark/benchmark_environment.h"
#include "nuri/tools/benchmark/benchmark_manifest.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <iostream>
#include <map>
#include <memory_resource>
#include <limits>
#include <optional>
#include <sstream>
#include <thread>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef PSAPI_VERSION
#define PSAPI_VERSION 2
#endif
#include <stdlib.h>
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

namespace nuri::tools::benchmark {
namespace {

constexpr double kBytesPerMiB = 1024.0 * 1024.0;

[[nodiscard]] double bytesToMiB(uint64_t bytes) {
  return static_cast<double>(bytes) / kBytesPerMiB;
}

class TrackingMemoryResource final : public std::pmr::memory_resource {
public:
  explicit TrackingMemoryResource(
      std::pmr::memory_resource *upstream = std::pmr::get_default_resource())
      : upstream_(upstream) {}

  [[nodiscard]] uint64_t currentBytes() const noexcept {
    return currentBytes_;
  }
  [[nodiscard]] uint64_t peakBytes() const noexcept {
    return peakBytes_;
  }

private:
  void *do_allocate(size_t bytes, size_t alignment) override {
    void *ptr = upstream_->allocate(bytes, alignment);
    currentBytes_ += bytes;
    peakBytes_ = std::max(peakBytes_, currentBytes_);
    return ptr;
  }

  void do_deallocate(void *ptr, size_t bytes, size_t alignment) override {
    upstream_->deallocate(ptr, bytes, alignment);
    currentBytes_ -= bytes;
  }

  bool do_is_equal(const std::pmr::memory_resource &other) const noexcept override {
    return this == &other;
  }

  std::pmr::memory_resource *upstream_ = std::pmr::get_default_resource();
  uint64_t currentBytes_ = 0u;
  uint64_t peakBytes_ = 0u;
};

struct ProcessMemorySnapshot {
  bool available = false;
  uint64_t workingSetBytes = 0u;
  uint64_t peakWorkingSetBytes = 0u;
  uint64_t privateUsageBytes = 0u;
  uint64_t pagefileUsageBytes = 0u;
  uint64_t peakPagefileUsageBytes = 0u;
};

[[nodiscard]] ProcessMemorySnapshot collectProcessMemorySnapshot() {
  ProcessMemorySnapshot snapshot{};
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS_EX counters{};
  counters.cb = sizeof(counters);
  if (K32GetProcessMemoryInfo(
          GetCurrentProcess(),
          reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&counters),
          sizeof(counters)) == FALSE) {
    return snapshot;
  }
  snapshot.available = true;
  snapshot.workingSetBytes = static_cast<uint64_t>(counters.WorkingSetSize);
  snapshot.peakWorkingSetBytes =
      static_cast<uint64_t>(counters.PeakWorkingSetSize);
  snapshot.privateUsageBytes = static_cast<uint64_t>(counters.PrivateUsage);
  snapshot.pagefileUsageBytes = static_cast<uint64_t>(counters.PagefileUsage);
  snapshot.peakPagefileUsageBytes =
      static_cast<uint64_t>(counters.PeakPagefileUsage);
#else
  struct rusage usage {};
  if (getrusage(RUSAGE_SELF, &usage) != 0) {
    return snapshot;
  }
  snapshot.available = true;
#if defined(__APPLE__)
  snapshot.peakWorkingSetBytes = static_cast<uint64_t>(usage.ru_maxrss);
#else
  snapshot.peakWorkingSetBytes =
      static_cast<uint64_t>(usage.ru_maxrss) * 1024u;
#endif
#endif
  return snapshot;
}

void addIfNonzero(std::map<std::string, double> &measurements,
                  std::string_view id, uint64_t value) {
  if (value != 0u) {
    measurements.emplace(std::string(id), static_cast<double>(value));
  }
}

void addBytesAsMiB(std::map<std::string, double> &measurements,
                   std::string_view id, uint64_t bytes) {
  if (bytes != 0u) {
    measurements.emplace(std::string(id), bytesToMiB(bytes));
  }
}

void addProcessMemoryMetrics(std::map<std::string, double> &measurements) {
  const ProcessMemorySnapshot snapshot = collectProcessMemorySnapshot();
  if (!snapshot.available) {
    return;
  }
  addBytesAsMiB(measurements, "memory.process.working_set_mb",
                snapshot.workingSetBytes);
  addBytesAsMiB(measurements, "memory.process.peak_working_set_mb",
                snapshot.peakWorkingSetBytes);
  addBytesAsMiB(measurements, "memory.process.private_usage_mb",
                snapshot.privateUsageBytes);
  addBytesAsMiB(measurements, "memory.process.pagefile_usage_mb",
                snapshot.pagefileUsageBytes);
  addBytesAsMiB(measurements, "memory.process.peak_pagefile_usage_mb",
                snapshot.peakPagefileUsageBytes);
}

void addPmrMemoryMetrics(std::map<std::string, double> &measurements,
                         const TrackingMemoryResource &rendererMemory,
                         const TrackingMemoryResource &pipelineMemory,
                         const TrackingMemoryResource &sceneMemory) {
  addBytesAsMiB(measurements, "memory.pmr.renderer_current_mb",
                rendererMemory.currentBytes());
  addBytesAsMiB(measurements, "memory.pmr.renderer_peak_mb",
                rendererMemory.peakBytes());
  addBytesAsMiB(measurements, "memory.pmr.pipeline_current_mb",
                pipelineMemory.currentBytes());
  addBytesAsMiB(measurements, "memory.pmr.pipeline_peak_mb",
                pipelineMemory.peakBytes());
  addBytesAsMiB(measurements, "memory.pmr.scene_current_mb",
                sceneMemory.currentBytes());
  addBytesAsMiB(measurements, "memory.pmr.scene_peak_mb",
                sceneMemory.peakBytes());
}

void addRendererFrameMetrics(std::map<std::string, double> &measurements,
                             const RenderFrameMetrics &metrics) {
  const OpaqueFrameMetrics &opaque = metrics.opaque;
  addIfNonzero(measurements, "renderer.opaque.total_instances",
               opaque.totalInstances);
  addIfNonzero(measurements, "renderer.opaque.visible_instances",
               opaque.visibleInstances);
  addIfNonzero(measurements, "renderer.opaque.instanced_draws",
               opaque.instancedDraws);
  addIfNonzero(measurements, "renderer.opaque.indirect_draw_calls",
               opaque.indirectDrawCalls);
  addIfNonzero(measurements, "renderer.opaque.indirect_commands",
               opaque.indirectCommands);
  addIfNonzero(measurements, "renderer.opaque.compute_dispatches",
               opaque.computeDispatches);
  addIfNonzero(measurements, "renderer.opaque.depth_prepass_draws",
               opaque.depthPrepassDraws);
  addIfNonzero(measurements, "renderer.opaque.tessellated_draws",
               opaque.tessellatedDraws);

  const ShadowFrameMetrics &shadow = metrics.shadow;
  addIfNonzero(measurements, "renderer.shadow.cascades", shadow.cascadeCount);
  addIfNonzero(measurements, "renderer.shadow.total_draws",
               shadow.totalDraws);
  addIfNonzero(measurements, "renderer.shadow.total_culled_draws",
               shadow.totalCulledDraws);
  addIfNonzero(measurements, "renderer.shadow.static_caster_entries",
               shadow.staticCasterEntries);
  addIfNonzero(measurements, "renderer.shadow.dynamic_caster_entries",
               shadow.dynamicCasterEntries);
  addIfNonzero(measurements, "renderer.shadow.filter_sample_budget",
               shadow.filterSampleBudget);
  addIfNonzero(measurements, "renderer.shadow.sdsm_compute_passes",
               shadow.sdsmComputePassCount);
  addBytesAsMiB(measurements, "gpu.memory.shadow.cascade_texture_mb",
                shadow.cascadeTextureBytes);

  const AntiAliasingFrameMetrics &aa = metrics.antiAliasing;
  addIfNonzero(measurements, "renderer.aa.motion_vector_textures",
               aa.motionVectorTextureCount);
  addIfNonzero(measurements, "renderer.aa.motion_vector_allocations",
               aa.motionVectorAllocationCount);
  addIfNonzero(measurements, "renderer.aa.motion_vector_reallocations",
               aa.motionVectorReallocationCount);
  addIfNonzero(measurements, "renderer.aa.velocity_passes",
               aa.velocityPassCount);
  addIfNonzero(measurements, "renderer.aa.velocity_draws",
               aa.velocityDrawCount);
  addIfNonzero(measurements, "renderer.aa.velocity_instances",
               aa.velocityInstanceCount);
  addIfNonzero(measurements, "renderer.aa.reactive_mask_passes",
               aa.reactiveMaskPassCount);
  addIfNonzero(measurements, "renderer.aa.reactive_mask_draws",
               aa.reactiveMaskDrawCount);
  addIfNonzero(measurements, "renderer.aa.taa_resolve_passes",
               aa.taaResolvePassCount);
  addIfNonzero(measurements, "renderer.aa.taa_copy_back_passes",
               aa.taaCopyBackPassCount);
  addIfNonzero(measurements, "renderer.aa.spatial_aa_passes",
               aa.spatialAAPassCount);
  addIfNonzero(measurements, "renderer.aa.msaa_resolve_passes",
               aa.msaaResolvePassCount);
  addBytesAsMiB(measurements, "gpu.memory.aa.motion_vector_total_mb",
                aa.motionVectorTotalBytes);
  addBytesAsMiB(measurements, "gpu.memory.aa.reactive_mask_total_mb",
                aa.reactiveMaskTotalBytes);
  addBytesAsMiB(measurements, "gpu.memory.aa.spatial_aa_total_mb",
                aa.spatialAATotalBytes);
  addBytesAsMiB(measurements, "gpu.memory.aa.msaa_total_mb",
                aa.msaaTotalBytes);

  const AmbientOcclusionFrameMetrics &ao = metrics.ambientOcclusion;
  addIfNonzero(measurements, "renderer.ao.normal_prepass_draws",
               ao.normalPrepassDraws);
  addIfNonzero(measurements, "renderer.ao.depth_prefilter_passes",
               ao.depthPrefilterPassCount);
  addIfNonzero(measurements, "renderer.ao.main_passes", ao.mainPassCount);
  addIfNonzero(measurements, "renderer.ao.temporal_passes",
               ao.temporalPassCount);
  addIfNonzero(measurements, "renderer.ao.texture_count", ao.textureCount);
  addBytesAsMiB(measurements, "gpu.memory.ao.total_texture_mb",
                ao.totalTextureBytes);

  const HDRPostProcessFrameMetrics &hdr = metrics.hdrPostProcess;
  addIfNonzero(measurements, "renderer.hdr.bloom_passes",
               hdr.bloomPassCount);
  addIfNonzero(measurements, "renderer.hdr.luminance_passes",
               hdr.luminancePassCount);
  addIfNonzero(measurements, "renderer.hdr.adaptation_passes",
               hdr.adaptationPassCount);
  addIfNonzero(measurements, "renderer.hdr.texture_count", hdr.textureCount);
  addBytesAsMiB(measurements, "gpu.memory.hdr.texture_mb", hdr.textureBytes);

  addIfNonzero(measurements, "renderer.transparent.mesh_draws",
               metrics.transparent.meshDraws);
  addIfNonzero(measurements, "renderer.transparent.contributor_sortable_draws",
               metrics.transparent.contributorSortableDraws);
  addIfNonzero(measurements, "renderer.transparent.contributor_fixed_draws",
               metrics.transparent.contributorFixedDraws);
  addIfNonzero(measurements, "renderer.transparent.pick_draws",
               metrics.transparent.pickDraws);

  const uint64_t estimatedFrameTextureBytes =
      shadow.cascadeTextureBytes + aa.motionVectorTotalBytes +
      aa.reactiveMaskTotalBytes + aa.spatialAATotalBytes + aa.msaaTotalBytes +
      ao.totalTextureBytes + hdr.textureBytes;
  addBytesAsMiB(measurements, "gpu.memory.frame_textures_estimated_mb",
                estimatedFrameTextureBytes);
}

class ScopedEnvVar final {
public:
  ScopedEnvVar(std::string name, std::string value)
      : name_(std::move(name)), oldValue_(readProcessEnvironment(name_)),
        hadOldValue_(!oldValue_.empty()) {
    set(value);
  }
  ~ScopedEnvVar() {
    if (hadOldValue_) {
      set(oldValue_);
    } else {
      unset();
    }
  }
  ScopedEnvVar(const ScopedEnvVar &) = delete;
  ScopedEnvVar &operator=(const ScopedEnvVar &) = delete;

private:
  void set(const std::string &value) {
#if defined(_WIN32)
    _putenv_s(name_.c_str(), value.c_str());
#else
    setenv(name_.c_str(), value.c_str(), 1);
#endif
  }
  void unset() {
#if defined(_WIN32)
    _putenv_s(name_.c_str(), "");
#else
    unsetenv(name_.c_str());
#endif
  }
  std::string name_;
  std::string oldValue_;
  bool hadOldValue_ = false;
};

class BenchmarkLogGuard final {
public:
  BenchmarkLogGuard() {
    std::filesystem::create_directories("logs");
    LogConfig config{};
    config.filePath = (std::filesystem::path("logs") /
                       (utcTimestampForPath() + "_nuri_bench.log"))
                          .string();
    config.logLevel = LogLevel::Info;
    config.consoleLevel = LogLevel::Warning;
    config.threadNames = false;
    Log::initialize(config);
  }
  ~BenchmarkLogGuard() { Log::shutdown(); }
  BenchmarkLogGuard(const BenchmarkLogGuard &) = delete;
  BenchmarkLogGuard &operator=(const BenchmarkLogGuard &) = delete;
};

[[nodiscard]] GPUBackendPreference backendPreference(std::string_view backend) {
  if (backend == "lvk") {
    return GPUBackendPreference::Lvk;
  }
  if (backend == "nvrhi") {
    return GPUBackendPreference::Nvrhi;
  }
  return GPUBackendPreference::Default;
}

[[nodiscard]] std::string resolveBackendName(const BenchmarkCase &benchmarkCase,
                                             std::string &source) {
  const std::string envBackend = readProcessEnvironment("NURI_GPU_BACKEND");
  if (benchmarkCase.backend != "default") {
    source = "manifest";
    return benchmarkCase.backend;
  }
  if (!envBackend.empty()) {
    source = "NURI_GPU_BACKEND";
    return envBackend;
  }
  source = "default";
  return "nvrhi";
}

[[nodiscard]] std::string
resolvePresentMode(const BenchmarkCase &benchmarkCase, std::string &source) {
  const std::string envPresent = readProcessEnvironment("NURI_PRESENT_MODE");
  if (benchmarkCase.presentMode != "default") {
    source = "manifest";
    return benchmarkCase.presentMode;
  }
  if (!envPresent.empty()) {
    source = "NURI_PRESENT_MODE";
    return envPresent;
  }
  source = "default";
  return "default";
}

[[nodiscard]] Result<bool, BenchmarkExitCode>
checkRequirements(const BenchmarkCase &benchmarkCase,
                  std::string_view backend,
                  std::vector<std::string> &warnings,
                  std::string &message) {
  if (!benchmarkCase.requirements.allowVisibleWindow) {
    message = "case requires hidden/headless execution, which is unavailable";
    return Result<bool, BenchmarkExitCode>::makeError(
        BenchmarkExitCode::EnvironmentUnavailable);
  }
  if (!benchmarkCase.requirements.backends.empty()) {
    bool supported = false;
    for (const std::string &allowed : benchmarkCase.requirements.backends) {
      supported = supported || allowed == backend || allowed == "default";
    }
    if (!supported) {
      message = "backend '" + std::string(backend) +
                "' is not allowed by case requirements";
      return Result<bool, BenchmarkExitCode>::makeError(
          BenchmarkExitCode::EnvironmentUnavailable);
    }
  }
  for (const std::string &asset : benchmarkCase.requirements.assets) {
    const size_t colon = asset.find(':');
    if (colon == std::string::npos) {
      message = "invalid asset requirement '" + asset + "'";
      return Result<bool, BenchmarkExitCode>::makeError(
          BenchmarkExitCode::InvalidInput);
    }
    auto path = resolveBenchmarkPath(asset.substr(0, colon),
                                     asset.substr(colon + 1u));
    if (path.hasError()) {
      message = path.error();
      return Result<bool, BenchmarkExitCode>::makeError(
          BenchmarkExitCode::EnvironmentUnavailable);
    }
    if (!std::filesystem::exists(path.value())) {
      message = "missing required asset: " + path.value().string();
      return Result<bool, BenchmarkExitCode>::makeError(
          BenchmarkExitCode::EnvironmentUnavailable);
    }
  }
  if (benchmarkCase.scene.kind == "prefab" &&
      benchmarkCase.scene.baseModelKind == "fitRadius") {
    warnings.push_back(
        "prefab fitRadius transforms are parsed but not enabled for benchmark "
        "comparison in this slice");
  }
  return Result<bool, BenchmarkExitCode>::makeResult(true);
}

void addGpuTimingMetric(std::map<std::string, double> &measurements,
                        std::string_view metricId, const GpuTimingReport &report,
                        GpuTimingScope scope, float timeMs) {
  if (hasGpuTimingScope(report, scope)) {
    measurements.emplace(std::string(metricId), static_cast<double>(timeMs));
  }
}

[[nodiscard]] uint64_t reportSourceFrameIndex(const GpuTimingReport &report) {
  static constexpr uint64_t kInvalid = std::numeric_limits<uint64_t>::max();
  const std::array<uint64_t, 12> sources{
      report.shadowSourceFrameIndex,
      report.shadowDepthSourceFrameIndex,
      report.shadowSdsmSourceFrameIndex,
      report.sceneColorDownsampleSourceFrameIndex,
      report.transmissionSourceFrameIndex,
      report.temporalAAResolveSourceFrameIndex,
      report.temporalAADebugSourceFrameIndex,
      report.spatialAASourceFrameIndex,
      report.opaqueSourceFrameIndex,
      report.msaaResolveSourceFrameIndex,
      report.gtaoSourceFrameIndex,
      report.hdrPostProcessSourceFrameIndex,
  };
  for (const uint64_t source : sources) {
    if (source != kInvalid) {
      return source;
    }
  }
  return kInvalid;
}

void applyGpuTimingReport(BenchmarkReport &report,
                          const GpuTimingReport &timingReport,
                          const std::map<uint64_t, size_t> &frameByIndex) {
  const uint64_t frameIndex = reportSourceFrameIndex(timingReport);
  const auto frameIt = frameByIndex.find(frameIndex);
  if (frameIt == frameByIndex.end()) {
    return;
  }
  BenchmarkFrameRecord &frame = report.frames[frameIt->second];
  double sum = 0.0;
  const auto add = [&](std::string_view id, GpuTimingScope scope, float ms) {
    if (hasGpuTimingScope(timingReport, scope)) {
      sum += static_cast<double>(ms);
      frame.measurements.emplace(std::string(id), static_cast<double>(ms));
    }
  };
  add("gpu.scopes.shadow_ms", GpuTimingScope::Shadow,
      timingReport.shadowTimeMs);
  add("gpu.scopes.shadow_depth_ms", GpuTimingScope::ShadowDepth,
      timingReport.shadowDepthTimeMs);
  add("gpu.scopes.shadow_sdsm_ms", GpuTimingScope::ShadowSdsm,
      timingReport.shadowSdsmTimeMs);
  add("gpu.scopes.opaque_ms", GpuTimingScope::Opaque,
      timingReport.opaqueTimeMs);
  add("gpu.scopes.gtao_ms", GpuTimingScope::GTAO, timingReport.gtaoTimeMs);
  add("gpu.scopes.msaa_resolve_ms", GpuTimingScope::MsaaResolve,
      timingReport.msaaResolveTimeMs);
  add("gpu.scopes.scene_color_downsample_ms",
      GpuTimingScope::SceneColorDownsample,
      timingReport.sceneColorDownsampleTimeMs);
  add("gpu.scopes.taa_resolve_ms", GpuTimingScope::TemporalAAResolve,
      timingReport.temporalAAResolveTimeMs);
  add("gpu.scopes.taa_debug_ms", GpuTimingScope::TemporalAADebug,
      timingReport.temporalAADebugTimeMs);
  add("gpu.scopes.spatial_aa_ms", GpuTimingScope::SpatialAA,
      timingReport.spatialAATimeMs);
  add("gpu.scopes.transmission_ms", GpuTimingScope::Transmission,
      timingReport.transmissionTimeMs);
  add("gpu.scopes.hdr_postprocess_ms", GpuTimingScope::HDRPostProcess,
      timingReport.hdrPostProcessTimeMs);
  if (sum > 0.0) {
    frame.measurements["gpu.scopes_sum_ms"] = sum;
  }
}

void drainGpuTimings(GPUDevice &gpu, BenchmarkReport &report,
                     const std::map<uint64_t, size_t> &frameByIndex) {
  std::array<GpuTimingReport, 32> reports{};
  size_t drained = 0u;
  do {
    drained = gpu.drainCompletedGpuTimingReports(reports);
    for (size_t i = 0u; i < drained; ++i) {
      applyGpuTimingReport(report, reports[i], frameByIndex);
    }
  } while (drained == reports.size());
}

[[nodiscard]] Result<bool, std::string>
populateScene(const BenchmarkCase &benchmarkCase, Renderer &renderer,
              RenderScene &scene,
              std::pmr::memory_resource *memory,
              std::optional<ScenePrefab> &prefab,
              std::optional<ScenePrefabAssets> &prefabAssets) {
  scene.bindResources(&renderer.resources());
  auto lightResult =
      scene.graph().addLight(scene.graph().rootNode(),
                             LightDesc{
                                 .type = LightType::Directional,
                                 .name = "benchmark_key",
                                 .color = glm::vec3(1.0f),
                                 .intensity = 4.0f,
                                 .enabled = true,
                             });
  if (lightResult.hasError()) {
    return Result<bool, std::string>::makeError(lightResult.error());
  }

  if (benchmarkCase.scene.kind == "prefab") {
    if (benchmarkCase.scene.pathBase.empty() ||
        benchmarkCase.scene.path.empty()) {
      return Result<bool, std::string>::makeError(
          "prefab scene requires pathBase and path");
    }
    auto path =
        resolveBenchmarkPath(benchmarkCase.scene.pathBase,
                             benchmarkCase.scene.path);
    if (path.hasError()) {
      return Result<bool, std::string>::makeError(path.error());
    }
    if (!std::filesystem::exists(path.value())) {
      return Result<bool, std::string>::makeError(
          "missing scene asset: " + path.value().string());
    }
    SceneImportOptions importOptions{};
    importOptions.assetBuildOptions.flipUVs = benchmarkCase.scene.flipUVs;
    auto prefabResult = SceneImporter::loadScenePrefabFromFile(
        path.value().string(), importOptions, memory);
    if (prefabResult.hasError()) {
      return Result<bool, std::string>::makeError(prefabResult.error());
    }
    prefab.emplace(std::move(prefabResult.value()));
    auto assetsResult = renderer.resources().acquireScenePrefabAssets(*prefab);
    if (assetsResult.hasError()) {
      return Result<bool, std::string>::makeError(assetsResult.error());
    }
    prefabAssets.emplace(std::move(assetsResult.value()));
    auto instantiateResult = scene.graph().instantiatePrefab(
        *prefab, scene.graph().rootNode(), *prefabAssets);
    if (instantiateResult.hasError()) {
      return Result<bool, std::string>::makeError(instantiateResult.error());
    }
  }

  auto syncResult = scene.graph().syncWorldTransforms();
  (void)syncResult;
  auto commitResult = scene.commit();
  if (commitResult.hasError()) {
    return Result<bool, std::string>::makeError(commitResult.error());
  }
  return Result<bool, std::string>::makeResult(true);
}

[[nodiscard]] Camera makeBenchmarkCamera(const BenchmarkCase &benchmarkCase) {
  Camera camera;
  camera.setPerspective(PerspectiveParams{
      .fovYRadians = glm::radians(benchmarkCase.camera.verticalFovDegrees),
      .nearPlane = benchmarkCase.camera.nearPlane,
      .farPlane = benchmarkCase.camera.farPlane,
  });
  const glm::vec3 direction =
      glm::length(benchmarkCase.camera.direction) > 1.0e-6f
          ? glm::normalize(benchmarkCase.camera.direction)
          : glm::vec3(0.0f, 0.0f, -1.0f);
  camera.setLookAt(benchmarkCase.camera.position,
                   benchmarkCase.camera.position + direction,
                   glm::vec3(0.0f, 1.0f, 0.0f));
  return camera;
}

void buildFrameContext(RenderFrameContext &frameContext, RenderScene &scene,
                       Renderer &renderer, RenderSettings &settings,
                       TemporalCameraHistoryState &cameraHistory,
                       const Camera &camera, uint64_t frameIndex,
                       double timeSeconds, double deltaSeconds,
                       uint32_t width, uint32_t height) {
  sanitizeBenchmarkRenderSettings(settings);
  frameContext.scene = &scene;
  frameContext.resources = &renderer.resources();
  frameContext.frameIndex = frameIndex;
  const MaterialTableSnapshot materialSnapshot =
      renderer.resources().materialSnapshot();
  const TemporalSceneContentState sceneContent{
      .lightTopologyVersion = scene.lightTopologyVersion(),
      .lightTransformVersion = scene.lightTransformVersion(),
      .materialTableVersion = materialSnapshot.version,
      .environmentVersion = scene.environmentVersion(),
  };
  frameContext.camera = makeTemporalCameraFrameState(
      camera, static_cast<float>(width) / static_cast<float>(height),
      settings.antiAliasing,
      TemporalCameraFrameDesc{
          .renderExtent = glm::uvec2(width, height),
          .sceneContent = sceneContent,
      },
      cameraHistory);
  settings.antiAliasing.debug.resetHistoryRequested = false;
  frameContext.settings = &settings;
  frameContext.metrics = {};
  frameContext.metrics.frameIndex = frameContext.frameIndex;
  frameContext.metrics.antiAliasing =
      makeAntiAliasingFrameMetrics(frameContext.camera);
  frameContext.sharedDepthTexture = {};
  frameContext.timeSeconds = timeSeconds;
  frameContext.deltaSeconds = deltaSeconds;
}

[[nodiscard]] double elapsedMs(
    std::chrono::steady_clock::time_point begin,
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now()) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

} // namespace

Result<std::string, std::string>
formatBenchmarkCaseListJson(const std::vector<BenchmarkCase> &cases,
                            std::string_view suite) {
  BenchmarkReport dummy{};
  (void)dummy;
  std::ostringstream out;
  out << "{\n  \"cases\": [\n";
  bool first = true;
  for (const BenchmarkCase &benchmarkCase : cases) {
    if (!suite.empty() && benchmarkCase.suite != suite) {
      continue;
    }
    if (!first) {
      out << ",\n";
    }
    first = false;
    out << "    {\"id\": \"" << benchmarkCase.id << "\", \"suite\": \""
        << benchmarkCase.suite << "\", \"description\": \""
        << benchmarkCase.description << "\"}";
  }
  out << "\n  ]\n}\n";
  return Result<std::string, std::string>::makeResult(out.str());
}

std::string formatBenchmarkCaseListText(const std::vector<BenchmarkCase> &cases,
                                        std::string_view suite) {
  std::ostringstream out;
  for (const BenchmarkCase &benchmarkCase : cases) {
    if (!suite.empty() && benchmarkCase.suite != suite) {
      continue;
    }
    out << benchmarkCase.id << " [" << benchmarkCase.suite << "] "
        << benchmarkCase.description << "\n";
  }
  return out.str();
}

Result<std::string, std::string>
formatBenchmarkCaseExplanationJson(const BenchmarkCase &benchmarkCase) {
  std::ostringstream out;
  out << "{\n"
      << "  \"id\": \"" << benchmarkCase.id << "\",\n"
      << "  \"suite\": \"" << benchmarkCase.suite << "\",\n"
      << "  \"description\": \"" << benchmarkCase.description << "\",\n"
      << "  \"sceneKind\": \"" << benchmarkCase.scene.kind << "\",\n"
      << "  \"backend\": \"" << benchmarkCase.backend << "\",\n"
      << "  \"samples\": " << benchmarkCase.samples << ",\n"
      << "  \"measurementFrames\": " << benchmarkCase.measurementFrames << "\n"
      << "}\n";
  return Result<std::string, std::string>::makeResult(out.str());
}

std::string formatBenchmarkCaseExplanationText(
    const BenchmarkCase &benchmarkCase) {
  std::ostringstream out;
  out << benchmarkCase.id << "\n"
      << "suite: " << benchmarkCase.suite << "\n"
      << "description: " << benchmarkCase.description << "\n"
      << "scene: " << benchmarkCase.scene.kind << "\n"
      << "backend: " << benchmarkCase.backend << "\n"
      << "resolution: " << benchmarkCase.resolution[0] << "x"
      << benchmarkCase.resolution[1] << "\n"
      << "frames: warmup=" << benchmarkCase.warmupFrames
      << " measured=" << benchmarkCase.measurementFrames << "\n";
  return out.str();
}

Result<std::string, std::string>
formatEffectiveConfigJson(const BenchmarkCase &benchmarkCase,
                          const BenchmarkRunOptions &options) {
  std::string backendSource;
  const std::string backend = resolveBackendName(benchmarkCase, backendSource);
  std::string presentSource;
  const std::string present = resolvePresentMode(benchmarkCase, presentSource);
  std::ostringstream out;
  out << "{\n"
      << "  \"case\": \"" << benchmarkCase.id << "\",\n"
      << "  \"backend\": \"" << backend << "\",\n"
      << "  \"backendSource\": \"" << backendSource << "\",\n"
      << "  \"presentMode\": \"" << present << "\",\n"
      << "  \"presentModeSource\": \"" << presentSource << "\",\n"
      << "  \"samples\": "
      << options.samplesOverride.value_or(benchmarkCase.samples) << ",\n"
      << "  \"artifactDir\": \"" << options.artifactDir.generic_string()
      << "\"\n"
      << "}\n";
  return Result<std::string, std::string>::makeResult(out.str());
}

BenchmarkRunResult runBenchmarkCase(BenchmarkCase benchmarkCase,
                                    const BenchmarkRunOptions &options) {
  BenchmarkRunResult result{};
  const uint32_t samples = options.samplesOverride.value_or(benchmarkCase.samples);
  benchmarkCase.samples = samples;
  std::string backendSource;
  const std::string backend = resolveBackendName(benchmarkCase, backendSource);
  std::string presentSource;
  const std::string presentMode =
      resolvePresentMode(benchmarkCase, presentSource);
  const std::filesystem::path artifactDir =
      options.artifactDir.empty()
          ? benchmarkRepoRoot() / "artifacts" / "bench" / utcTimestampForPath()
          : options.artifactDir;
  const std::filesystem::path reportPath =
      options.jsonOut.empty()
          ? artifactDir / "cases" / (benchmarkCase.id + ".json")
          : options.jsonOut;
  result.reportPath = reportPath;

  BenchmarkReport report{};
  report.generatedAtUtc = utcTimestampIso8601();
  report.command = options.command;
  report.benchmarkCase = benchmarkCase;
  report.run.samples = samples;
  report.run.warmupFrames = benchmarkCase.warmupFrames;
  report.run.measurementFrames = benchmarkCase.measurementFrames;
  report.run.cooldownFrames = benchmarkCase.cooldownFrames;
  report.run.maxDrainFrames = benchmarkCase.maxDrainFrames;
  report.run.drainTimeoutMs = benchmarkCase.drainTimeoutMs;
  report.run.fixedDeltaSeconds = benchmarkCase.fixedDeltaSeconds;
  report.artifacts.artifactDir = artifactDir;
  report.timingDrain.drainTimeoutMs = benchmarkCase.drainTimeoutMs;
  report.environment = collectBenchmarkEnvironment(
      backend, backendSource, presentMode, presentSource,
      options.tracyDiagnostic);
  report.environment.renderGraphWorkerCount =
      benchmarkCase.renderGraph.workerCount;
  report.environment.renderGraphParallelCompile =
      benchmarkCase.renderGraph.parallelCompile;
  report.environment.renderGraphParallelRecording =
      benchmarkCase.renderGraph.parallelRecording;
  report.warnings.push_back(
      "first-slice benchmark uses the swapchain-present renderer path");
  if (options.tracyDiagnostic) {
    report.run.validForComparison = false;
    report.warnings.push_back(
        "Tracy diagnostic mode is not valid for authoritative comparison");
  }

  std::string requirementMessage;
  auto requirements = checkRequirements(benchmarkCase, backend, report.warnings,
                                        requirementMessage);
  if (requirements.hasError()) {
    report.run.validForComparison = false;
    report.warnings.push_back(requirementMessage);
    result.exitCode = requirements.error();
    result.message = requirementMessage;
    computeBenchmarkReportStats(report);
    (void)writeBenchmarkReportFile(report, reportPath, options.verboseFrames);
    result.report = std::move(report);
    return result;
  }

  if (options.dryRun) {
    result.exitCode = BenchmarkExitCode::Success;
    result.message = "dry run succeeded";
    report.warnings.push_back("dry run: renderer was not initialized");
    report.run.validForComparison = false;
    computeBenchmarkReportStats(report);
    (void)writeBenchmarkReportFile(report, reportPath, options.verboseFrames);
    result.report = std::move(report);
    return result;
  }

  try {
    BenchmarkLogGuard logGuard;
    std::vector<std::unique_ptr<ScopedEnvVar>> env;
    env.push_back(std::make_unique<ScopedEnvVar>(
        "NURI_RENDER_GRAPH_WORKER_COUNT",
        std::to_string(benchmarkCase.renderGraph.workerCount)));
    env.push_back(std::make_unique<ScopedEnvVar>(
        "NURI_RENDER_GRAPH_DISABLE_PARALLEL_COMPILE",
        benchmarkCase.renderGraph.parallelCompile ? "0" : "1"));
    env.push_back(std::make_unique<ScopedEnvVar>(
        "NURI_RENDER_GRAPH_DISABLE_PARALLEL_RECORDING",
        benchmarkCase.renderGraph.parallelRecording ? "0" : "1"));
    if (presentSource == "manifest" && presentMode != "default") {
      env.push_back(
          std::make_unique<ScopedEnvVar>("NURI_PRESENT_MODE", presentMode));
    }

    auto configResult = loadRuntimeConfigFromEnvOrDefault();
    if (configResult.hasError()) {
      result.exitCode = BenchmarkExitCode::EnvironmentUnavailable;
      result.message = configResult.error();
      report.run.validForComparison = false;
      report.warnings.push_back(configResult.error());
      computeBenchmarkReportStats(report);
      (void)writeBenchmarkReportFile(report, reportPath, options.verboseFrames);
      result.report = std::move(report);
      return result;
    }
    RuntimeConfig config = std::move(configResult.value());
    config.window.title = "nuri-bench " + benchmarkCase.id;
    config.window.width = static_cast<int32_t>(benchmarkCase.resolution[0]);
    config.window.height = static_cast<int32_t>(benchmarkCase.resolution[1]);
    config.window.mode = WindowMode::Windowed;

    std::unique_ptr<Window> window =
        Window::create(config.window.title, config.window.width,
                       config.window.height, config.window.mode);
    if (!window) {
      result.exitCode = BenchmarkExitCode::EnvironmentUnavailable;
      result.message = "failed to create benchmark window";
      report.run.validForComparison = false;
      report.warnings.push_back(result.message);
      computeBenchmarkReportStats(report);
      (void)writeBenchmarkReportFile(report, reportPath, options.verboseFrames);
      result.report = std::move(report);
      return result;
    }
    GPUDeviceCreateDesc deviceDesc{};
    deviceDesc.backend = backendPreference(backend);
    std::unique_ptr<GPUDevice> gpu = GPUDevice::create(*window, deviceDesc);
    if (!gpu) {
      result.exitCode = BenchmarkExitCode::EnvironmentUnavailable;
      result.message = "failed to create GPU device";
      report.run.validForComparison = false;
      report.warnings.push_back(result.message);
      computeBenchmarkReportStats(report);
      (void)writeBenchmarkReportFile(report, reportPath, options.verboseFrames);
      result.report = std::move(report);
      return result;
    }
    report.environment.swapchainImageCount = gpu->getSwapchainImageCount();

    TrackingMemoryResource rendererMemoryTracker;
    TrackingMemoryResource pipelineMemoryTracker;
    TrackingMemoryResource sceneMemoryTracker;
    std::pmr::unsynchronized_pool_resource rendererMemory(
        &rendererMemoryTracker);
    std::pmr::unsynchronized_pool_resource pipelineMemory(
        &pipelineMemoryTracker);
    std::pmr::unsynchronized_pool_resource sceneMemory(&sceneMemoryTracker);
    std::unique_ptr<Renderer> renderer =
        Renderer::create(*gpu, rendererMemory);
    RenderPipeline pipeline(&pipelineMemory);
    auto pipelineResult = registerDefaultRenderPipeline(
        pipeline, *gpu, config.shaders, &pipelineMemory);
    if (pipelineResult.hasError()) {
      result.exitCode = BenchmarkExitCode::RuntimeError;
      result.message = pipelineResult.error();
      report.run.validForComparison = false;
      report.warnings.push_back(result.message);
      computeBenchmarkReportStats(report);
      (void)writeBenchmarkReportFile(report, reportPath, options.verboseFrames);
      result.report = std::move(report);
      return result;
    }

    RenderScene scene(&sceneMemory);
    std::optional<ScenePrefab> prefab;
    std::optional<ScenePrefabAssets> prefabAssets;
    auto sceneResult = populateScene(benchmarkCase, *renderer, scene,
                                     &sceneMemory, prefab, prefabAssets);
    if (sceneResult.hasError()) {
      result.exitCode = BenchmarkExitCode::EnvironmentUnavailable;
      result.message = sceneResult.error();
      report.run.validForComparison = false;
      report.warnings.push_back(result.message);
      computeBenchmarkReportStats(report);
      (void)writeBenchmarkReportFile(report, reportPath, options.verboseFrames);
      result.report = std::move(report);
      return result;
    }

    RenderSettings settings = benchmarkCase.settings;
    Camera camera = makeBenchmarkCamera(benchmarkCase);
    TemporalCameraHistoryState cameraHistory{};
    RenderFrameContext frameContext{};
    uint64_t frameIndex = 0u;
    double timeSeconds = 0.0;
    std::map<uint64_t, size_t> measuredFrameByIndex;

    const auto renderOneFrame = [&](uint32_t sampleIndex,
                                    bool measured) -> Result<bool, std::string> {
      window->pollEvents();
      BenchmarkFrameRecord frame{};
      frame.frameIndex = frameIndex;
      frame.sampleIndex = sampleIndex;
      frame.measured = measured;
      const auto totalBegin = std::chrono::steady_clock::now();
      const auto tickBegin = std::chrono::steady_clock::now();
      const double sceneTickMs = elapsedMs(tickBegin);
      const auto commitBegin = std::chrono::steady_clock::now();
      auto commitResult = scene.commit();
      if (commitResult.hasError()) {
        return Result<bool, std::string>::makeError(commitResult.error());
      }
      const double sceneCommitMs = elapsedMs(commitBegin);
      buildFrameContext(frameContext, scene, *renderer, settings, cameraHistory,
                        camera, frameIndex, timeSeconds,
                        benchmarkCase.fixedDeltaSeconds,
                        benchmarkCase.resolution[0],
                        benchmarkCase.resolution[1]);
      const auto renderBegin = std::chrono::steady_clock::now();
      auto renderResult = renderer->render(pipeline, frameContext);
      const double renderSubmitMs = elapsedMs(renderBegin);
      if (renderResult.hasError()) {
        return Result<bool, std::string>::makeError(renderResult.error());
      }
      const double totalMs = elapsedMs(totalBegin);
      frame.metrics = frameContext.metrics;
      if (measured) {
        frame.measurements["cpu.total_ms"] = totalMs;
        frame.measurements["cpu.scene_tick_ms"] = sceneTickMs;
        frame.measurements["cpu.scene_commit_ms"] = sceneCommitMs;
        frame.measurements["cpu.render_submit_ms"] = renderSubmitMs;
        addRendererFrameMetrics(frame.measurements, frame.metrics);
        addProcessMemoryMetrics(frame.measurements);
        addPmrMemoryMetrics(frame.measurements, rendererMemoryTracker,
                            pipelineMemoryTracker, sceneMemoryTracker);
        measuredFrameByIndex.emplace(frame.frameIndex, report.frames.size());
      }
      report.frames.push_back(std::move(frame));
      ++frameIndex;
      timeSeconds += benchmarkCase.fixedDeltaSeconds;
      drainGpuTimings(*gpu, report, measuredFrameByIndex);
      return Result<bool, std::string>::makeResult(true);
    };

    for (uint32_t sampleIndex = 0u; sampleIndex < samples; ++sampleIndex) {
      for (uint32_t i = 0u; i < benchmarkCase.warmupFrames; ++i) {
        auto frameResult = renderOneFrame(sampleIndex, false);
        if (frameResult.hasError()) {
          result.exitCode = BenchmarkExitCode::RuntimeError;
          result.message = frameResult.error();
          report.warnings.push_back(result.message);
          report.run.validForComparison = false;
          computeBenchmarkReportStats(report);
          (void)writeBenchmarkReportFile(report, reportPath,
                                         options.verboseFrames);
          result.report = std::move(report);
          return result;
        }
      }
      for (uint32_t i = 0u; i < benchmarkCase.measurementFrames; ++i) {
        auto frameResult = renderOneFrame(sampleIndex, true);
        if (frameResult.hasError()) {
          result.exitCode = BenchmarkExitCode::RuntimeError;
          result.message = frameResult.error();
          report.warnings.push_back(result.message);
          report.run.validForComparison = false;
          computeBenchmarkReportStats(report);
          (void)writeBenchmarkReportFile(report, reportPath,
                                         options.verboseFrames);
          result.report = std::move(report);
          return result;
        }
      }
      for (uint32_t i = 0u; i < benchmarkCase.cooldownFrames; ++i) {
        auto frameResult = renderOneFrame(sampleIndex, false);
        if (frameResult.hasError()) {
          break;
        }
      }
    }

    const auto drainBegin = std::chrono::steady_clock::now();
    for (uint32_t drainFrame = 0u; drainFrame < benchmarkCase.maxDrainFrames;
         ++drainFrame) {
      drainGpuTimings(*gpu, report, measuredFrameByIndex);
      report.timingDrain.drainFrames = drainFrame;
      if (elapsedMs(drainBegin) >
          static_cast<double>(benchmarkCase.drainTimeoutMs)) {
        report.timingDrain.drainComplete = false;
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    report.timingDrain.droppedGpuTimingReports =
        gpu->droppedGpuTimingReportCount();
    gpu->waitIdle();
    drainGpuTimings(*gpu, report, measuredFrameByIndex);
  } catch (const std::exception &ex) {
    result.exitCode = BenchmarkExitCode::RuntimeError;
    result.message = ex.what();
    report.run.validForComparison = false;
    report.warnings.push_back(result.message);
  }

  computeBenchmarkReportStats(report);
  if (report.stats.find("gpu.scopes_sum_ms") == report.stats.end()) {
    report.unavailableMetrics.push_back("gpu.scopes_sum_ms");
    if (std::find(benchmarkCase.requiredMetrics.begin(),
                  benchmarkCase.requiredMetrics.end(),
                  "gpu.scopes_sum_ms") != benchmarkCase.requiredMetrics.end()) {
      report.run.validForComparison = false;
    }
  }
  auto writeResult =
      writeBenchmarkReportFile(report, reportPath, options.verboseFrames);
  if (writeResult.hasError() && result.exitCode == BenchmarkExitCode::Success) {
    result.exitCode = BenchmarkExitCode::RuntimeError;
    result.message = writeResult.error();
  }
  if (result.exitCode == BenchmarkExitCode::Success) {
    result.message = "benchmark run complete";
  }
  result.report = std::move(report);
  return result;
}

BenchmarkSuiteRunResult
runBenchmarkSuite(std::vector<BenchmarkCase> benchmarkCases,
                  std::string_view suite, const BenchmarkRunOptions &options) {
  BenchmarkSuiteRunResult suiteResult{};
  for (BenchmarkCase &benchmarkCase : benchmarkCases) {
    if (benchmarkCase.suite != suite) {
      continue;
    }
    BenchmarkRunOptions caseOptions = options;
    caseOptions.jsonOut.clear();
    auto result = runBenchmarkCase(std::move(benchmarkCase), caseOptions);
    if (static_cast<int>(result.exitCode) >
        static_cast<int>(suiteResult.exitCode)) {
      suiteResult.exitCode = result.exitCode;
    }
    suiteResult.caseResults.push_back(std::move(result));
  }
  suiteResult.message = "suite run complete";
  return suiteResult;
}

} // namespace nuri::tools::benchmark
