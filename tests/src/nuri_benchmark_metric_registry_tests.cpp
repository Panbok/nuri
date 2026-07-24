#include "tests_pch.h"

#include "nuri/gfx/ddgi/ddgi_types.h"
#include "nuri/tools/benchmark/benchmark_manifest.h"
#include "nuri/tools/benchmark/benchmark_metric_registry.h"
#include "nuri/tools/benchmark/benchmark_report.h"
#include "nuri/tools/benchmark/benchmark_runner.h"
#include "nuri/tools/core/result_envelope_v2.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>

namespace {

using namespace nuri::tools::benchmark;

[[nodiscard]] std::filesystem::path makeMetricRegistryTempPath() {
  const auto tick =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  return std::filesystem::temp_directory_path() /
         ("nuri_metric_registry_" + std::to_string(tick) + ".json");
}

TEST(NuriBenchmarkMetricRegistryTest, CoreDescriptorsAreTypedAndBounded) {
  const BenchmarkMetricDescriptor *cpu =
      findBenchmarkMetricDescriptor("cpu.render_submit_ms");
  ASSERT_NE(cpu, nullptr);
  EXPECT_EQ(cpu->idRule, BenchmarkMetricIdRule::Exact);
  EXPECT_EQ(cpu->unit, BenchmarkMetricUnit::Milliseconds);
  EXPECT_EQ(cpu->numericType, BenchmarkMetricNumericType::Float64);
  EXPECT_EQ(cpu->direction, BenchmarkMetricDirection::LowerIsBetter);
  EXPECT_EQ(cpu->aggregation,
            BenchmarkMetricAggregation::MedianAndP95AcrossMeasuredFrames);
  EXPECT_EQ(cpu->availability, BenchmarkMetricAvailability::EveryMeasuredFrame);
  EXPECT_EQ(cpu->samplingPhase,
            BenchmarkMetricSamplingPhase::CpuMeasuredRegion);
  EXPECT_EQ(cpu->gateRole, BenchmarkMetricGateRole::Primary);

  const BenchmarkMetricDescriptor *counter =
      findBenchmarkMetricDescriptor("renderer.visibility.meshlet_emitted");
  ASSERT_NE(counter, nullptr);
  EXPECT_EQ(counter->numericType,
            BenchmarkMetricNumericType::Uint64EncodedAsFloat64);
  EXPECT_EQ(counter->unit, BenchmarkMetricUnit::Count);
  EXPECT_EQ(counter->gateRole,
            BenchmarkMetricGateRole::WorkloadCharacterization);

  EXPECT_NE(findBenchmarkMetricDescriptor(
                "rendergraph.summary.transient_texture_physical_count"),
            nullptr);
  EXPECT_NE(findBenchmarkMetricDescriptor("memory.process.working_set_mb"),
            nullptr);
  EXPECT_NE(findBenchmarkMetricDescriptor("benchmark.camera.direction_delta"),
            nullptr);
  EXPECT_NE(findBenchmarkMetricDescriptor(
                "texture.normal_variance_artifact_build_ms"),
            nullptr);
  EXPECT_NE(findBenchmarkMetricDescriptor(
                "texture.io.normal_variance_artifact_write_mb"),
            nullptr);
  EXPECT_NE(findBenchmarkMetricDescriptor(
                "gpu.memory.aa.normal_variance_contract_textures_mb"),
            nullptr);
  EXPECT_NE(findBenchmarkMetricDescriptor(
                "texture.cache.normal_variance_contract_rejections"),
            nullptr);
  EXPECT_NE(findBenchmarkMetricDescriptor("renderer.aa.spatial_aa_allocations"),
            nullptr);
  EXPECT_NE(
      findBenchmarkMetricDescriptor("renderer.aa.spatial_aa_reallocations"),
      nullptr);
  for (std::string_view metric :
       {"renderer.opaque.meshlet_dispatches",
        "renderer.opaque.meshlet_task_groups",
        "renderer.opaque.meshlet_candidates",
        "renderer.opaque.meshlet_mode_required",
        "renderer.opaque.meshlet_mode_active",
        "renderer.opaque.meshlet_rejected_missing_feature",
        "renderer.opaque.meshlet_rejected_missing_asset_data",
        "renderer.opaque.meshlet_rejected_incompatible_frame",
        "renderer.opaque.meshlet_hybrid_coverage_classic_batches",
        "renderer.opaque.meshlet_hybrid_coverage_classic_instances",
        "renderer.opaque.auto_lod_active",
        "renderer.opaque.auto_lod_history_reset",
        "renderer.opaque.auto_lod_transitions",
        "renderer.opaque.auto_lod_lod0_instances",
        "renderer.opaque.auto_lod_lod1_instances"}) {
    EXPECT_NE(findBenchmarkMetricDescriptor(metric), nullptr) << metric;
  }

  const BenchmarkMetricDescriptor *frameGpu =
      findBenchmarkMetricDescriptor("gpu.frame_ms");
  ASSERT_NE(frameGpu, nullptr);
  EXPECT_EQ(frameGpu->unit, BenchmarkMetricUnit::Milliseconds);
  EXPECT_EQ(frameGpu->availability,
            BenchmarkMetricAvailability::WhenWholeFrameGpuTimingAvailable);
  EXPECT_EQ(frameGpu->gateRole, BenchmarkMetricGateRole::Primary);
  for (std::string_view scope :
       {"gpu.scopes.velocity_ms", "gpu.scopes.reactive_mask_ms",
        "gpu.scopes.taa_copy_back_ms", "gpu.scopes.gtao_temporal_ms"}) {
    EXPECT_NE(findBenchmarkMetricDescriptor(scope), nullptr) << scope;
  }

  const BenchmarkMetricDescriptor *motionClassRatio =
      findBenchmarkMetricDescriptor(
          "renderer.aa.motion_class_background_rotation_ratio");
  ASSERT_NE(motionClassRatio, nullptr);
  EXPECT_EQ(motionClassRatio->unit, BenchmarkMetricUnit::Ratio);
  EXPECT_EQ(motionClassRatio->availability,
            BenchmarkMetricAvailability::WhenMotionClassCoverageAvailable);
  EXPECT_EQ(benchmarkMetricAvailabilityName(motionClassRatio->availability),
            "when-motion-class-coverage-available");

  const BenchmarkMetricDescriptor *coverageRatio =
      findBenchmarkMetricDescriptor("renderer.ddgi.scene_coverage_ratio");
  ASSERT_NE(coverageRatio, nullptr);
  EXPECT_EQ(coverageRatio->unit, BenchmarkMetricUnit::Ratio);
  const BenchmarkMetricDescriptor *coverageResolve =
      findBenchmarkMetricDescriptor("renderer.ddgi.coverage_resolve_cpu_ms");
  ASSERT_NE(coverageResolve, nullptr);
  EXPECT_EQ(coverageResolve->unit, BenchmarkMetricUnit::Milliseconds);
  for (std::string_view id : {"renderer.ddgi.probe_state_readback_source_frame",
                              "renderer.ddgi.probe_state_readback_stale_frames",
                              "renderer.ddgi.primary_queries_issued",
                              "renderer.ddgi.total_queries_issued",
                              "renderer.ddgi.trace_counter_source_frame"}) {
    EXPECT_NE(findBenchmarkMetricDescriptor(id), nullptr) << id;
  }
  for (uint32_t slot = 0u; slot < nuri::kMaxDDGIEffectiveVolumes; ++slot) {
    const std::string prefix =
        "renderer.ddgi.volume" + std::to_string(slot) + ".";
    for (std::string_view suffix :
         {"total_probes", "invalid_probes", "updates", "primary_queries_issued",
          "update_age_p95", "scheduled_quota", "deficit", "starvation_frames",
          "confidence"}) {
      EXPECT_NE(findBenchmarkMetricDescriptor(prefix + std::string(suffix)),
                nullptr)
          << prefix << suffix;
    }
  }
}

TEST(NuriBenchmarkMetricRegistryTest,
     RegisteredFrameMeasurementsAreContiguousAndDoNotOwnMetricIds) {
  const auto cpuIndex = findExactBenchmarkMetricIndex("cpu.render_submit_ms");
  const auto counterIndex =
      findExactBenchmarkMetricIndex("renderer.opaque.total_instances");
  ASSERT_TRUE(cpuIndex.has_value());
  ASSERT_TRUE(counterIndex.has_value());
  EXPECT_FALSE(
      findExactBenchmarkMetricIndex("rendergraph.pass.000.opaque.cpu_ms")
          .has_value());
  ASSERT_EQ(benchmarkMetricDescriptor(*cpuIndex)->idOrRule,
            "cpu.render_submit_ms");

  static_assert(
      std::contiguous_iterator<BenchmarkFrameMeasurements::const_iterator>);
  BenchmarkFrameMeasurements measurements;
  measurements.reserve(3u);
  const size_t reservedCapacity = measurements.capacity();
  measurements.appendRegistered(*cpuIndex, 1.25);
  measurements.appendRegistered(*counterIndex, 0.0);
  measurements.appendRegistered(*cpuIndex, 1.5);
  measurements.appendRegistered(BenchmarkMetricIndex{}, 9.0);

  EXPECT_EQ(measurements.capacity(), reservedCapacity);
  EXPECT_EQ(measurements.registeredCount(), 3u);
  EXPECT_EQ(measurements.ownedMetricIdCount(), 0u);
  EXPECT_EQ(measurements.begin()->id(), "cpu.render_submit_ms");
  EXPECT_DOUBLE_EQ(measurements.begin()->second, 1.25);

  measurements.appendOwned("rendergraph.pass.000.opaque.cpu_ms", 0.75);
  EXPECT_EQ(measurements.ownedMetricIdCount(), 1u);
}

TEST(NuriBenchmarkMetricRegistryTest, OnlySafePassTimingRulesAreDynamic) {
  const BenchmarkMetricDescriptor *cpu = findBenchmarkMetricDescriptor(
      "rendergraph.pass.000.opaque_depth_pre_pass.cpu_ms");
  ASSERT_NE(cpu, nullptr);
  EXPECT_EQ(cpu->idRule, BenchmarkMetricIdRule::RenderGraphPassCpuTiming);
  const BenchmarkMetricDescriptor *gpu = findBenchmarkMetricDescriptor(
      "rendergraph.pass.1042.hdr_postprocess.gpu_ms");
  ASSERT_NE(gpu, nullptr);
  EXPECT_EQ(gpu->idRule, BenchmarkMetricIdRule::RenderGraphPassGpuTiming);

  EXPECT_EQ(findBenchmarkMetricDescriptor("rendergraph.pass.00.opaque.cpu_ms"),
            nullptr);
  EXPECT_EQ(findBenchmarkMetricDescriptor("rendergraph.pass.000.Opaque.cpu_ms"),
            nullptr);
  EXPECT_EQ(findBenchmarkMetricDescriptor(
                "rendergraph.pass.000.opaque/escape.cpu_ms"),
            nullptr);
  EXPECT_EQ(findBenchmarkMetricDescriptor("gpu.scopes.future_scope_ms"),
            nullptr);
  EXPECT_EQ(findBenchmarkMetricDescriptor("renderer.opaque.typo_draws"),
            nullptr);
  EXPECT_EQ(findBenchmarkMetricDescriptor("custom.required_ms"), nullptr);
}

TEST(NuriBenchmarkMetricRegistryTest,
     ManifestRejectsUnregisteredRequiredMetric) {
  const std::filesystem::path path = makeMetricRegistryTempPath();
  {
    std::ofstream file(path, std::ios::binary);
    ASSERT_TRUE(file.is_open());
    file << R"({
      "schemaVersion": 1,
      "id": "registry.unknown",
      "suite": "registry",
      "requiredMetrics": ["custom.required_ms"]
    })";
  }

  auto loaded = loadBenchmarkCaseManifest(path);
  EXPECT_TRUE(loaded.hasError());
  EXPECT_NE(loaded.error().find("unregistered"), std::string::npos);
  std::error_code error;
  std::filesystem::remove(path, error);
}

TEST(NuriBenchmarkMetricRegistryTest,
     ProfileRequiredMetricIsRejectedBeforeDryRunSetup) {
  BenchmarkCase benchmarkCase{};
  benchmarkCase.id = "registry.profile";
  benchmarkCase.suite = "registry";
  BenchmarkRunOptions options{};
  options.dryRun = true;
  const std::filesystem::path artifacts =
      makeMetricRegistryTempPath().replace_extension();
  options.artifactDir = artifacts;
  options.baselineProfileId = "local-nvrhi-visible";
  options.baselineProfileRequiredMetrics = {"custom.required_ms"};

  const BenchmarkRunResult result = runBenchmarkCase(benchmarkCase, options);
  EXPECT_EQ(result.exitCode, BenchmarkExitCode::InvalidInput);
  EXPECT_NE(result.message.find("unregistered"), std::string::npos);
  EXPECT_TRUE(result.report.frames.empty());
  EXPECT_FALSE(result.reportPath.empty());
  EXPECT_FALSE(result.envelopePath.empty());
  std::ifstream envelopeFile(result.envelopePath, std::ios::binary);
  ASSERT_TRUE(envelopeFile.is_open());
  const std::string envelopeJson((std::istreambuf_iterator<char>(envelopeFile)),
                                 std::istreambuf_iterator<char>());
  auto envelope = nuri::tools::core::readResultEnvelopeV2(envelopeJson);
  ASSERT_FALSE(envelope.hasError()) << envelope.error();
  EXPECT_EQ(envelope.value().status, nuri::tools::core::ToolOutcome::Invalid);
  EXPECT_EQ(envelope.value().exitCode,
            static_cast<int>(BenchmarkExitCode::InvalidInput));
  std::error_code error;
  std::filesystem::remove_all(artifacts, error);
}

TEST(NuriBenchmarkMetricRegistryTest,
     ZeroIsObservedWhileUnknownDiagnosticMetricRemainsExplicit) {
  BenchmarkReport report{};
  report.run.samples = 1u;
  report.run.measurementFrames = 1u;
  report.run.validForComparison = true;
  report.benchmarkCase.requiredMetrics = {"renderer.opaque.total_instances"};
  report.frames.push_back(BenchmarkFrameRecord{
      .frameIndex = 0u,
      .sampleIndex = 0u,
      .measured = true,
      .measurements = {{"renderer.opaque.total_instances", 0.0},
                       {"diagnostic.experimental", 0.0}},
  });

  computeBenchmarkReportStats(report);

  ASSERT_EQ(report.stats.count("renderer.opaque.total_instances"), 1u);
  EXPECT_EQ(report.stats.at("renderer.opaque.total_instances").count, 1u);
  EXPECT_DOUBLE_EQ(report.stats.at("renderer.opaque.total_instances").median,
                   0.0);
  EXPECT_TRUE(report.run.validForComparison);
  EXPECT_TRUE(report.unavailableMetrics.empty());
  ASSERT_EQ(report.stats.count("diagnostic.experimental"), 1u);
  ASSERT_EQ(report.unregisteredObservedMetrics.size(), 1u);
  EXPECT_EQ(report.unregisteredObservedMetrics.front(),
            "diagnostic.experimental");

  const std::filesystem::path path = makeMetricRegistryTempPath();
  auto written = writeBenchmarkReportFile(report, path, true);
  ASSERT_FALSE(written.hasError()) << written.error();
  auto loaded = readBenchmarkReportFile(path);
  ASSERT_FALSE(loaded.hasError()) << loaded.error();
  ASSERT_EQ(loaded->unregisteredObservedMetrics.size(), 1u);
  EXPECT_EQ(loaded->unregisteredObservedMetrics.front(),
            "diagnostic.experimental");
  std::error_code error;
  std::filesystem::remove(path, error);
}

} // namespace
