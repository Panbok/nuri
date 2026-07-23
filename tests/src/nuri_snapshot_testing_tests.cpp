#include "tests_pch.h"

#include "render_graph_test_support.h"

#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/tools/core/result_envelope_v2.h"
#include "nuri/tools/core/sha256.h"
#include "nuri/tools/snapshot/snapshot_baseline.h"
#include "nuri/tools/snapshot/snapshot_capture_point.h"
#include "nuri/tools/snapshot/snapshot_compare.h"
#include "nuri/tools/snapshot/snapshot_html_report.h"
#include "nuri/tools/snapshot/snapshot_image.h"
#include "nuri/tools/snapshot/snapshot_manifest.h"
#include "nuri/tools/snapshot/snapshot_report.h"
#include "nuri/tools/snapshot/snapshot_runner.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace nuri;
using namespace nuri::test_support;
using namespace nuri::tools::snapshot;

std::filesystem::path makeTempPath(std::string_view stem,
                                   std::string_view extension) {
  const auto tick =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  return std::filesystem::temp_directory_path() /
         ("nuri_" + std::string(stem) + "_" + std::to_string(tick) +
          std::string(extension));
}

void writeFile(const std::filesystem::path &path, std::string_view text) {
  std::ofstream file(path, std::ios::binary);
  ASSERT_TRUE(file.is_open());
  file << text;
}

std::string replaceFirst(std::string text, std::string_view needle,
                         std::string_view replacement) {
  const size_t position = text.find(needle);
  EXPECT_NE(position, std::string::npos);
  if (position != std::string::npos) {
    text.replace(position, needle.size(), replacement);
  }
  return text;
}

RenderCapturePoint makeCapturePoint(TextureHandle texture,
                                    std::string_view name = "final_color") {
  return RenderCapturePoint{
      .name = name,
      .version = 1u,
      .texture = texture,
      .format = Format::RGBA8_UNORM,
      .dimensions = TextureDimensions{.width = 2u, .height = 2u, .depth = 1u},
      .frameIndex = 7u,
      .kind = RenderCaptureValueKind::Color,
      .lifetime = RenderCaptureLifetimeClass::ToolCaptureTexture,
      .colorSpace = "display_sdr",
      .defaultCompareProfile = "ldr_color",
      .producerPassLabel = "test pass",
  };
}

SnapshotReport makeReport(std::filesystem::path caseDir = {}) {
  SnapshotReport report{};
  report.generatedAtUtc = "2026-06-27T00:00:00Z";
  report.command = "nuri-snapshot capture --case smoke.case";
  report.environment.repoRoot = "repo";
  report.environment.commitHash = "abc123";
  report.environment.branchName = "codex/snapshot";
  report.environment.dirty = true;
  report.environment.osName = "test-os";
  report.environment.cpuName = "test-cpu";
  report.environment.gpuBackend = "fake<gpu>&";
  report.environment.gpuDeviceName = "test-gpu";
  report.environment.gpuVendorId = 0x10deu;
  report.environment.gpuDeviceId = 0x2684u;
  report.environment.gpuDriverVersion = "test-driver";
  report.environment.resolvedWindowMode = "visible";
  report.snapshotCase.id = "case.<final>&color";
  report.snapshotCase.suite = "smoke";
  report.artifacts.caseDir = std::move(caseDir);
  report.artifacts.caseHtml = report.artifacts.caseDir / "report.html";
  report.captures.push_back(SnapshotCaptureReport{
      .target = "final<color>&",
      .artifactStem = "final_color",
      .profile = "ldr_color",
      .required = false,
      .available = true,
      .capturePointVersion = 1u,
      .captureFrameIndex = 4u,
      .kind = "color",
      .lifetime = "tool_capture_texture",
      .format = "RGBA8_UNORM",
      .colorSpace = "display_sdr",
      .width = 2u,
      .height = 2u,
      .actualHash =
          "sha256:"
          "0000000000000000000000000000000000000000000000000000000000000001",
      .expectedHash =
          "sha256:"
          "0000000000000000000000000000000000000000000000000000000000000002",
      .actual = "final_color.raw",
      .actualMetadata = "final_color.json",
      .preview = "final_color_preview.png",
      .expected = "expected.png",
      .diff = "diff.png",
      .metrics = SnapshotCompareMetrics{.meanAbsError = 0.1,
                                        .rmse = 0.2,
                                        .maxAbsError = 0.3,
                                        .p99AbsError = 0.25,
                                        .failingValues = 4u,
                                        .comparedValues = 16u},
      .semanticMetrics = SnapshotSemanticMetrics{.unit = "display_linear",
                                                 .meanError = 0.11,
                                                 .maxError = 0.31,
                                                 .validPixels = 4u,
                                                 .ignoredPixels = 1u,
                                                 .changedPixels = 2u,
                                                 .changedBoundsValid = true,
                                                 .minChangedX = 0u,
                                                 .minChangedY = 0u,
                                                 .maxChangedX = 1u,
                                                 .maxChangedY = 1u,
                                                 .maxErrorX = 1u,
                                                 .maxErrorY = 0u},
      .failedThresholds = {"max_abs_error"},
      .producerPassLabel = "present<pass>",
      .status = "captured",
      .statusReason = "capture<ok>&",
  });
  report.availableCapturePoints = {"final_color"};
  report.rendererMetricValues["renderer.custom"] = 42.5;
  report.captureSynchronization = "test_sync";
  report.reproduceCommand = "nuri-snapshot run --case smoke.case";
  report.warnings = {"test warning"};
  report.errors = {"test error"};
  report.rendererMetrics.frameIndex = 7u;
  report.rendererMetrics.antiAliasing.historyValid = true;
  report.rendererMetrics.antiAliasing.motionVectorFormat = Format::RG16_FLOAT;
  report.rendererMetrics.antiAliasing.velocityPassCount = 1u;
  report.rendererMetrics.antiAliasing.velocityDrawCount = 2u;
  report.rendererMetrics.antiAliasing.velocityPreviousTransformValidCount = 3u;
  report.rendererMetrics.antiAliasing.previousTransformCacheValid = true;
  report.rendererMetrics.antiAliasing
      .transparentTransmissionFeedbackRefreshCount = 1u;
  report.rendererMetrics.antiAliasing.transparentTransmissionBlendDrawCount =
      2u;
  report.rendererMetrics.antiAliasing
      .transparentTransmissionFeedbackSourceAvailable = 1u;
  report.rendererMetrics.antiAliasing.taaTransparentPostTaaDrawCount = 3u;
  report.rendererMetrics.visibility.gpuMainReadbackAvailable = 1u;
  report.rendererMetrics.visibility.gpuMainReadbackSourceFrame = 7u;
  report.rendererMetrics.visibility.gpuMainReadbackStaleFrameCount = 1u;
  report.rendererMetrics.visibility.gpuMainReadbackErrorCount = 2u;
  report.rendererMetrics.visibility.meshletReadbackAvailable = 1u;
  report.rendererMetrics.visibility.meshletReadbackSourceFrame = 6u;
  report.rendererMetrics.visibility.meshletReadbackStaleFrameCount = 2u;
  report.rendererMetrics.visibility.meshletReadbackErrorCount = 3u;
  report.rendererMetrics.visibility.meshletEmitted = 12u;
  report.rendererMetrics.visibility.meshletTaskGroupsExecuted = 2u;
  report.rendererMetrics.visibility.shadowCpuCandidates = 9u;
  report.rendererMetrics.visibility.shadowCpuRejected = 4u;
  report.rendererMetrics.shadow.cascadeCount = 2u;
  report.rendererMetrics.shadow.totalDraws = 4u;
  report.rendererMetrics.transparent.meshDraws = 5u;
  return report;
}

TEST(NuriSnapshotTestingTest, ManifestRejectsUnknownKeys) {
  const std::filesystem::path path = makeTempPath("snapshot_manifest", ".json");
  writeFile(path,
            R"json({
              "schemaVersion": 1,
              "id": "bad.case",
              "suite": "bad",
              "unexpected": true
            })json");

  auto loaded = loadSnapshotCaseManifest(path);
  EXPECT_TRUE(loaded.hasError());
  EXPECT_NE(loaded.error().find("unknown key"), std::string::npos);

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

TEST(NuriSnapshotTestingTest,
     ManifestParsesMsaaEnvironmentAndSpatialCleanupContract) {
  const std::filesystem::path path =
      makeTempPath("snapshot_msaa_environment", ".json");
  writeFile(path,
            R"json({
              "schemaVersion": 1,
              "id": "msaa.environment",
              "suite": "renderer",
              "environment": {
                "cubemap": {
                  "pathBase": "texturesRoot",
                  "path": "sky.hdr",
                  "kind": "EquirectHdrCubemap",
                  "debugName": "test_sky",
                  "required": true
                }
              },
              "settings": {
                "antiAliasing": {
                  "mode": "MSAA8x",
                  "spatialPostMsaaCleanup": true
                }
              },
              "requirements": {"msaaSamples": 8},
              "captures": [{"target": "final_color"}]
            })json");

  auto loaded = loadSnapshotCaseManifest(path);
  ASSERT_FALSE(loaded.hasError()) << loaded.error();
  ASSERT_TRUE(loaded.value().requirements.msaaSamples.has_value());
  EXPECT_EQ(*loaded.value().requirements.msaaSamples, 8u);
  EXPECT_EQ(loaded.value().settings.antiAliasing.mode,
            AntiAliasingMode::MSAA8x);
  EXPECT_TRUE(
      loaded.value().settings.antiAliasing.debug.spatialPostMsaaCleanup);
  EXPECT_EQ(loaded.value().environment.cubemap.pathBase, "texturesRoot");
  EXPECT_EQ(loaded.value().environment.cubemap.path,
            std::filesystem::path("sky.hdr"));
  EXPECT_EQ(loaded.value().environment.cubemap.kind, "EquirectHdrCubemap");
  EXPECT_EQ(loaded.value().environment.cubemap.debugName, "test_sky");
  EXPECT_TRUE(loaded.value().environment.cubemap.required);

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

TEST(NuriSnapshotTestingTest, ManifestParsesStrictVersionedDDGICoverage) {
  const std::filesystem::path path =
      makeTempPath("snapshot_ddgi_coverage", ".json");
  const std::string manifest = R"json({
    "schemaVersion": 1,
    "id": "ddgi.coverage",
    "suite": "ddgi",
    "settings": {"ddgi": {"coverage": {
      "schemaVersion": 1,
      "mode": "Hybrid",
      "constraintPolicy": "PreserveNearSpacing",
      "sceneBoundsSource": "Authored",
      "authoredBounds": {"min": [-4, -2, -6], "max": [8, 10, 12]},
      "cascadeCount": 4,
      "cascadeProbeCounts": [24, 14, 20],
      "requestedNearSpacing": [1, 1.25, 1.5],
      "spacingRatio": 2.5,
      "requestedCoverageHalfExtents": [60, 35, 55],
      "scenePaddingCells": 3,
      "transitionCells": 2,
      "includeAuthoredVolumes": false
    }}},
    "captures": [{"target": "final_color"}]
  })json";
  writeFile(path, manifest);

  auto loaded = loadSnapshotCaseManifest(path);
  ASSERT_FALSE(loaded.hasError()) << loaded.error();
  const DDGICoverageSettings &coverage = loaded.value().settings.ddgi.coverage;
  EXPECT_EQ(coverage.mode, DDGICoverageMode::Hybrid);
  EXPECT_EQ(coverage.constraintPolicy,
            DDGICoverageConstraintPolicy::PreserveNearSpacing);
  EXPECT_EQ(coverage.sceneBoundsSource, DDGISceneBoundsSource::Authored);
  EXPECT_TRUE(coverage.authoredBounds.valid);
  EXPECT_TRUE(coverage.authoredBounds.complete);
  EXPECT_FLOAT_EQ(coverage.authoredBounds.bounds.min_.x, -4.0f);
  EXPECT_FLOAT_EQ(coverage.authoredBounds.bounds.min_.y, -2.0f);
  EXPECT_FLOAT_EQ(coverage.authoredBounds.bounds.min_.z, -6.0f);
  EXPECT_FLOAT_EQ(coverage.authoredBounds.bounds.max_.x, 8.0f);
  EXPECT_FLOAT_EQ(coverage.authoredBounds.bounds.max_.y, 10.0f);
  EXPECT_FLOAT_EQ(coverage.authoredBounds.bounds.max_.z, 12.0f);
  EXPECT_EQ(coverage.cascadeCount, 4u);
  EXPECT_EQ(coverage.cascadeProbeCounts.x, 24u);
  EXPECT_EQ(coverage.cascadeProbeCounts.y, 14u);
  EXPECT_EQ(coverage.cascadeProbeCounts.z, 20u);
  EXPECT_FLOAT_EQ(coverage.requestedNearSpacing.x, 1.0f);
  EXPECT_FLOAT_EQ(coverage.requestedNearSpacing.y, 1.25f);
  EXPECT_FLOAT_EQ(coverage.requestedNearSpacing.z, 1.5f);
  EXPECT_FLOAT_EQ(coverage.spacingRatio, 2.5f);
  EXPECT_FLOAT_EQ(coverage.requestedCoverageHalfExtents.x, 60.0f);
  EXPECT_FLOAT_EQ(coverage.requestedCoverageHalfExtents.y, 35.0f);
  EXPECT_FLOAT_EQ(coverage.requestedCoverageHalfExtents.z, 55.0f);
  EXPECT_EQ(coverage.scenePaddingCells, 3u);
  EXPECT_EQ(coverage.transitionCells, 2u);
  EXPECT_FALSE(coverage.includeAuthoredVolumes);

  writeFile(path,
            R"json({
              "schemaVersion": 1,
              "id": "ddgi.coverage.default",
              "suite": "ddgi",
              "settings": {"ddgi": {"enabled": true}},
              "captures": [{"target": "final_color"}]
            })json");
  loaded = loadSnapshotCaseManifest(path);
  ASSERT_FALSE(loaded.hasError()) << loaded.error();
  EXPECT_EQ(loaded.value().settings.ddgi.coverage.mode,
            DDGICoverageMode::Manual);
  EXPECT_FALSE(loaded.value().settings.ddgi.coverage.authoredBounds.valid);

  const std::array invalidManifests{
      replaceFirst(manifest, "\"schemaVersion\": 1,\n      \"mode\"",
                   "\"schemaVersion\": 2,\n      \"mode\""),
      replaceFirst(manifest, "\"includeAuthoredVolumes\": false",
                   "\"includeAuthoredVolumes\": false, \"unexpected\": true"),
      replaceFirst(
          manifest,
          "\"authoredBounds\": {\"min\": [-4, -2, -6], \"max\": [8, 10, 12]}",
          "\"authoredBounds\": {\"min\": [-4, -2, -6]}")};
  for (const std::string &invalidManifest : invalidManifests) {
    writeFile(path, invalidManifest);
    auto invalid = loadSnapshotCaseManifest(path);
    EXPECT_TRUE(invalid.hasError());
    if (invalid.hasError()) {
      EXPECT_NE(invalid.error().find("settings.ddgi.coverage"),
                std::string::npos);
    }
  }

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

TEST(NuriSnapshotTestingTest, ManifestRejectsRemovedBackends) {
  const std::filesystem::path path =
      makeTempPath("snapshot_removed_backend", ".json");
  writeFile(path,
            R"json({
              "schemaVersion": 1,
              "id": "backend.removed",
              "suite": "backend",
              "backend": "unsupported"
            })json");
  auto loaded = loadSnapshotCaseManifest(path);
  EXPECT_TRUE(loaded.hasError());
  EXPECT_NE(loaded.error().find("default or nvrhi"), std::string::npos);

  writeFile(path,
            R"json({
              "schemaVersion": 1,
              "id": "backend.requirement_removed",
              "suite": "backend",
              "backend": "nvrhi",
              "requirements": {"backends": ["unsupported"]}
            })json");
  loaded = loadSnapshotCaseManifest(path);
  EXPECT_TRUE(loaded.hasError());
  EXPECT_NE(loaded.error().find("unsupported backend"), std::string::npos);

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

TEST(NuriSnapshotTestingTest,
     ManifestAppliesShadowPresetBeforeExplicitOverrides) {
  const std::filesystem::path path =
      makeTempPath("snapshot_shadow_overrides", ".json");
  const std::string manifest = R"json({
    "schemaVersion": 1,
    "id": "shadow.overrides",
    "suite": "shadow",
    "settings": {"shadow": {
      "qualityPreset": "Ultra",
      "depthFormat": "D32_FLOAT",
      "maxDistance": 73.0,
      "maxDistanceFadeFraction": 0.25,
      "splitLambda": 0.2,
      "cascadeBlendFraction": 0.03,
      "pcfSampleCount": 7,
      "sdsmTemporalBlend": 0.4,
      "enableCascadeCasterCulling": false,
      "debug": {"visualizeShadowFactor": true}
    }},
    "captures": [{"target": "final_color"}]
  })json";
  writeFile(path, manifest);

  auto loaded = loadSnapshotCaseManifest(path);
  ASSERT_FALSE(loaded.hasError()) << loaded.error();
  const RenderSettings::ShadowSettings &shadow = loaded.value().settings.shadow;
  EXPECT_EQ(shadow.qualityPreset, ShadowQualityPreset::Ultra);
  EXPECT_EQ(shadow.depthFormat, Format::D32_FLOAT);
  EXPECT_FLOAT_EQ(shadow.maxDistance, 73.0f);
  EXPECT_FLOAT_EQ(shadow.maxDistanceFadeFraction, 0.25f);
  EXPECT_FLOAT_EQ(shadow.splitLambda, 0.2f);
  EXPECT_FLOAT_EQ(shadow.cascadeBlendFraction, 0.03f);
  EXPECT_EQ(shadow.pcfSampleCount, 7u);
  EXPECT_FLOAT_EQ(shadow.sdsmTemporalBlend, 0.4f);
  EXPECT_FALSE(shadow.debug.enableCascadeCasterCulling);
  EXPECT_TRUE(shadow.debug.visualizeShadowFactor);

  writeFile(path, replaceFirst(manifest, "D32_FLOAT", "invalid"));
  auto invalid = loadSnapshotCaseManifest(path);
  EXPECT_TRUE(invalid.hasError());
  EXPECT_NE(invalid.error().find("depthFormat"), std::string::npos);

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

TEST(NuriSnapshotTestingTest, ManifestRejectsUnsafeIdentifiers) {
  const std::filesystem::path path =
      makeTempPath("snapshot_manifest_traversal", ".json");
  writeFile(path,
            R"json({
              "schemaVersion": 1,
              "id": "../outside",
              "suite": "smoke",
              "captures": ["final_color"]
            })json");

  auto loaded = loadSnapshotCaseManifest(path);
  EXPECT_TRUE(loaded.hasError());
  EXPECT_NE(loaded.error().find("id"), std::string::npos);

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

TEST(NuriSnapshotTestingTest, ManifestRejectsKnownNotCapturableTargets) {
  const std::filesystem::path path =
      makeTempPath("snapshot_manifest_diagnostic", ".json");
  writeFile(path,
            R"json({
              "schemaVersion": 1,
              "id": "bad.diagnostic",
              "suite": "bad",
              "captures": [
                {
                  "target": "gtao_raw",
                  "required": true
                }
              ]
            })json");

  auto loaded = loadSnapshotCaseManifest(path);
  EXPECT_TRUE(loaded.hasError());
  EXPECT_NE(loaded.error().find("known but not capturable yet"),
            std::string::npos);

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

TEST(NuriSnapshotTestingTest, ManifestRejectsUnknownAndMismatchedProfiles) {
  const auto checkProfile = [](std::string_view profile) {
    const std::filesystem::path path =
        makeTempPath("snapshot_manifest_profile", ".json");
    writeFile(path, std::string(R"json({
      "schemaVersion": 1,
      "id": "bad.profile",
      "suite": "bad",
      "captures": [{"target": "scene_depth", "profile": ")json") +
                        std::string(profile) + R"json("}]
    })json");
    auto loaded = loadSnapshotCaseManifest(path);
    std::error_code ec;
    std::filesystem::remove(path, ec);
    return loaded;
  };

  auto unknown = checkProfile("typo_profile");
  EXPECT_TRUE(unknown.hasError());
  EXPECT_NE(unknown.error().find("unknown compare profile"), std::string::npos);

  auto mismatched = checkProfile("ldr_color");
  EXPECT_TRUE(mismatched.hasError());
  EXPECT_NE(mismatched.error().find("not compatible"), std::string::npos);
}

TEST(NuriSnapshotTestingTest, CatalogListsCapturableAndDiagnosticTargets) {
  const SnapshotCaptureCatalogEntry *finalColor =
      findSnapshotCaptureCatalogEntry("final_color");
  ASSERT_NE(finalColor, nullptr);
  EXPECT_EQ(finalColor->availability, SnapshotCaptureAvailability::FirstSlice);
  EXPECT_EQ(finalColor->defaultCompareProfile, "ldr_color");

  const SnapshotCaptureCatalogEntry *gtaoEdges =
      findSnapshotCaptureCatalogEntry("gtao_edges");
  ASSERT_NE(gtaoEdges, nullptr);
  EXPECT_EQ(gtaoEdges->availability, SnapshotCaptureAvailability::FirstSlice);
  EXPECT_TRUE(gtaoEdges->diagnosticWork.empty());

  const SnapshotCaptureCatalogEntry *motionClass =
      findSnapshotCaptureCatalogEntry("motion_class");
  ASSERT_NE(motionClass, nullptr);
  EXPECT_EQ(motionClass->availability, SnapshotCaptureAvailability::FirstSlice);
  EXPECT_EQ(motionClass->defaultCompareProfile, "mask");

  const SnapshotCaptureCatalogEntry *gtaoPreviousDepth =
      findSnapshotCaptureCatalogEntry("gtao_previous_depth");
  ASSERT_NE(gtaoPreviousDepth, nullptr);
  EXPECT_EQ(gtaoPreviousDepth->availability,
            SnapshotCaptureAvailability::FirstSlice);
  EXPECT_EQ(gtaoPreviousDepth->defaultCompareProfile, "depth");
  for (uint32_t slot = 0u; slot < kMaxDDGIEffectiveVolumes; ++slot) {
    const std::string irradiance =
        "ddgi_volume" + std::to_string(slot) + "_irradiance_atlas";
    const std::string distance =
        "ddgi_volume" + std::to_string(slot) + "_distance_atlas";
    const SnapshotCaptureCatalogEntry *irradianceEntry =
        findSnapshotCaptureCatalogEntry(irradiance);
    const SnapshotCaptureCatalogEntry *distanceEntry =
        findSnapshotCaptureCatalogEntry(distance);
    ASSERT_NE(irradianceEntry, nullptr) << irradiance;
    ASSERT_NE(distanceEntry, nullptr) << distance;
    EXPECT_EQ(irradianceEntry->version, kDDGICaptureSemanticsVersion);
    EXPECT_EQ(distanceEntry->version, kDDGICaptureSemanticsVersion);
  }
  for (std::string_view semantic :
       {"ddgi_coverage_debug_preview", "ddgi_classification_debug_preview",
        "ddgi_dirty_region_debug_preview"}) {
    const SnapshotCaptureCatalogEntry *entry =
        findSnapshotCaptureCatalogEntry(semantic);
    ASSERT_NE(entry, nullptr) << semantic;
    EXPECT_EQ(entry->version, kDDGICaptureSemanticsVersion);
    EXPECT_EQ(entry->kind, RenderCaptureValueKind::DebugPreview);
  }
}

TEST(NuriSnapshotTestingTest, CaptureRequestsAndRegistryAreStable) {
  RenderCaptureRequest requests{};
  requests.request("final_color");
  requests.request("final_color");
  requests.request("scene_depth");
  ASSERT_EQ(requests.names().size(), 2u);
  EXPECT_TRUE(requests.contains("final_color"));
  EXPECT_FALSE(requests.contains("missing"));

  RenderCaptureRegistry registry{};
  registry.publish(makeCapturePoint(TextureHandle{}, "ignored"));
  EXPECT_EQ(registry.points().size(), 0u);

  TextureHandle texture{.index = 3u, .generation = 1u};
  registry.publish(makeCapturePoint(texture));
  ASSERT_NE(registry.find("final_color"), nullptr);
  EXPECT_EQ(registry.points().size(), 1u);

  RenderCapturePoint replacement = makeCapturePoint(texture);
  replacement.frameIndex = 8u;
  replacement.producerPassLabel = "replacement";
  registry.publish(replacement);
  ASSERT_NE(registry.find("final_color"), nullptr);
  EXPECT_EQ(registry.find("final_color")->frameIndex, 8u);
  EXPECT_EQ(registry.points().size(), 1u);

  registry.clear();
  EXPECT_TRUE(registry.points().empty());
}

TEST(NuriSnapshotTestingTest, FakeGpuTextureReadbackIsTightlyPacked) {
  FakeRendererGPUDevice gpu;
  auto textureResult = gpu.createTexture(
      makeTransientTextureDesc(Format::RGBA8_UNORM, 3u, 2u), "snapshot_source");
  ASSERT_FALSE(textureResult.hasError()) << textureResult.error();

  std::array<std::byte, 24u> bytes{};
  for (size_t i = 0u; i < bytes.size(); ++i) {
    bytes[i] = static_cast<std::byte>(i);
  }
  auto seed = gpu.seedTextureBytes(textureResult.value(), bytes);
  ASSERT_FALSE(seed.hasError()) << seed.error();

  std::array<std::byte, 8u> out{};
  auto read = gpu.readTexture(textureResult.value(),
                              TextureReadbackRegion{
                                  .x = 1u,
                                  .y = 1u,
                                  .width = 2u,
                                  .height = 1u,
                              },
                              out);
  ASSERT_FALSE(read.hasError()) << read.error();
  for (size_t i = 0u; i < out.size(); ++i) {
    EXPECT_EQ(out[i], bytes[16u + i]);
  }
}

TEST(NuriSnapshotTestingTest, SnapshotArtifactsDecodeAndCompare) {
  FakeRendererGPUDevice gpu;
  auto textureResult = gpu.createTexture(
      makeTransientTextureDesc(Format::RGBA8_UNORM, 2u, 2u), "snapshot_source");
  ASSERT_FALSE(textureResult.hasError()) << textureResult.error();

  const std::array<std::byte, 16u> bytes{
      std::byte{0},   std::byte{0},   std::byte{0},   std::byte{255},
      std::byte{255}, std::byte{0},   std::byte{0},   std::byte{255},
      std::byte{0},   std::byte{255}, std::byte{0},   std::byte{255},
      std::byte{0},   std::byte{0},   std::byte{255}, std::byte{255},
  };
  auto seed = gpu.seedTextureBytes(textureResult.value(), bytes);
  ASSERT_FALSE(seed.hasError()) << seed.error();

  auto readback =
      readSnapshotCapture(gpu, makeCapturePoint(textureResult.value()));
  ASSERT_FALSE(readback.hasError()) << readback.error();
  EXPECT_EQ(readback.value().rowStride, 8u);
  EXPECT_EQ(readback.value().hash, snapshotHashBytes(readback.value().bytes));

  auto decoded = decodeSnapshotImage(readback.value());
  ASSERT_FALSE(decoded.hasError()) << decoded.error();
  EXPECT_EQ(decoded.value().width, 2u);
  EXPECT_EQ(decoded.value().height, 2u);
  EXPECT_EQ(decoded.value().channelCount, 4u);
  EXPECT_FLOAT_EQ(decoded.value().values[3], 1.0f);

  SnapshotArtifactPaths paths{};
  const std::filesystem::path stem = makeTempPath("snapshot_artifact", "");
  auto written = writeSnapshotArtifacts(readback.value(), stem, paths);
  ASSERT_FALSE(written.hasError()) << written.error();
  EXPECT_TRUE(std::filesystem::exists(paths.raw));
  EXPECT_TRUE(std::filesystem::exists(paths.metadata));
  EXPECT_TRUE(std::filesystem::exists(paths.preview));
  EXPECT_EQ(paths.raw.extension(), ".png");

  auto actual = readSnapshotImageFile(paths.raw);
  ASSERT_FALSE(actual.hasError()) << actual.error();
  EXPECT_EQ(actual.value().width, 2u);
  EXPECT_EQ(actual.value().height, 2u);
  auto preview = readSnapshotImageFile(paths.preview);
  ASSERT_FALSE(preview.hasError()) << preview.error();
  EXPECT_EQ(preview.value().width, 2u);
  EXPECT_EQ(preview.value().height, 2u);

  SnapshotCompareResult compared = compareSnapshotImages(
      decoded.value(), decoded.value(), builtinSnapshotCompareProfile("exact"));
  EXPECT_TRUE(compared.compatible);
  EXPECT_TRUE(compared.passed);
  EXPECT_EQ(compared.metrics.failingValues, 0u);

  const std::filesystem::path comparisonPath =
      makeTempPath("snapshot_comparison", ".json");
  auto comparisonWritten =
      writeSnapshotComparisonFile(compared, "exact", comparisonPath);
  ASSERT_FALSE(comparisonWritten.hasError()) << comparisonWritten.error();
  std::ifstream comparisonFile(comparisonPath, std::ios::binary);
  const std::string comparisonJson(
      (std::istreambuf_iterator<char>(comparisonFile)),
      std::istreambuf_iterator<char>());
  EXPECT_NE(comparisonJson.find("\"kind\": \"nuri.snapshot.comparison\""),
            std::string::npos);
  EXPECT_NE(comparisonJson.find("\"passed\": true"), std::string::npos);

  std::error_code ec;
  std::filesystem::remove(paths.raw, ec);
  std::filesystem::remove(paths.metadata, ec);
  std::filesystem::remove(paths.preview, ec);
  std::filesystem::remove(comparisonPath, ec);
}

TEST(NuriSnapshotTestingTest, MipReadbackUsesReducedDimensionsEverywhere) {
  SnapshotReadbackImage readback{};
  readback.point = makeCapturePoint(TextureHandle{});
  readback.point.dimensions =
      TextureDimensions{.width = 4u, .height = 4u, .depth = 1u};
  readback.point.mip = 1u;
  readback.point.ddgiMetadata = DDGICaptureMetadata{
      .effectiveKeyHash = 0x123456789abcdef0ull,
      .coverageGeneration = 7u,
      .layoutGeneration = 8u,
      .resourceGeneration = 9u,
      .sceneBoundsGeneration = 10u,
      .effectiveKind = 2u,
      .cascadeIndex = 3u,
      .ringOrigin = {1u, 2u, 3u},
      .cameraCell = {-4, 5, -6},
      .requestedHalfExtents = {55.0f, 30.0f, 55.0f},
      .achievedHalfExtents = {60.0f, 36.0f, 60.0f},
      .fadeStartHalfExtents = {52.0f, 28.0f, 52.0f},
      .fadeEndHalfExtents = {58.0f, 34.0f, 58.0f},
      .transitionCells = 2u,
      .valid = 1u,
  };
  readback.rowStride = 8u;
  readback.bytes.assign(16u, std::byte{255});
  readback.hash = snapshotHashBytes(readback.bytes);

  auto decoded = decodeSnapshotImage(readback);
  ASSERT_FALSE(decoded.hasError()) << decoded.error();
  EXPECT_EQ(decoded.value().width, 2u);
  EXPECT_EQ(decoded.value().height, 2u);
  EXPECT_EQ(decoded.value().values.size(), 16u);

  SnapshotArtifactPaths paths{};
  const std::filesystem::path stem = makeTempPath("snapshot_mip", "");
  auto written = writeSnapshotArtifacts(readback, stem, paths);
  ASSERT_FALSE(written.hasError()) << written.error();
  auto image = readSnapshotImageFile(paths.raw);
  ASSERT_FALSE(image.hasError()) << image.error();
  EXPECT_EQ(image.value().width, 2u);
  EXPECT_EQ(image.value().height, 2u);
  auto metadata = readSnapshotArtifactMetadata(paths.metadata);
  ASSERT_FALSE(metadata.hasError()) << metadata.error();
  EXPECT_EQ(metadata.value().width, 2u);
  EXPECT_EQ(metadata.value().height, 2u);
  EXPECT_EQ(metadata.value().mip, 1u);
  EXPECT_EQ(metadata.value().kind, "color");
  EXPECT_EQ(metadata.value().profile, "ldr_color");
  EXPECT_EQ(metadata.value().ddgiMetadata.effectiveKeyHash,
            0x123456789abcdef0ull);
  EXPECT_EQ(metadata.value().ddgiMetadata.coverageGeneration, 7u);
  EXPECT_EQ(metadata.value().ddgiMetadata.ringOrigin, glm::uvec3(1u, 2u, 3u));
  EXPECT_EQ(metadata.value().ddgiMetadata.cameraCell, glm::ivec3(-4, 5, -6));
  EXPECT_EQ(metadata.value().ddgiMetadata.transitionCells, 2u);
  EXPECT_EQ(metadata.value().ddgiMetadata.valid, 1u);

  std::error_code ec;
  std::filesystem::remove(paths.raw, ec);
  std::filesystem::remove(paths.metadata, ec);
  std::filesystem::remove(paths.preview, ec);
}

TEST(NuriSnapshotTestingTest, FloatArtifactsWriteExrAndPreviewPng) {
  FakeRendererGPUDevice gpu;
  auto textureResult = gpu.createTexture(
      makeTransientTextureDesc(Format::R32_FLOAT, 2u, 1u), "snapshot_depth");
  ASSERT_FALSE(textureResult.hasError()) << textureResult.error();

  std::array<float, 2u> values{0.25f, 0.75f};
  std::array<std::byte, sizeof(values)> bytes{};
  std::memcpy(bytes.data(), values.data(), bytes.size());
  auto seed = gpu.seedTextureBytes(textureResult.value(), bytes);
  ASSERT_FALSE(seed.hasError()) << seed.error();

  RenderCapturePoint point =
      makeCapturePoint(textureResult.value(), "scene_depth");
  point.format = Format::R32_FLOAT;
  point.dimensions = TextureDimensions{.width = 2u, .height = 1u, .depth = 1u};
  point.kind = RenderCaptureValueKind::Depth;
  point.colorSpace = "depth_linear";
  point.defaultCompareProfile = "depth";

  auto readback = readSnapshotCapture(gpu, point);
  ASSERT_FALSE(readback.hasError()) << readback.error();

  SnapshotArtifactPaths paths{};
  const std::filesystem::path stem = makeTempPath("snapshot_exr", "");
  auto written = writeSnapshotArtifacts(readback.value(), stem, paths);
  ASSERT_FALSE(written.hasError()) << written.error();
  EXPECT_EQ(paths.raw.extension(), ".exr");
  EXPECT_TRUE(std::filesystem::exists(paths.preview));

  auto actual = readSnapshotImageFile(paths.raw);
  ASSERT_FALSE(actual.hasError()) << actual.error();
  ASSERT_EQ(actual.value().values.size(), 2u);
  EXPECT_NEAR(actual.value().values[0], 0.25f, 1.0e-6f);
  EXPECT_NEAR(actual.value().values[1], 0.75f, 1.0e-6f);

  std::error_code ec;
  std::filesystem::remove(paths.raw, ec);
  std::filesystem::remove(paths.metadata, ec);
  std::filesystem::remove(paths.preview, ec);
}

TEST(NuriSnapshotTestingTest, ReportJsonRoundTripsCaptureSummary) {
  const std::filesystem::path path = makeTempPath("snapshot_report", ".json");
  SnapshotReport report = makeReport(path.parent_path());

  auto written = writeSnapshotReportFile(report, path);
  ASSERT_FALSE(written.hasError()) << written.error();

  auto loaded = readSnapshotReportFile(path);
  ASSERT_FALSE(loaded.hasError()) << loaded.error();
  EXPECT_EQ(loaded.value().kind, "nuri.snapshot.report");
  EXPECT_EQ(loaded.value().baselineProfile, report.baselineProfile);
  EXPECT_EQ(loaded.value().baselineProfileCompatible,
            report.baselineProfileCompatible);
  EXPECT_EQ(loaded.value().snapshotCase.id, "case.<final>&color");
  ASSERT_EQ(loaded.value().captures.size(), 1u);
  EXPECT_EQ(loaded.value().artifacts.caseHtml, report.artifacts.caseHtml);
  EXPECT_EQ(loaded.value().captures[0].target, "final<color>&");
  EXPECT_FALSE(loaded.value().captures[0].required);
  EXPECT_EQ(loaded.value().captures[0].preview, "final_color_preview.png");
  EXPECT_EQ(loaded.value().captures[0].format, "RGBA8_UNORM");
  EXPECT_EQ(loaded.value().captures[0].kind, "color");
  EXPECT_EQ(loaded.value().captures[0].producerPassLabel, "present<pass>");
  EXPECT_EQ(loaded.value().captures[0].status, "captured");
  EXPECT_EQ(loaded.value().captures[0].width, 2u);
  EXPECT_EQ(loaded.value().environment.commitHash, "abc123");
  EXPECT_TRUE(loaded.value().environment.dirty);
  EXPECT_EQ(loaded.value().environment.gpuDeviceName, "test-gpu");
  EXPECT_EQ(loaded.value().environment.gpuVendorId, 0x10deu);
  EXPECT_EQ(loaded.value().environment.gpuDeviceId, 0x2684u);
  EXPECT_EQ(loaded.value().environment.gpuDriverVersion, "test-driver");
  EXPECT_EQ(loaded.value().command, report.command);
  EXPECT_EQ(loaded.value().captures[0].expectedHash,
            "sha256:"
            "0000000000000000000000000000000000000000000000000000000000000002");
  EXPECT_DOUBLE_EQ(loaded.value().captures[0].metrics.maxAbsError, 0.3);
  EXPECT_EQ(loaded.value().captures[0].semanticMetrics.unit, "display_linear");
  EXPECT_DOUBLE_EQ(loaded.value().captures[0].semanticMetrics.maxError, 0.31);
  EXPECT_EQ(loaded.value().captures[0].semanticMetrics.changedPixels, 2u);
  EXPECT_TRUE(loaded.value().captures[0].semanticMetrics.changedBoundsValid);
  EXPECT_EQ(loaded.value().captures[0].failedThresholds,
            std::vector<std::string>{"max_abs_error"});
  EXPECT_EQ(loaded.value().availableCapturePoints,
            std::vector<std::string>{"final_color"});
  EXPECT_DOUBLE_EQ(loaded.value().rendererMetricValues.at("renderer.custom"),
                   42.5);
  EXPECT_EQ(loaded.value().captureSynchronization, "test_sync");
  EXPECT_EQ(loaded.value().reproduceCommand, report.reproduceCommand);
  EXPECT_EQ(loaded.value().warnings, std::vector<std::string>{"test warning"});
  EXPECT_EQ(loaded.value().errors, std::vector<std::string>{"test error"});
  EXPECT_EQ(loaded.value().rendererMetrics.frameIndex, 7u);
  EXPECT_TRUE(loaded.value().rendererMetrics.antiAliasing.historyValid);
  EXPECT_EQ(loaded.value().rendererMetrics.antiAliasing.motionVectorFormat,
            Format::RG16_FLOAT);
  EXPECT_EQ(loaded.value().rendererMetrics.antiAliasing.velocityPassCount, 1u);
  EXPECT_EQ(
      loaded.value()
          .rendererMetrics.antiAliasing.velocityPreviousTransformValidCount,
      3u);
  EXPECT_TRUE(
      loaded.value().rendererMetrics.antiAliasing.previousTransformCacheValid);
  EXPECT_EQ(loaded.value()
                .rendererMetrics.antiAliasing
                .transparentTransmissionFeedbackRefreshCount,
            1u);
  EXPECT_EQ(
      loaded.value()
          .rendererMetrics.antiAliasing.transparentTransmissionBlendDrawCount,
      2u);
  EXPECT_EQ(loaded.value()
                .rendererMetrics.antiAliasing.taaTransparentPostTaaDrawCount,
            3u);
  EXPECT_EQ(loaded.value().rendererMetrics.shadow.cascadeCount, 2u);
  EXPECT_EQ(loaded.value().rendererMetrics.visibility.gpuMainReadbackAvailable,
            1u);
  EXPECT_EQ(
      loaded.value().rendererMetrics.visibility.gpuMainReadbackSourceFrame, 7u);
  EXPECT_EQ(
      loaded.value().rendererMetrics.visibility.gpuMainReadbackStaleFrameCount,
      1u);
  EXPECT_EQ(loaded.value().rendererMetrics.visibility.gpuMainReadbackErrorCount,
            2u);
  EXPECT_EQ(loaded.value().rendererMetrics.visibility.meshletReadbackAvailable,
            1u);
  EXPECT_EQ(
      loaded.value().rendererMetrics.visibility.meshletReadbackSourceFrame, 6u);
  EXPECT_EQ(
      loaded.value().rendererMetrics.visibility.meshletReadbackStaleFrameCount,
      2u);
  EXPECT_EQ(loaded.value().rendererMetrics.visibility.meshletReadbackErrorCount,
            3u);
  EXPECT_EQ(loaded.value().rendererMetrics.visibility.meshletEmitted, 12u);
  EXPECT_EQ(loaded.value().rendererMetrics.visibility.meshletTaskGroupsExecuted,
            2u);
  EXPECT_EQ(loaded.value().rendererMetrics.visibility.shadowCpuCandidates, 9u);
  EXPECT_EQ(loaded.value().rendererMetrics.visibility.shadowCpuRejected, 4u);
  EXPECT_EQ(loaded.value().rendererMetrics.shadow.totalDraws, 4u);
  EXPECT_EQ(loaded.value().rendererMetrics.transparent.meshDraws, 5u);
  auto html = writeSnapshotHtmlReport(loaded.value());
  ASSERT_FALSE(html.hasError()) << html.error();
  EXPECT_NE(html.value().find("src=\"final_color_preview.png\""),
            std::string::npos);
  EXPECT_NE(html.value().find("Blink actual / expected"), std::string::npos);
  EXPECT_NE(html.value().find("semantic mean / max"), std::string::npos);
  EXPECT_NE(html.value().find("display_linear"), std::string::npos);
  EXPECT_EQ(html.value().find("src=\"\""), std::string::npos);

  auto json = writeSnapshotReportJson(loaded.value());
  ASSERT_FALSE(json.hasError()) << json.error();
  EXPECT_NE(json.value().find("\"rendererMetrics\""), std::string::npos);
  EXPECT_NE(json.value().find("\"velocityPassCount\": 1"), std::string::npos);
  EXPECT_NE(json.value().find("\"velocityPreviousTransformValidCount\": 3"),
            std::string::npos);
  EXPECT_NE(json.value().find("\"cascadeCount\": 2"), std::string::npos);
  EXPECT_NE(json.value().find("\"gpuMainReadbackAvailable\": 1"),
            std::string::npos);
  EXPECT_NE(json.value().find("\"gpuMainReadbackSourceFrame\": 7"),
            std::string::npos);
  EXPECT_NE(json.value().find("\"gpuMainReadbackStaleFrameCount\": 1"),
            std::string::npos);
  EXPECT_NE(json.value().find("\"gpuMainReadbackErrorCount\": 2"),
            std::string::npos);
  EXPECT_NE(json.value().find("\"meshletReadbackAvailable\": 1"),
            std::string::npos);
  EXPECT_NE(json.value().find("\"meshletReadbackSourceFrame\": 6"),
            std::string::npos);
  EXPECT_NE(json.value().find("\"meshletReadbackStaleFrameCount\": 2"),
            std::string::npos);
  EXPECT_NE(json.value().find("\"meshletReadbackErrorCount\": 3"),
            std::string::npos);
  EXPECT_NE(json.value().find("\"meshletEmitted\": 12"), std::string::npos);
  EXPECT_NE(json.value().find("\"meshletTaskGroupsExecuted\": 2"),
            std::string::npos);
  EXPECT_NE(json.value().find("\"shadowCpuCandidates\": 9"), std::string::npos);
  EXPECT_NE(json.value().find("\"shadowCpuRejected\": 4"), std::string::npos);
  EXPECT_NE(
      json.value().find("\"transparentTransmissionFeedbackRefreshCount\": 1"),
      std::string::npos);
  EXPECT_NE(json.value().find("\"meshDraws\": 5"), std::string::npos);

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

TEST(NuriSnapshotTestingTest, ReportReaderRejectsWrongKindAndUnknownFields) {
  const std::filesystem::path wrongKind =
      makeTempPath("snapshot_report_wrong_kind", ".json");
  writeFile(wrongKind,
            R"json({"schemaVersion":1,"kind":"nuri.other.report"})json");
  auto wrong = readSnapshotReportFile(wrongKind);
  EXPECT_TRUE(wrong.hasError());
  EXPECT_NE(wrong.error().find("kind"), std::string::npos);

  const std::filesystem::path unknownField =
      makeTempPath("snapshot_report_unknown_field", ".json");
  writeFile(
      unknownField,
      R"json({"schemaVersion":1,"kind":"nuri.snapshot.report","surprise":1})json");
  auto unknown = readSnapshotReportFile(unknownField);
  EXPECT_TRUE(unknown.hasError());
  EXPECT_NE(unknown.error().find("unknown root key"), std::string::npos);

  std::error_code ec;
  std::filesystem::remove(wrongKind, ec);
  std::filesystem::remove(unknownField, ec);
}

TEST(NuriSnapshotTestingTest, ApprovalRejectsIncompleteRequiredCapture) {
  SnapshotReport report{};
  report.snapshotCase.id = "smoke.incomplete";
  report.snapshotCase.suite = "smoke";
  report.artifacts.caseDir = makeTempPath("snapshot_approval", "");
  report.captures.push_back(SnapshotCaptureReport{
      .target = "final_color",
      .profile = "ldr_color",
      .required = true,
      .available = false,
      .status = "missing_capture_point",
  });

  auto approved = approveSnapshotBaselines(report, "local-nvrhi-visible",
                                           "must reject incomplete source");
  EXPECT_TRUE(approved.hasError());
  EXPECT_NE(approved.error().find("required capture"), std::string::npos);
}

TEST(NuriSnapshotTestingTest,
     BaselinePlanIsDeterministicAndBoundToSourceArtifacts) {
  const std::filesystem::path caseDir =
      makeTempPath("snapshot_approval_plan", "");
  std::filesystem::create_directories(caseDir);
  writeFile(caseDir / "final_color.raw", "first raw payload");
  writeFile(caseDir / "final_color.json", "{\"descriptor\":1}");
  writeFile(caseDir / "final_color_preview.png", "preview payload");

  SnapshotReport report{};
  report.generatedAtUtc = "2026-07-10T00:00:00Z";
  report.environment.commitHash = "abc123";
  report.snapshotCase.id = "smoke.approval-plan-unit";
  report.snapshotCase.suite = "smoke";
  report.artifacts.caseDir = caseDir;
  report.captures.push_back(SnapshotCaptureReport{
      .target = "final_color",
      .profile = "ldr_color",
      .required = true,
      .available = true,
      .capturePointVersion = 1u,
      .kind = "color",
      .colorSpace = "display_sdr",
      .width = 2u,
      .height = 2u,
      .actualHash =
          "sha256:"
          "0000000000000000000000000000000000000000000000000000000000000001",
      .actual = "final_color.raw",
      .actualMetadata = "final_color.json",
      .preview = "final_color_preview.png",
      .status = "captured",
  });

  auto first = planSnapshotBaselines(report, "local-nvrhi-visible",
                                     "reviewed visual change", "unit-test");
  ASSERT_FALSE(first.hasError()) << first.error();
  auto repeated = planSnapshotBaselines(report, "local-nvrhi-visible",
                                        "reviewed visual change", "unit-test");
  ASSERT_FALSE(repeated.hasError()) << repeated.error();
  EXPECT_EQ(first.value().digest, repeated.value().digest);
  EXPECT_EQ(first.value().digest.rfind("sha256:", 0u), 0u);

  auto json = writeSnapshotBaselinePlanJson(first.value());
  ASSERT_FALSE(json.hasError()) << json.error();
  EXPECT_NE(json.value().find("\"kind\": \"nuri.snapshot.baseline_plan\""),
            std::string::npos);
  EXPECT_NE(json.value().find(first.value().digest), std::string::npos);

  writeFile(caseDir / "final_color.raw", "changed raw payload");
  auto changed = planSnapshotBaselines(report, "local-nvrhi-visible",
                                       "reviewed visual change", "unit-test");
  ASSERT_FALSE(changed.hasError()) << changed.error();
  EXPECT_NE(first.value().digest, changed.value().digest);

  std::error_code ec;
  std::filesystem::remove_all(caseDir, ec);
}

TEST(NuriSnapshotTestingTest,
     GovernedApprovalPromotesVerifiedBaselineAndCarriesHistory) {
  const std::filesystem::path tempRoot =
      makeTempPath("snapshot_governed_approval", "");
  const std::filesystem::path caseDir = tempRoot / "run";
  const std::filesystem::path baselineRoot = tempRoot / "baselines";
  std::filesystem::create_directories(caseDir);

  SnapshotReadbackImage image{};
  image.point = makeCapturePoint({});
  image.bytes.resize(16u);
  for (size_t index = 0u; index < image.bytes.size(); ++index) {
    image.bytes[index] = static_cast<std::byte>(index * 7u);
  }
  image.rowStride = 8u;
  image.hash = snapshotHashBytes(image.bytes);
  SnapshotArtifactPaths paths{};
  auto artifacts = writeSnapshotArtifacts(image, caseDir / "final_color", paths,
                                          "ldr_color");
  ASSERT_FALSE(artifacts.hasError()) << artifacts.error();

  SnapshotReport report{};
  report.generatedAtUtc = "2026-07-10T00:00:00Z";
  report.environment.commitHash = "abc123";
  report.snapshotCase.id = "smoke.governed-approval-unit";
  report.snapshotCase.suite = "smoke";
  report.snapshotCase.captures.push_back(
      {.name = "final_color", .profile = "ldr_color", .required = true});
  report.artifacts.caseDir = caseDir;
  report.captures.push_back(SnapshotCaptureReport{
      .target = "final_color",
      .profile = "ldr_color",
      .required = true,
      .available = true,
      .capturePointVersion = 1u,
      .kind = "color",
      .format = "RGBA8_UNORM",
      .colorSpace = "display_sdr",
      .origin = "top_left",
      .width = 2u,
      .height = 2u,
      .actualHash = image.hash,
      .actual = paths.raw.filename(),
      .actualMetadata = paths.metadata.filename(),
      .preview = paths.preview.filename(),
      .status = "captured",
  });

  auto plan =
      planSnapshotBaselines(report, "local-nvrhi-visible",
                            "reviewed unit change", "unit-test", baselineRoot);
  ASSERT_FALSE(plan.hasError()) << plan.error();
  auto approved = approveSnapshotBaselines(
      report, "local-nvrhi-visible", "reviewed unit change",
      plan.value().digest, "unit-test", baselineRoot);
  ASSERT_FALSE(approved.hasError()) << approved.error();
  EXPECT_TRUE(approved.value());
  auto verified = verifySnapshotBaseline(report.snapshotCase,
                                         "local-nvrhi-visible", baselineRoot);
  ASSERT_FALSE(verified.hasError()) << verified.error();

  const std::filesystem::path baselineCase = baselineRoot /
                                             "local-nvrhi-visible" / "smoke" /
                                             "smoke.governed-approval-unit";
  auto firstApprovalDigest =
      nuri::tools::core::sha256File(baselineCase / "approval.json");
  ASSERT_FALSE(firstApprovalDigest.hasError()) << firstApprovalDigest.error();
  auto stale = approveSnapshotBaselines(report, "local-nvrhi-visible",
                                        "reviewed unit change", "sha256:stale",
                                        "unit-test", baselineRoot);
  EXPECT_TRUE(stale.hasError());
  auto unchangedApprovalDigest =
      nuri::tools::core::sha256File(baselineCase / "approval.json");
  ASSERT_FALSE(unchangedApprovalDigest.hasError())
      << unchangedApprovalDigest.error();
  EXPECT_EQ(firstApprovalDigest.value(), unchangedApprovalDigest.value());

  image.bytes[0] = std::byte{0xff};
  image.hash = snapshotHashBytes(image.bytes);
  artifacts = writeSnapshotArtifacts(image, caseDir / "final_color", paths,
                                     "ldr_color");
  ASSERT_FALSE(artifacts.hasError()) << artifacts.error();
  report.captures[0].actualHash = image.hash;
  auto replacementPlan =
      planSnapshotBaselines(report, "local-nvrhi-visible",
                            "reviewed replacement", "unit-test", baselineRoot);
  ASSERT_FALSE(replacementPlan.hasError()) << replacementPlan.error();
  approved = approveSnapshotBaselines(
      report, "local-nvrhi-visible", "reviewed replacement",
      replacementPlan.value().digest, "unit-test", baselineRoot);
  ASSERT_FALSE(approved.hasError()) << approved.error();
  verified = verifySnapshotBaseline(report.snapshotCase, "local-nvrhi-visible",
                                    baselineRoot);
  ASSERT_FALSE(verified.hasError()) << verified.error();
  size_t historyFiles = 0u;
  for (const auto &entry :
       std::filesystem::directory_iterator(baselineCase / "history")) {
    historyFiles += entry.is_regular_file() ? 1u : 0u;
  }
  EXPECT_GE(historyFiles, 5u);

  const auto currentPlan = std::find_if(
      std::filesystem::directory_iterator(baselineCase / "history"),
      std::filesystem::directory_iterator{}, [&](const auto &entry) {
        return entry.is_regular_file() &&
               entry.path().filename().string().ends_with(
                   replacementPlan.value().digest.substr(7u, 16u) +
                   ".plan.json");
      });
  ASSERT_NE(currentPlan, std::filesystem::directory_iterator{});
  std::ifstream currentPlanFile(currentPlan->path(), std::ios::binary);
  const std::string currentPlanText(
      (std::istreambuf_iterator<char>(currentPlanFile)),
      std::istreambuf_iterator<char>());
  currentPlanFile.close();
  writeFile(currentPlan->path(), currentPlanText + "\n");
  verified = verifySnapshotBaseline(report.snapshotCase, "local-nvrhi-visible",
                                    baselineRoot);
  EXPECT_TRUE(verified.hasError());
  EXPECT_NE(verified.error().find("plan"), std::string::npos);
  writeFile(currentPlan->path(), currentPlanText);

  writeFile(baselineCase / paths.raw.filename(), "tampered payload");
  verified = verifySnapshotBaseline(report.snapshotCase, "local-nvrhi-visible",
                                    baselineRoot);
  EXPECT_TRUE(verified.hasError());
  EXPECT_NE(verified.error().find("digest mismatch"), std::string::npos);

  SnapshotCase legacyCase{};
  legacyCase.id = "smoke.legacy-baseline-unit";
  legacyCase.suite = "smoke";
  const std::filesystem::path legacyDir =
      baselineRoot / "local-nvrhi-visible" / "smoke" / legacyCase.id;
  std::filesystem::create_directories(legacyDir);
  writeFile(legacyDir / "final_color.png", "legacy payload");
  auto legacy =
      verifySnapshotBaseline(legacyCase, "local-nvrhi-visible", baselineRoot);
  EXPECT_TRUE(legacy.hasError());
  EXPECT_NE(legacy.error().find("approval.json is missing"), std::string::npos);

  std::error_code ec;
  std::filesystem::remove_all(tempRoot, ec);
}

TEST(NuriSnapshotTestingTest, EmptySuiteIsInvalidAndWritesTruthfulEnvelope) {
  const std::filesystem::path artifactDir =
      makeTempPath("snapshot_empty_suite", "");
  SnapshotRunOptions options{};
  options.artifactDir = artifactDir;
  SnapshotSuiteRunResult result = runSnapshotSuite({}, "missing", options);
  EXPECT_EQ(result.exitCode, SnapshotExitCode::InvalidInput);
  EXPECT_EQ(result.caseResults.size(), 0u);
  EXPECT_TRUE(std::filesystem::exists(result.reportPath));
  std::ifstream report(result.reportPath, std::ios::binary);
  const std::string json((std::istreambuf_iterator<char>(report)),
                         std::istreambuf_iterator<char>());
  EXPECT_NE(json.find("\"selected\": 0"), std::string::npos);
  EXPECT_NE(json.find("\"status\": \"invalid\""), std::string::npos);
  auto envelope = nuri::tools::core::readResultEnvelopeV2(json);
  ASSERT_FALSE(envelope.hasError()) << envelope.error();
  EXPECT_EQ(envelope.value().exitCode, 2);
  EXPECT_EQ(envelope.value().selection.selected, 0u);
  ASSERT_TRUE(envelope.value().environmentFingerprint.has_value());
  EXPECT_EQ(envelope.value().environmentFingerprint->size(), 71u);
  ASSERT_TRUE(envelope.value().workloadFingerprint.has_value());
  EXPECT_EQ(envelope.value().workloadFingerprint->size(), 71u);

  std::error_code ec;
  std::filesystem::remove_all(artifactDir, ec);
}

TEST(NuriSnapshotTestingTest, HtmlEscapesCaptureTextAndShowsMissingState) {
  SnapshotReport report = makeReport();
  report.captures[0].status = "missing_capture_point";
  report.captures[0].statusReason = "target<not>&published";

  auto html = writeSnapshotHtmlReport(report);
  ASSERT_FALSE(html.hasError()) << html.error();
  EXPECT_NE(html.value().find("<!doctype html>"), std::string::npos);
  EXPECT_NE(html.value().find("<html lang=\"en\">"), std::string::npos);
  EXPECT_NE(html.value().find("class=\"skip-link\""), std::string::npos);
  EXPECT_NE(html.value().find("<main id=\"main-content\""), std::string::npos);
  EXPECT_NE(html.value().find("id=\"capture-search\""), std::string::npos);
  EXPECT_NE(html.value().find("aria-live=\"polite\""), std::string::npos);
  EXPECT_NE(html.value().find("<caption>Image comparison and semantic error"),
            std::string::npos);
  EXPECT_NE(html.value().find("prefers-reduced-motion"), std::string::npos);
  EXPECT_NE(html.value().find("case.&lt;final&gt;&amp;color"),
            std::string::npos);
  EXPECT_NE(html.value().find("final&lt;color&gt;&amp;"), std::string::npos);
  EXPECT_NE(html.value().find("target&lt;not&gt;&amp;published"),
            std::string::npos);
  EXPECT_NE(html.value().find("status-missing_capture_point"),
            std::string::npos);
  EXPECT_EQ(html.value().find("target<not>&published"), std::string::npos);
}

} // namespace
