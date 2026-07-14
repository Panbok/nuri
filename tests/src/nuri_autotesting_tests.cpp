#include "tests_pch.h"

#include "nuri/tools/autotest/autotest_assertion.h"
#include "nuri/tools/autotest/autotest_manifest.h"
#include "nuri/tools/autotest/autotest_record.h"
#include "nuri/tools/autotest/autotest_report.h"
#include "nuri/tools/autotest/autotest_runner.h"
#include "nuri/tools/autotest/autotest_timeline.h"
#include "nuri/tools/core/result_envelope_v2.h"
#include "nuri/tools/core/sha256.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <string>

namespace {

using namespace nuri;
using namespace nuri::tools::autotest;

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

AutotestReport makeReport(std::filesystem::path caseDir = {}) {
  AutotestReport report{};
  report.generatedAtUtc = "2026-06-27T00:00:00Z";
  report.status = "pass";
  report.exitCode = AutotestExitCode::Success;
  report.selection = AutotestSelectionSummary{.requested = "case.<auto>&test",
                                              .selected = 1u,
                                              .attempted = 1u,
                                              .completed = 1u,
                                              .passed = 1u};
  report.testCase.id = "case.<auto>&test";
  report.testCase.suite = "smoke";
  report.testCase.scene.contentHash = "case-auto-test-v1";
  report.testCase.resolution = {640u, 360u};
  report.testCase.fixedDeltaSeconds = 1.0 / 60.0;
  report.run.fixedDeltaSeconds = report.testCase.fixedDeltaSeconds;
  report.run.warmupFrames = 2u;
  report.run.endFrame = 4u;
  report.run.renderedFrames = 5u;
  report.run.readoutDrainFrames = 1u;
  report.run.readoutDrainFrameLimit = 4u;
  report.run.readoutDrainElapsedMs = 12u;
  report.run.requestedWindowMode = "visible";
  report.run.resolvedWindowMode = "visible";
  report.run.windowModeSource = "manifest";
  report.environment.gpuBackend = "nvrhi";
  report.environment.gpuBackendSource = "default";
  report.environment.resolvedPresentMode = "immediate";
  report.environment.resolvedWindowMode = "visible";
  report.environment.buildType = "Release";
  report.environment.cmakeToolProfile = "autotest";
  report.environment.commitHash = "deadbeef";
  report.environment.cpuLogicalThreadCount = 8u;
  report.environment.gpuDeviceName = "test gpu";
  report.environment.gpuVendorId = 0x10deu;
  report.environment.gpuDeviceId = 0x2684u;
  report.environment.gpuDriverVersion = "test-driver";
  report.artifacts.caseDir = std::move(caseDir);
  report.artifacts.caseHtml = report.artifacts.caseDir / "report.html";
  AutotestCheckpointReport checkpoint{};
  checkpoint.id = "end<frame>&";
  checkpoint.frame = 4u;
  checkpoint.measurements["renderer.frame_index"] = 4.0;
  nuri::tools::snapshot::SnapshotCaptureReport snapshot{};
  snapshot.target = "final<color>&";
  snapshot.artifactStem = "final_color";
  snapshot.profile = "ldr_color";
  snapshot.required = true;
  snapshot.available = true;
  snapshot.capturePointVersion = 3u;
  snapshot.captureFrameIndex = 4u;
  snapshot.kind = "color";
  snapshot.origin = "top_left";
  snapshot.mip = 1u;
  snapshot.layer = 2u;
  snapshot.preview = "checkpoints/end/final_color_preview.png";
  snapshot.status = "captured";
  snapshot.statusReason = "capture<ok>&";
  snapshot.semanticMetrics.unit = "display_linear";
  snapshot.semanticMetrics.maxError = 0.125;
  checkpoint.captures.push_back(AutotestCaptureReport{
      .checkpointId = checkpoint.id,
      .checkpointFrame = checkpoint.frame,
      .target = snapshot.target,
      .profile = snapshot.profile,
      .required = true,
      .compare = false,
      .snapshot = std::move(snapshot),
  });
  report.testCase.checkpoints.push_back(AutotestCheckpoint{
      .id = checkpoint.id,
      .frame = checkpoint.frame,
      .captures = {AutotestCaptureTarget{.target = "final<color>&",
                                         .profile = "ldr_color",
                                         .required = true,
                                         .compare = false}},
  });
  checkpoint.assertions.push_back(AutotestAssertionResult{
      .id = "frame_index",
      .metric = "renderer.frame_index",
      .statistic = {},
      .status = AutotestAssertionStatus::Pass,
      .statusReason = "passed",
      .actual = 4.0,
      .hasActual = true,
      .sampleCount = 1u,
      .expectedSampleCount = 1u,
  });
  AutotestReadoutReport readout{};
  readout.checkpointId = checkpoint.id;
  readout.id = "shadow<probe>&";
  readout.type = "shadowInspect";
  readout.requestId = 42u;
  readout.requestFrame = 4u;
  readout.resultFrame = 5u;
  readout.required = true;
  readout.status = "pass";
  readout.statusReason = "readout<ok>&";
  readout.values["valid"] = 1.0;
  readout.values["cascadeIndex"] = 1.0;
  readout.assertions.push_back(AutotestAssertionResult{
      .id = "shadow_valid",
      .metric = "valid",
      .status = AutotestAssertionStatus::Pass,
      .statusReason = "passed",
      .actual = 1.0,
      .hasActual = true,
      .sampleCount = 1u,
      .expectedSampleCount = 1u,
  });
  checkpoint.readouts.push_back(std::move(readout));
  checkpoint.warnings.push_back("checkpoint warning");
  checkpoint.errors.push_back("checkpoint error");
  report.checkpoints.push_back(std::move(checkpoint));
  AutotestMetricWindowReport window{};
  window.id = "steady<pan>&";
  window.startFrame = 2u;
  window.endFrame = 4u;
  window.assertions.push_back(AutotestAssertionResult{
      .id = "p95_window",
      .metric = "gpu.scopes.taa_resolve_ms",
      .statistic = "p95",
      .status = AutotestAssertionStatus::Warn,
      .statusReason = "threshold_failed",
      .actual = 0.9,
      .hasActual = true,
      .sampleCount = 3u,
      .expectedSampleCount = 3u,
  });
  window.warnings.push_back("window warning");
  window.errors.push_back("window error");
  report.metricWindows.push_back(std::move(window));
  report.frames.push_back(AutotestFrameReport{
      .frameIndex = 4u,
      .measurements = {{"renderer.frame_index", 4.0}},
  });
  report.unavailableMetrics.push_back("gpu.scopes.missing_ms");
  report.warnings.push_back("report warning");
  report.errors.push_back("report error");
  return report;
}

AutotestReport makeApprovalCandidate(const std::filesystem::path &caseDir) {
  AutotestReport report{};
  report.generatedAtUtc = "2026-07-10T00:00:00Z";
  report.status = "pass";
  report.exitCode = AutotestExitCode::Success;
  report.selection = AutotestSelectionSummary{.requested = "approval_case",
                                              .selected = 1u,
                                              .attempted = 1u,
                                              .completed = 1u,
                                              .passed = 1u};
  report.testCase.id = "approval_case";
  report.testCase.suite = "smoke";
  report.testCase.scene.contentHash = "approval-case-v1";
  report.testCase.resolution = {2u, 2u};
  report.testCase.fixedDeltaSeconds = 1.0 / 60.0;
  report.testCase.endFrame = 4u;
  report.run.fixedDeltaSeconds = report.testCase.fixedDeltaSeconds;
  report.run.endFrame = report.testCase.endFrame;
  report.run.renderedFrames = 5u;
  report.environment.gpuBackend = "nvrhi";
  report.environment.gpuBackendSource = "default";
  report.environment.resolvedPresentMode = "immediate";
  report.environment.resolvedWindowMode = "visible";
  report.environment.buildType = "Release";
  report.environment.cmakeToolProfile = "autotest";
  report.environment.commitHash = "candidate-commit";
  report.artifacts.caseDir = caseDir;
  report.testCase.checkpoints.push_back(AutotestCheckpoint{
      .id = "end",
      .frame = 4u,
      .captures = {AutotestCaptureTarget{.target = "final_color",
                                         .profile = "ldr_color",
                                         .required = true,
                                         .compare = false}},
  });
  nuri::tools::snapshot::SnapshotCaptureReport snapshot{};
  snapshot.target = "final_color";
  snapshot.artifactStem = "final_color";
  snapshot.profile = "ldr_color";
  snapshot.required = true;
  snapshot.available = true;
  snapshot.capturePointVersion = 1u;
  snapshot.captureFrameIndex = 4u;
  snapshot.kind = "color";
  snapshot.format = "RGBA8_UNORM";
  snapshot.width = 2u;
  snapshot.height = 2u;
  snapshot.origin = "top_left";
  snapshot.colorSpace = "srgb";
  snapshot.actual = "checkpoints/end_frame_4/final_color.png";
  snapshot.actualMetadata = "checkpoints/end_frame_4/final_color.json";
  snapshot.preview = "checkpoints/end_frame_4/final_color_preview.png";
  snapshot.status = "captured";
  snapshot.statusReason = "captured";
  report.checkpoints.push_back(AutotestCheckpointReport{
      .id = "end",
      .frame = 4u,
      .captures = {AutotestCaptureReport{.checkpointId = "end",
                                         .checkpointFrame = 4u,
                                         .target = "final_color",
                                         .profile = "ldr_color",
                                         .required = true,
                                         .compare = false,
                                         .snapshot = std::move(snapshot)}},
  });
  return report;
}

void writeApprovalCandidateCapture(AutotestReport &report,
                                   std::string_view payload) {
  auto &snapshot = report.checkpoints[0].captures[0].snapshot;
  const std::filesystem::path raw = report.artifacts.caseDir / snapshot.actual;
  const std::filesystem::path metadata =
      report.artifacts.caseDir / snapshot.actualMetadata;
  const std::filesystem::path preview =
      report.artifacts.caseDir / snapshot.preview;
  std::filesystem::create_directories(raw.parent_path());
  writeFile(raw, payload);
  writeFile(preview, "synthetic-preview");
  auto digest = nuri::tools::core::sha256File(raw);
  ASSERT_FALSE(digest.hasError()) << digest.error();
  snapshot.actualHash = "sha256:" + digest.value();
  writeFile(metadata, "{\n  \"target\": \"final_color\",\n"
                      "  \"capturePointVersion\": 1,\n"
                      "  \"kind\": \"color\",\n"
                      "  \"format\": 1,\n"
                      "  \"width\": 2,\n  \"height\": 2,\n"
                      "  \"mip\": 0,\n  \"layer\": 0,\n"
                      "  \"origin\": \"top_left\",\n"
                      "  \"colorSpace\": \"srgb\",\n"
                      "  \"profile\": \"ldr_color\",\n"
                      "  \"payload\": \"final_color.png\",\n"
                      "  \"hash\": \"" +
                          snapshot.actualHash + "\"\n}\n");
  auto metadataWritten = writeAutotestRecordMetadataFile(
      report, report.artifacts.caseDir, "local-nvrhi-visible");
  ASSERT_FALSE(metadataWritten.hasError()) << metadataWritten.error();
}

std::map<std::string, std::string>
readTreeFiles(const std::filesystem::path &root) {
  std::map<std::string, std::string> files;
  if (!std::filesystem::is_directory(root)) {
    return files;
  }
  for (const auto &entry :
       std::filesystem::recursive_directory_iterator(root)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    std::ifstream file(entry.path(), std::ios::binary);
    files.emplace(
        std::filesystem::relative(entry.path(), root).generic_string(),
        std::string((std::istreambuf_iterator<char>(file)),
                    std::istreambuf_iterator<char>()));
  }
  return files;
}

TEST(NuriAutotestingTest, ManifestRejectsUnknownKeys) {
  const std::filesystem::path path = makeTempPath("autotest_manifest", ".json");
  writeFile(path,
            R"json({
              "schemaVersion": 1,
              "id": "bad.case",
              "suite": "bad",
              "unexpected": true
            })json");

  auto loaded = loadAutotestCaseManifest(path);
  EXPECT_TRUE(loaded.hasError());
  EXPECT_NE(loaded.error().find("unknown key"), std::string::npos);

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

TEST(NuriAutotestingTest, ManifestAppliesShadowPresetBeforeExplicitOverrides) {
  const std::filesystem::path path =
      makeTempPath("autotest_shadow_overrides", ".json");
  const std::string manifest = R"json({
    "schemaVersion": 1,
    "id": "shadow.overrides",
    "suite": "shadow",
    "endFrame": 4,
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
    }}
  })json";
  writeFile(path, manifest);

  auto loaded = loadAutotestCaseManifest(path);
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
  auto invalid = loadAutotestCaseManifest(path);
  EXPECT_TRUE(invalid.hasError());
  EXPECT_NE(invalid.error().find("depthFormat"), std::string::npos);

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

TEST(NuriAutotestingTest, ManifestRejectsUnsafeCaseId) {
  const std::filesystem::path path =
      makeTempPath("autotest_unsafe_case_id", ".json");
  writeFile(path,
            R"json({
              "schemaVersion": 1,
              "id": "../escaped",
              "suite": "smoke"
            })json");

  auto loaded = loadAutotestCaseManifest(path);
  EXPECT_TRUE(loaded.hasError());
  EXPECT_NE(loaded.error().find("id"), std::string::npos);

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

TEST(NuriAutotestingTest, CaseValidationRejectsAmbiguousScenarioIds) {
  AutotestCase testCase{};
  testCase.id = "ambiguous.case";
  testCase.suite = "smoke";
  AutotestCheckpoint checkpoint{};
  checkpoint.id = "end";
  checkpoint.frame = testCase.endFrame;
  checkpoint.captures = {
      AutotestCaptureTarget{.target = "final_color", .profile = "ldr_color"},
      AutotestCaptureTarget{.target = "final_color", .profile = "ldr_color"},
  };
  testCase.checkpoints = {checkpoint};
  auto validated = validateAutotestCase(testCase);
  EXPECT_TRUE(validated.hasError());
  EXPECT_NE(validated.error().find("duplicate capture target"),
            std::string::npos);

  testCase.checkpoints[0].captures.resize(1u);
  AutotestMetricAssertion assertion{};
  assertion.id = "visible";
  assertion.metric = "renderer.visible";
  assertion.hasMin = true;
  testCase.checkpoints[0].assertions = {assertion, assertion};
  validated = validateAutotestCase(testCase);
  EXPECT_TRUE(validated.hasError());
  EXPECT_NE(validated.error().find("duplicate assertion id"),
            std::string::npos);
}

TEST(NuriAutotestingTest, CaseValidationRejectsConflictingReadoutChannels) {
  AutotestCase testCase{};
  testCase.id = "readout.conflict";
  testCase.suite = "smoke";
  testCase.checkpoints = {
      AutotestCheckpoint{
          .id = "left",
          .frame = 2u,
          .readouts = {AutotestReadoutRequest{.id = "left_probe",
                                              .type = "opaquePick"}},
      },
      AutotestCheckpoint{
          .id = "right",
          .frame = 2u,
          .readouts = {AutotestReadoutRequest{.id = "right_probe",
                                              .type = "opaquePick"}},
      },
  };

  auto validated = validateAutotestCase(testCase);
  EXPECT_TRUE(validated.hasError());
  EXPECT_NE(validated.error().find("more than once"), std::string::npos);
}

TEST(NuriAutotestingTest, CaseValidationRejectsInvalidRangesAndCameraPaths) {
  AutotestCase testCase{};
  testCase.id = "invalid.ranges";
  testCase.suite = "smoke";
  testCase.resolution = {0u, 360u};
  auto validated = validateAutotestCase(testCase);
  EXPECT_TRUE(validated.hasError());

  testCase.resolution = {640u, 360u};
  testCase.camera.nearPlane = 2.0f;
  testCase.camera.farPlane = 1.0f;
  validated = validateAutotestCase(testCase);
  EXPECT_TRUE(validated.hasError());

  testCase.camera.nearPlane = 0.05f;
  testCase.camera.farPlane = 100.0f;
  AutotestCameraPath path{};
  path.id = "pan";
  path.startFrame = 0u;
  path.endFrame = 2u;
  path.keyframes = {
      AutotestCameraKeyframe{.frame = 0u},
      AutotestCameraKeyframe{.frame = 0u},
  };
  testCase.timeline.cameraPaths = {path};
  validated = validateAutotestCase(testCase);
  EXPECT_TRUE(validated.hasError());
  EXPECT_NE(validated.error().find("duplicate keyframe frame"),
            std::string::npos);
}

TEST(NuriAutotestingTest, TimelineInterpolatesCameraPaths) {
  const std::filesystem::path panPath =
      std::filesystem::path(PROJECT_SOURCE_DIR) / "tools" / "cases" /
      "autotests" / "smoke" / "camera_pan.json";
  auto pan = loadAutotestCaseManifest(panPath);
  ASSERT_FALSE(pan.hasError()) << pan.error();

  auto camera = evaluateAutotestCameraAtFrame(pan.value(), 4u);
  ASSERT_FALSE(camera.hasError()) << camera.error();
  EXPECT_NEAR(camera.value().position.x, 0.0f, 1.0e-5f);
  EXPECT_TRUE(camera.value().hasTarget);

  auto plan = compileAutotestTimeline(pan.value());
  ASSERT_FALSE(plan.hasError()) << plan.error();
  ASSERT_EQ(plan.value().size(), 9u);
  EXPECT_TRUE(plan.value()[0].resetTemporalHistory);
  EXPECT_EQ(plan.value()[2].checkpoints.size(), 1u);
  EXPECT_EQ(plan.value()[8].checkpoints.size(), 1u);
}

TEST(NuriAutotestingTest, TimelineSetSettingsPatchesPersist) {
  const std::filesystem::path path =
      makeTempPath("autotest_set_settings", ".json");
  writeFile(path,
            R"json({
              "schemaVersion": 1,
              "id": "settings.case",
              "suite": "smoke",
              "endFrame": 3,
              "settings": {
                "antiAliasing": {
                  "mode": "None"
                },
                "ambientOcclusion": {
                  "mode": "Disabled"
                }
              },
              "timeline": {
                "events": [
                  {
                    "frame": 2,
                    "type": "setSettings",
                    "settings": {
                      "antiAliasing": {
                        "mode": "TAA",
                        "qualityPreset": "Quality"
                      }
                    }
                  }
                ]
              }
            })json");

  auto loaded = loadAutotestCaseManifest(path);
  ASSERT_FALSE(loaded.hasError()) << loaded.error();
  ASSERT_EQ(loaded.value().timeline.events.size(), 1u);
  EXPECT_EQ(loaded.value().timeline.events[0].type, "setSettings");
  EXPECT_TRUE(loaded.value().timeline.events[0].hasSettings);

  auto plan = compileAutotestTimeline(loaded.value());
  ASSERT_FALSE(plan.hasError()) << plan.error();
  ASSERT_EQ(plan.value().size(), 4u);
  EXPECT_EQ(plan.value()[1].settings.antiAliasing.mode, AntiAliasingMode::None);
  EXPECT_EQ(plan.value()[2].settings.antiAliasing.mode, AntiAliasingMode::TAA);
  EXPECT_EQ(plan.value()[3].settings.antiAliasing.mode, AntiAliasingMode::TAA);
  EXPECT_EQ(plan.value()[3].settings.antiAliasing.qualityPreset,
            TemporalAAQualityPreset::Quality);
  EXPECT_EQ(plan.value()[3].settings.ambientOcclusion.mode,
            AmbientOcclusionMode::Disabled);

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

TEST(NuriAutotestingTest, ReadoutDrainFramesPreserveFinalPlannedState) {
  AutotestCase testCase{};
  testCase.id = "drain.state";
  testCase.suite = "smoke";
  testCase.endFrame = 2u;
  testCase.settings.antiAliasing.mode = AntiAliasingMode::None;
  AutotestTimelineEvent settingsEvent{};
  settingsEvent.frame = 2u;
  settingsEvent.type = "setSettings";
  settingsEvent.hasSettings = true;
  settingsEvent.settings = testCase.settings;
  settingsEvent.settings.antiAliasing.mode = AntiAliasingMode::TAA;
  settingsEvent.settings.ambientOcclusion.mode = AmbientOcclusionMode::GTAO;
  testCase.timeline.events.push_back(settingsEvent);
  testCase.checkpoints = {AutotestCheckpoint{
      .id = "end",
      .frame = 2u,
      .readouts = {AutotestReadoutRequest{.id = "probe", .type = "opaquePick"}},
  }};

  auto plan = compileAutotestTimeline(testCase);
  ASSERT_FALSE(plan.hasError()) << plan.error();
  ASSERT_EQ(plan.value().size(),
            testCase.endFrame + 1u + kAutotestReadoutDrainFrameLimit);
  const AutotestFramePlan &lastPlanned = plan.value()[testCase.endFrame];
  const AutotestFramePlan &firstDrain = plan.value()[testCase.endFrame + 1u];
  EXPECT_TRUE(firstDrain.drainOnly);
  EXPECT_TRUE(firstDrain.checkpoints.empty());
  EXPECT_EQ(firstDrain.settings.antiAliasing.mode,
            lastPlanned.settings.antiAliasing.mode);
  EXPECT_EQ(firstDrain.settings.ambientOcclusion.mode,
            lastPlanned.settings.ambientOcclusion.mode);
  EXPECT_EQ(firstDrain.camera.position, lastPlanned.camera.position);
}

TEST(NuriAutotestingTest, AssertionsDistinguishZeroFromUnavailable) {
  std::map<std::string, double> measurements;
  measurements["renderer.shadow.total_draws"] = 0.0;

  AutotestMetricAssertion zero{};
  zero.id = "zero";
  zero.metric = "renderer.shadow.total_draws";
  zero.hasEquals = true;
  zero.equals = 0.0;
  auto result = evaluateAutotestAssertion(zero, measurements);
  EXPECT_EQ(result.status, AutotestAssertionStatus::Pass);
  EXPECT_TRUE(result.hasActual);

  AutotestMetricAssertion missing{};
  missing.id = "missing";
  missing.metric = "gpu.scopes.taa_resolve_ms";
  missing.hasMax = true;
  missing.max = 1.0;
  result = evaluateAutotestAssertion(missing, measurements);
  EXPECT_EQ(result.status, AutotestAssertionStatus::Invalid);

  missing.optional = true;
  result = evaluateAutotestAssertion(missing, measurements);
  EXPECT_EQ(result.status, AutotestAssertionStatus::Unavailable);
}

TEST(NuriAutotestingTest, BaselinePlanIsSha256BoundAndMutationFree) {
  const std::filesystem::path root = makeTempPath("autotest_baseline_plan", "");
  const std::filesystem::path candidateCase =
      root / "candidates" / "cases" / "approval_case";
  const std::filesystem::path baselineRoot = root / "baselines" / "render";
  AutotestReport report = makeApprovalCandidate(candidateCase);
  writeApprovalCandidateCapture(report, "candidate-v1");
  const AutotestBaselineApprovalOptions options{.baselineRoot = baselineRoot};

  auto first =
      planAutotestBaselines(report.testCase, report, "local-nvrhi-visible",
                            "reviewed change", "test actor", options);
  ASSERT_FALSE(first.hasError()) << first.error();
  EXPECT_EQ(first.value().digest.rfind("sha256:", 0u), 0u);
  EXPECT_FALSE(first.value().entries.empty());
  auto planJson = writeAutotestBaselinePlanJson(first.value());
  ASSERT_FALSE(planJson.hasError()) << planJson.error();
  EXPECT_NE(planJson.value().find("nuri.autotest.baseline_plan"),
            std::string::npos);
  EXPECT_NE(planJson.value().find(first.value().digest), std::string::npos);
  EXPECT_FALSE(std::filesystem::exists(baselineRoot));

  auto repeated =
      planAutotestBaselines(report.testCase, report, "local-nvrhi-visible",
                            "reviewed change", "test actor", options);
  ASSERT_FALSE(repeated.hasError()) << repeated.error();
  EXPECT_EQ(repeated.value().digest, first.value().digest);

  auto changedActor =
      planAutotestBaselines(report.testCase, report, "local-nvrhi-visible",
                            "reviewed change", "different actor", options);
  ASSERT_FALSE(changedActor.hasError()) << changedActor.error();
  EXPECT_NE(changedActor.value().digest, first.value().digest);

  writeApprovalCandidateCapture(report, "candidate-v2");
  auto changed =
      planAutotestBaselines(report.testCase, report, "local-nvrhi-visible",
                            "reviewed change", "test actor", options);
  ASSERT_FALSE(changed.hasError()) << changed.error();
  EXPECT_NE(changed.value().digest, first.value().digest);
  EXPECT_FALSE(std::filesystem::exists(baselineRoot));

  std::error_code ec;
  std::filesystem::remove_all(root, ec);
}

TEST(NuriAutotestingTest, BaselinePlanRejectsEscapingCandidateArtifact) {
  const std::filesystem::path root =
      makeTempPath("autotest_escaping_candidate", "");
  const std::filesystem::path candidateCase =
      root / "candidates" / "cases" / "approval_case";
  const std::filesystem::path baselineRoot = root / "baselines" / "render";
  AutotestReport report = makeApprovalCandidate(candidateCase);
  writeApprovalCandidateCapture(report, "candidate-v1");
  report.checkpoints[0].captures[0].snapshot.actual = "../outside.png";

  auto plan = planAutotestBaselines(
      report.testCase, report, "local-nvrhi-visible", "reviewed change",
      "test actor",
      AutotestBaselineApprovalOptions{.baselineRoot = baselineRoot});

  EXPECT_TRUE(plan.hasError());
  EXPECT_FALSE(std::filesystem::exists(baselineRoot));

  std::error_code ec;
  std::filesystem::remove_all(root, ec);
}

TEST(NuriAutotestingTest, BaselinePromotionCarriesHistoryAndRollsBackFailure) {
  const std::filesystem::path root =
      makeTempPath("autotest_baseline_promotion", "");
  const std::filesystem::path candidateCase =
      root / "candidates" / "cases" / "approval_case";
  const std::filesystem::path baselineRoot = root / "baselines" / "render";
  const std::filesystem::path baselineCase =
      baselineRoot / "local-nvrhi-visible" / "autotests" / "smoke" /
      "approval_case";
  AutotestReport report = makeApprovalCandidate(candidateCase);
  writeApprovalCandidateCapture(report, "candidate-v1");
  const AutotestBaselineApprovalOptions options{.baselineRoot = baselineRoot};

  auto firstPlan =
      planAutotestBaselines(report.testCase, report, "local-nvrhi-visible",
                            "first approval", "test actor", options);
  ASSERT_FALSE(firstPlan.hasError()) << firstPlan.error();
  auto approved = approveAutotestBaselines(
      report.testCase, report, "local-nvrhi-visible", "first approval",
      firstPlan.value().digest, "test actor", options);
  ASSERT_FALSE(approved.hasError()) << approved.error();
  const auto firstHistory = readTreeFiles(baselineCase / "history");
  ASSERT_FALSE(firstHistory.empty());

  writeApprovalCandidateCapture(report, "candidate-v2");
  auto secondPlan =
      planAutotestBaselines(report.testCase, report, "local-nvrhi-visible",
                            "second approval", "test actor", options);
  ASSERT_FALSE(secondPlan.hasError()) << secondPlan.error();
  EXPECT_GT(secondPlan.value().historyFileCount, 0u);
  EXPECT_EQ(secondPlan.value().historyDigest.rfind("sha256:", 0u), 0u);
  approved = approveAutotestBaselines(
      report.testCase, report, "local-nvrhi-visible", "second approval",
      secondPlan.value().digest, "test actor", options);
  ASSERT_FALSE(approved.hasError()) << approved.error();
  const auto secondHistory = readTreeFiles(baselineCase / "history");
  for (const auto &[path, contents] : firstHistory) {
    ASSERT_EQ(secondHistory.count(path), 1u) << path;
    EXPECT_EQ(secondHistory.at(path), contents) << path;
  }
  const auto beforeFailure = readTreeFiles(baselineCase);

  writeApprovalCandidateCapture(report, "candidate-v3");
  const AutotestBaselineApprovalOptions failingOptions{
      .baselineRoot = baselineRoot, .failAfterBackupForTesting = true};
  auto failingPlan =
      planAutotestBaselines(report.testCase, report, "local-nvrhi-visible",
                            "failing approval", "test actor", failingOptions);
  ASSERT_FALSE(failingPlan.hasError()) << failingPlan.error();
  approved = approveAutotestBaselines(
      report.testCase, report, "local-nvrhi-visible", "failing approval",
      failingPlan.value().digest, "test actor", failingOptions);
  EXPECT_TRUE(approved.hasError());
  EXPECT_EQ(readTreeFiles(baselineCase), beforeFailure);
  for (const auto &entry :
       std::filesystem::directory_iterator(baselineCase.parent_path())) {
    const std::string name = entry.path().filename().string();
    EXPECT_EQ(name.find(".stage-"), std::string::npos);
    EXPECT_EQ(name.find(".backup-"), std::string::npos);
  }

  std::error_code ec;
  std::filesystem::remove_all(root, ec);
}

TEST(NuriAutotestingTest,
     BaselineInspectAndVerifyRejectsTreeAndHistoryTampering) {
  const std::filesystem::path root =
      makeTempPath("autotest_baseline_verify", "");
  const std::filesystem::path candidateCase =
      root / "candidates" / "cases" / "approval_case";
  const std::filesystem::path baselineRoot = root / "baselines" / "render";
  const std::filesystem::path baselineCase =
      baselineRoot / "local-nvrhi-visible" / "autotests" / "smoke" /
      "approval_case";
  AutotestReport report = makeApprovalCandidate(candidateCase);
  writeApprovalCandidateCapture(report, "candidate-v1");

  auto missing = inspectAutotestBaseline(report.testCase, "local-nvrhi-visible",
                                         baselineRoot);
  ASSERT_FALSE(missing.hasError()) << missing.error();
  EXPECT_NE(missing.value().find("\"exists\": false"), std::string::npos);
  EXPECT_NE(missing.value().find("\"verified\": false"), std::string::npos);
  EXPECT_FALSE(std::filesystem::exists(baselineRoot));

  const AutotestBaselineApprovalOptions options{.baselineRoot = baselineRoot};
  auto plan =
      planAutotestBaselines(report.testCase, report, "local-nvrhi-visible",
                            "verified approval", "test actor", options);
  ASSERT_FALSE(plan.hasError()) << plan.error();
  auto approved = approveAutotestBaselines(
      report.testCase, report, "local-nvrhi-visible", "verified approval",
      plan.value().digest, "test actor", options);
  ASSERT_FALSE(approved.hasError()) << approved.error();
  auto verified = verifyAutotestBaseline(report.testCase, "local-nvrhi-visible",
                                         baselineRoot);
  ASSERT_FALSE(verified.hasError()) << verified.error();
  auto inspection = inspectAutotestBaseline(
      report.testCase, "local-nvrhi-visible", baselineRoot);
  ASSERT_FALSE(inspection.hasError()) << inspection.error();
  EXPECT_NE(inspection.value().find(
                "\"kind\": \"nuri.autotest.baseline_inspection\""),
            std::string::npos);
  EXPECT_NE(inspection.value().find("\"verified\": true"), std::string::npos);
  EXPECT_NE(inspection.value().find(plan.value().digest), std::string::npos);

  writeFile(baselineCase / "unexpected.txt", "unreviewed");
  verified = verifyAutotestBaseline(report.testCase, "local-nvrhi-visible",
                                    baselineRoot);
  EXPECT_TRUE(verified.hasError());
  EXPECT_NE(verified.error().find("unreviewed"), std::string::npos);
  std::filesystem::remove(baselineCase / "unexpected.txt");

  const std::filesystem::path preview =
      baselineCase / "checkpoints/end_frame_4/final_color_preview.png";
  std::filesystem::remove(preview);
  verified = verifyAutotestBaseline(report.testCase, "local-nvrhi-visible",
                                    baselineRoot);
  EXPECT_TRUE(verified.hasError());
  EXPECT_NE(verified.error().find("missing"), std::string::npos);
  writeFile(preview, "synthetic-preview");

  const std::filesystem::path payload =
      baselineCase / "checkpoints/end_frame_4/final_color.png";
  writeFile(payload, "tampered");
  verified = verifyAutotestBaseline(report.testCase, "local-nvrhi-visible",
                                    baselineRoot);
  EXPECT_TRUE(verified.hasError());
  EXPECT_NE(verified.error().find("digest mismatch"), std::string::npos);
  writeFile(payload, "candidate-v1");

  const auto history = readTreeFiles(baselineCase / "history");
  const auto planHistory =
      std::find_if(history.begin(), history.end(), [](const auto &entry) {
        return entry.first.ends_with("-plan.json");
      });
  ASSERT_NE(planHistory, history.end());
  writeFile(baselineCase / "history" / planHistory->first,
            planHistory->second + "\n");
  verified = verifyAutotestBaseline(report.testCase, "local-nvrhi-visible",
                                    baselineRoot);
  EXPECT_TRUE(verified.hasError());
  EXPECT_NE(verified.error().find("reviewed plan history is invalid"),
            std::string::npos);
  writeFile(baselineCase / "history" / planHistory->first, planHistory->second);
  verified = verifyAutotestBaseline(report.testCase, "local-nvrhi-visible",
                                    baselineRoot);
  ASSERT_FALSE(verified.hasError()) << verified.error();

  const std::filesystem::path outside = root / "outside.txt";
  const std::filesystem::path link = baselineCase / "unexpected-link";
  writeFile(outside, "outside");
  std::error_code linkError;
  std::filesystem::create_symlink(outside, link, linkError);
  if (!linkError) {
    verified = verifyAutotestBaseline(report.testCase, "local-nvrhi-visible",
                                      baselineRoot);
    EXPECT_TRUE(verified.hasError());
    EXPECT_NE(verified.error().find("link or escape"), std::string::npos);
  }

  std::error_code ec;
  std::filesystem::remove_all(root, ec);
}

TEST(NuriAutotestingTest, MetricWindowsParseAndEvaluateStatistics) {
  const std::filesystem::path path =
      makeTempPath("autotest_metric_windows", ".json");
  writeFile(path,
            R"json({
              "schemaVersion": 1,
              "id": "metric.window.case",
              "suite": "smoke",
              "endFrame": 4,
              "metricWindows": [
                {
                  "id": "steady",
                  "frames": [1, 4],
                  "assertions": [
                    {
                      "id": "frame_max",
                      "metric": "renderer.frame_index",
                      "max": 4
                    },
                    {
                      "id": "taa_p95",
                      "metric": "gpu.scopes.taa_resolve_ms",
                      "p95Max": 3.9,
                      "severity": "warn"
                    }
                  ]
                }
              ]
            })json");

  auto loaded = loadAutotestCaseManifest(path);
  ASSERT_FALSE(loaded.hasError()) << loaded.error();
  ASSERT_EQ(loaded.value().metricWindows.size(), 1u);
  EXPECT_EQ(loaded.value().metricWindows[0].id, "steady");
  ASSERT_EQ(loaded.value().metricWindows[0].assertions.size(), 2u);
  EXPECT_TRUE(loaded.value().metricWindows[0].assertions[1].hasP95Max);

  std::map<uint64_t, std::map<std::string, double>> frames;
  for (uint64_t frame = 1u; frame <= 4u; ++frame) {
    frames[frame]["renderer.frame_index"] = static_cast<double>(frame);
    frames[frame]["gpu.scopes.taa_resolve_ms"] = static_cast<double>(frame);
  }
  auto results = evaluateAutotestMetricWindowAssertions(
      loaded.value().metricWindows[0], frames);
  ASSERT_EQ(results.size(), 2u);
  EXPECT_EQ(results[0].status, AutotestAssertionStatus::Pass);
  EXPECT_EQ(results[0].statistic, "max");
  EXPECT_DOUBLE_EQ(results[0].actual, 4.0);
  EXPECT_EQ(results[0].sampleCount, 4u);
  EXPECT_EQ(results[1].status, AutotestAssertionStatus::Pass);
  EXPECT_EQ(results[1].statistic, "p95");
  EXPECT_NEAR(results[1].actual, 3.85, 1.0e-9);

  AutotestMetricWindowAssertion missing{};
  missing.id = "missing";
  missing.metric = "renderer.not_present";
  missing.hasMax = true;
  missing.max = 1.0;
  auto missingResult =
      evaluateAutotestMetricWindowAssertion(missing, frames, 1u, 4u);
  EXPECT_EQ(missingResult.status, AutotestAssertionStatus::Invalid);
  missing.optional = true;
  missingResult =
      evaluateAutotestMetricWindowAssertion(missing, frames, 1u, 4u);
  EXPECT_EQ(missingResult.status, AutotestAssertionStatus::Unavailable);

  AutotestMetricWindowAssertion warn{};
  warn.id = "warn";
  warn.metric = "gpu.scopes.taa_resolve_ms";
  warn.severity = "warn";
  warn.hasP95Max = true;
  warn.p95Max = 3.0;
  auto warnResult = evaluateAutotestMetricWindowAssertion(warn, frames, 1u, 4u);
  EXPECT_EQ(warnResult.status, AutotestAssertionStatus::Warn);
  EXPECT_EQ(warnResult.statistic, "p95");

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

TEST(NuriAutotestingTest, RequiredMetricWindowRejectsPartialSamples) {
  AutotestMetricWindowAssertion assertion{};
  assertion.id = "complete";
  assertion.metric = "renderer.frame_index";
  assertion.hasMax = true;
  assertion.max = 4.0;
  std::map<uint64_t, std::map<std::string, double>> frames;
  frames[1u][assertion.metric] = 1.0;
  frames[2u][assertion.metric] = 2.0;
  frames[4u][assertion.metric] = 4.0;

  AutotestAssertionResult result =
      evaluateAutotestMetricWindowAssertion(assertion, frames, 1u, 4u);
  EXPECT_EQ(result.status, AutotestAssertionStatus::Invalid);
  EXPECT_EQ(result.statusReason, "required_metric_window_incomplete");
  EXPECT_EQ(result.sampleCount, 3u);
  EXPECT_EQ(result.expectedSampleCount, 4u);

  assertion.optional = true;
  result = evaluateAutotestMetricWindowAssertion(assertion, frames, 1u, 4u);
  EXPECT_EQ(result.status, AutotestAssertionStatus::Unavailable);
  EXPECT_EQ(result.statusReason, "optional_metric_window_incomplete");
}

TEST(NuriAutotestingTest, GpuTimingMapsEachScopeToItsOwnFrame) {
  GpuTimingReport report{};
  report.availableScopeMask =
      gpuTimingScopeToBit(GpuTimingScope::Shadow) |
      gpuTimingScopeToBit(GpuTimingScope::TemporalAAResolve);
  report.shadowSourceFrameIndex = 2u;
  report.temporalAAResolveSourceFrameIndex = 5u;
  report.shadowTimeMs = 0.25f;
  report.temporalAAResolveTimeMs = 0.75f;

  std::map<uint64_t, std::map<std::string, double>> frames;
  applyAutotestGpuTimingReport(frames, report);

  ASSERT_EQ(frames.size(), 2u);
  EXPECT_NEAR(frames[2u]["gpu.scopes.shadow_ms"], 0.25, 1.0e-6);
  EXPECT_NEAR(frames[2u]["gpu.scopes_sum_ms"], 0.25, 1.0e-6);
  EXPECT_EQ(frames[2u].find("gpu.scopes.taa_resolve_ms"), frames[2u].end());
  EXPECT_NEAR(frames[5u]["gpu.scopes.taa_resolve_ms"], 0.75, 1.0e-6);
  EXPECT_NEAR(frames[5u]["gpu.scopes_sum_ms"], 0.75, 1.0e-6);
}

TEST(NuriAutotestingTest, GpuTimingSumSkipsParentedChildScopes) {
  GpuTimingReport report{};
  report.availableScopeMask =
      gpuTimingScopeToBit(GpuTimingScope::WholeFrame) |
      gpuTimingScopeToBit(GpuTimingScope::Shadow) |
      gpuTimingScopeToBit(GpuTimingScope::ShadowDepth) |
      gpuTimingScopeToBit(GpuTimingScope::ShadowSdsm) |
      gpuTimingScopeToBit(GpuTimingScope::Opaque) |
      gpuTimingScopeToBit(GpuTimingScope::Velocity) |
      gpuTimingScopeToBit(GpuTimingScope::ReactiveMask) |
      gpuTimingScopeToBit(GpuTimingScope::TemporalAAResolve) |
      gpuTimingScopeToBit(GpuTimingScope::TemporalAACopyBack) |
      gpuTimingScopeToBit(GpuTimingScope::GTAO) |
      gpuTimingScopeToBit(GpuTimingScope::GTAOTemporal);
  report.shadowSourceFrameIndex = 7u;
  report.wholeFrameSourceFrameIndex = 7u;
  report.shadowDepthSourceFrameIndex = 7u;
  report.shadowSdsmSourceFrameIndex = 7u;
  report.opaqueSourceFrameIndex = 7u;
  report.velocitySourceFrameIndex = 7u;
  report.reactiveMaskSourceFrameIndex = 7u;
  report.temporalAAResolveSourceFrameIndex = 7u;
  report.temporalAACopyBackSourceFrameIndex = 7u;
  report.gtaoSourceFrameIndex = 7u;
  report.gtaoTemporalSourceFrameIndex = 7u;
  report.shadowTimeMs = 2.0f;
  report.wholeFrameTimeMs = 12.0f;
  report.shadowDepthTimeMs = 1.5f;
  report.shadowSdsmTimeMs = 0.25f;
  report.opaqueTimeMs = 3.0f;
  report.velocityTimeMs = 0.4f;
  report.reactiveMaskTimeMs = 0.2f;
  report.temporalAAResolveTimeMs = 1.0f;
  report.temporalAACopyBackTimeMs = 0.3f;
  report.gtaoTimeMs = 2.0f;
  report.gtaoTemporalTimeMs = 0.5f;

  std::map<uint64_t, std::map<std::string, double>> frames;
  applyAutotestGpuTimingReport(frames, report);

  ASSERT_EQ(frames.size(), 1u);
  EXPECT_NEAR(frames[7u]["gpu.frame_ms"], 12.0, 1.0e-6);
  EXPECT_NEAR(frames[7u]["gpu.scopes.shadow_ms"], 2.0, 1.0e-6);
  EXPECT_NEAR(frames[7u]["gpu.scopes.shadow_depth_ms"], 1.5, 1.0e-6);
  EXPECT_NEAR(frames[7u]["gpu.scopes.shadow_sdsm_ms"], 0.25, 1.0e-6);
  EXPECT_NEAR(frames[7u]["gpu.scopes.opaque_ms"], 3.0, 1.0e-6);
  EXPECT_NEAR(frames[7u]["gpu.scopes.velocity_ms"], 0.4, 1.0e-6);
  EXPECT_NEAR(frames[7u]["gpu.scopes.reactive_mask_ms"], 0.2, 1.0e-6);
  EXPECT_NEAR(frames[7u]["gpu.scopes.taa_resolve_ms"], 1.0, 1.0e-6);
  EXPECT_NEAR(frames[7u]["gpu.scopes.taa_copy_back_ms"], 0.3, 1.0e-6);
  EXPECT_NEAR(frames[7u]["gpu.scopes.gtao_ms"], 2.0, 1.0e-6);
  EXPECT_NEAR(frames[7u]["gpu.scopes.gtao_temporal_ms"], 0.5, 1.0e-6);
  EXPECT_NEAR(frames[7u]["gpu.scopes_sum_ms"], 8.0, 1.0e-6);
}

TEST(NuriAutotestingTest, ReportJsonRoundTripsAndHtmlEscapes) {
  const std::filesystem::path path = makeTempPath("autotest_report", ".json");
  AutotestReport report = makeReport(path.parent_path());
  report.baselineProfileCompatible = true;

  auto written = writeAutotestReportFile(report, path);
  ASSERT_FALSE(written.hasError()) << written.error();

  std::ifstream envelopeFile(path, std::ios::binary);
  const std::string envelopeJson((std::istreambuf_iterator<char>(envelopeFile)),
                                 std::istreambuf_iterator<char>());
  envelopeFile.close();
  auto envelope = nuri::tools::core::readResultEnvelopeV2(envelopeJson);
  ASSERT_FALSE(envelope.hasError()) << envelope.error();
  EXPECT_EQ(envelope.value().tool, nuri::tools::core::ResultToolV2::Autotest);
  EXPECT_EQ(envelope.value().status, nuri::tools::core::ToolOutcome::Pass);
  EXPECT_NE(envelope.value().payloadJson.find("\"checkpoints\""),
            std::string::npos);
  EXPECT_NE(envelope.value().payloadJson.find("\"metricWindows\""),
            std::string::npos);
  EXPECT_NE(envelope.value().payloadJson.find("\"frames\""), std::string::npos);

  auto loaded = readAutotestReportFile(path);
  ASSERT_FALSE(loaded.hasError()) << loaded.error();
  EXPECT_EQ(loaded.value().kind, "nuri.autotest.report");
  EXPECT_EQ(loaded.value().baselineProfile, "local-nvrhi-visible");
  EXPECT_EQ(loaded.value().status, "pass");
  EXPECT_EQ(loaded.value().exitCode, AutotestExitCode::Success);
  EXPECT_EQ(loaded.value().selection.requested, "case.<auto>&test");
  EXPECT_EQ(loaded.value().selection.passed, 1u);
  EXPECT_EQ(loaded.value().testCase.id, "case.<auto>&test");
  EXPECT_EQ(loaded.value().testCase.resolution[0], 640u);
  EXPECT_EQ(loaded.value().environment.commitHash, "deadbeef");
  EXPECT_EQ(loaded.value().environment.gpuDeviceName, "test gpu");
  EXPECT_EQ(loaded.value().environment.gpuVendorId, 0x10deu);
  EXPECT_EQ(loaded.value().environment.gpuDeviceId, 0x2684u);
  EXPECT_EQ(loaded.value().environment.gpuDriverVersion, "test-driver");
  EXPECT_EQ(loaded.value().run.renderedFrames, 5u);
  EXPECT_EQ(loaded.value().run.readoutDrainFrames, 1u);
  EXPECT_EQ(loaded.value().run.readoutDrainFrameLimit, 4u);
  EXPECT_EQ(loaded.value().run.readoutDrainElapsedMs, 12u);
  EXPECT_EQ(loaded.value().run.windowModeSource, "manifest");
  ASSERT_EQ(loaded.value().checkpoints.size(), 1u);
  EXPECT_EQ(loaded.value().checkpoints[0].id, "end<frame>&");
  EXPECT_EQ(loaded.value().checkpoints[0].measurements["renderer.frame_index"],
            4.0);
  ASSERT_EQ(loaded.value().checkpoints[0].captures.size(), 1u);
  EXPECT_EQ(loaded.value().checkpoints[0].captures[0].snapshot.artifactStem,
            "final_color");
  EXPECT_EQ(loaded.value().checkpoints[0].captures[0].snapshot.mip, 1u);
  EXPECT_EQ(loaded.value().checkpoints[0].captures[0].snapshot.layer, 2u);
  EXPECT_EQ(loaded.value().checkpoints[0].captures[0].snapshot.kind, "color");
  EXPECT_EQ(
      loaded.value().checkpoints[0].captures[0].snapshot.semanticMetrics.unit,
      "display_linear");
  EXPECT_DOUBLE_EQ(loaded.value()
                       .checkpoints[0]
                       .captures[0]
                       .snapshot.semanticMetrics.maxError,
                   0.125);
  EXPECT_EQ(loaded.value().checkpoints[0].warnings[0], "checkpoint warning");
  EXPECT_EQ(loaded.value().checkpoints[0].errors[0], "checkpoint error");
  ASSERT_EQ(loaded.value().metricWindows.size(), 1u);
  EXPECT_EQ(loaded.value().metricWindows[0].id, "steady<pan>&");
  ASSERT_EQ(loaded.value().metricWindows[0].assertions.size(), 1u);
  EXPECT_EQ(loaded.value().metricWindows[0].assertions[0].statistic, "p95");
  EXPECT_EQ(loaded.value().metricWindows[0].assertions[0].sampleCount, 3u);
  EXPECT_EQ(loaded.value().metricWindows[0].assertions[0].expectedSampleCount,
            3u);
  EXPECT_EQ(loaded.value().metricWindows[0].warnings[0], "window warning");
  EXPECT_EQ(loaded.value().metricWindows[0].errors[0], "window error");
  ASSERT_EQ(loaded.value().checkpoints[0].readouts.size(), 1u);
  EXPECT_EQ(loaded.value().checkpoints[0].readouts[0].id, "shadow<probe>&");
  EXPECT_EQ(loaded.value().checkpoints[0].readouts[0].values["valid"], 1.0);
  ASSERT_EQ(loaded.value().checkpoints[0].readouts[0].assertions.size(), 1u);
  ASSERT_EQ(loaded.value().frames.size(), 1u);
  EXPECT_EQ(loaded.value().frames[0].frameIndex, 4u);
  EXPECT_EQ(loaded.value().frames[0].measurements["renderer.frame_index"], 4.0);
  ASSERT_EQ(loaded.value().unavailableMetrics.size(), 1u);
  EXPECT_EQ(loaded.value().unavailableMetrics[0], "gpu.scopes.missing_ms");
  EXPECT_EQ(loaded.value().warnings[0], "report warning");
  EXPECT_EQ(loaded.value().errors[0], "report error");

  auto html = writeAutotestHtmlReport(report);
  ASSERT_FALSE(html.hasError()) << html.error();
  EXPECT_NE(html.value().find("case.&lt;auto&gt;&amp;test"), std::string::npos);
  EXPECT_NE(html.value().find("capture&lt;ok&gt;&amp;"), std::string::npos);
  EXPECT_EQ(html.value().find("capture<ok>&"), std::string::npos);
  EXPECT_NE(html.value().find("steady&lt;pan&gt;&amp;"), std::string::npos);
  EXPECT_NE(html.value().find("shadow&lt;probe&gt;&amp;"), std::string::npos);
  EXPECT_NE(html.value().find("readout&lt;ok&gt;&amp;"), std::string::npos);
  EXPECT_NE(html.value().find("<html lang=\"en\">"), std::string::npos);
  EXPECT_NE(html.value().find("class=\"skip-link\""), std::string::npos);
  EXPECT_NE(html.value().find("<main id=\"main-content\""), std::string::npos);
  EXPECT_NE(html.value().find("id=\"checkpoint-search\""), std::string::npos);
  EXPECT_NE(html.value().find("aria-live=\"polite\""), std::string::npos);
  EXPECT_NE(html.value().find("<caption>Checkpoint assertion outcomes"),
            std::string::npos);
  EXPECT_NE(html.value().find("class=\"table-wrap\""), std::string::npos);
  EXPECT_NE(html.value().find("prefers-reduced-motion"), std::string::npos);

  auto suiteHtml = writeAutotestSuiteHtml({report}, "suite<auto>&");
  ASSERT_FALSE(suiteHtml.hasError()) << suiteHtml.error();
  EXPECT_NE(suiteHtml.value().find("suite&lt;auto&gt;&amp;"),
            std::string::npos);
  EXPECT_NE(suiteHtml.value().find("role=\"search\""), std::string::npos);
  EXPECT_NE(suiteHtml.value().find("id=\"case-results\""), std::string::npos);
  EXPECT_NE(suiteHtml.value().find("class=\"case-grid\""), std::string::npos);

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

TEST(NuriAutotestingTest, DetailedReportReaderRejectsNestedContractDrift) {
  AutotestReport report = makeReport("artifacts/autotests/contract/case");
  report.artifacts.artifactDir = "artifacts/autotests/contract";
  auto json = writeAutotestReportJson(report);
  ASSERT_FALSE(json.hasError()) << json.error();
  const std::vector<std::string> invalid{
      replaceFirst(json.value(), "\"measurements\": {",
                   "\"measurements\": {\"surprise\": \"bad\","),
      replaceFirst(json.value(), "\"validForComparison\": true",
                   "\"validForComparison\": \"true\""),
      replaceFirst(
          json.value(), "\"validForComparison\": true",
          "\"validForComparison\": true, \"validForComparison\": true"),
      replaceFirst(json.value(), "artifacts/autotests/contract",
                   "artifacts/../escape"),
      json.value().substr(0u, json.value().size() / 2u)};
  for (size_t index = 0u; index < invalid.size(); ++index) {
    const auto path = makeTempPath(
        "autotest_adversarial_report_" + std::to_string(index), ".json");
    writeFile(path, invalid[index]);
    EXPECT_TRUE(readAutotestReportFile(path).hasError()) << index;
    std::filesystem::remove(path);
  }
}

TEST(NuriAutotestingTest, DryRunWritesCheckpointReports) {
  const std::filesystem::path path = makeTempPath("autotest_dry_case", ".json");
  writeFile(path,
            R"json({
              "schemaVersion": 1,
              "id": "dry.run.case",
              "suite": "smoke",
              "endFrame": 4,
              "checkpoints": [
                {
                  "id": "warm",
                  "frame": 2,
                  "captures": [
                    {
                      "target": "final_color",
                      "profile": "ldr_color",
                      "required": true,
                      "compare": true
                    }
                  ]
                },
                {
                  "id": "end",
                  "frame": 4,
                  "captures": [
                    {
                      "target": "final_color",
                      "profile": "ldr_color",
                      "required": true,
                      "compare": true
                    }
                  ]
                }
              ]
            })json");
  auto testCase = loadAutotestCaseManifest(path);
  ASSERT_FALSE(testCase.hasError()) << testCase.error();

  const std::filesystem::path artifacts = makeTempPath("autotest_dry", "");
  AutotestRunResult result = runAutotestCase(
      std::move(testCase.value()),
      AutotestRunOptions{.artifactDir = artifacts, .dryRun = true});

  EXPECT_EQ(result.exitCode, AutotestExitCode::Success) << result.message;
  EXPECT_TRUE(std::filesystem::exists(result.reportPath));
  EXPECT_TRUE(std::filesystem::exists(result.htmlPath));
  ASSERT_EQ(result.report.checkpoints.size(), 2u);
  EXPECT_EQ(result.report.checkpoints[0].captures[0].snapshot.statusReason,
            "dry_run");

  std::error_code ec;
  std::filesystem::remove_all(artifacts, ec);
  std::filesystem::remove(path, ec);
}

TEST(NuriAutotestingTest, RunRejectsUnsafeBaselineProfileBeforeWriting) {
  AutotestCase testCase{};
  testCase.id = "safe.case";
  testCase.suite = "smoke";
  const std::filesystem::path artifacts =
      makeTempPath("autotest_unsafe_profile", "");

  AutotestRunResult result = runAutotestCase(
      std::move(testCase), AutotestRunOptions{.artifactDir = artifacts,
                                              .baselineProfile = "../escaped",
                                              .dryRun = true});

  EXPECT_EQ(result.exitCode, AutotestExitCode::InvalidInput);
  EXPECT_TRUE(result.reportPath.empty());
  EXPECT_FALSE(std::filesystem::exists(artifacts));
}

} // namespace
