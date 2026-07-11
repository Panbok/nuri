#include "nuri/tools/benchmark/benchmark_environment.h"
#include "nuri/tools/benchmark/benchmark_report.h"
#include "nuri/tools/core/result_envelope_v2.h"
#include "nuri/tools/core/run_workspace.h"
#include "nuri/tools/core/sha256.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <thread>

namespace {

[[nodiscard]] std::optional<std::string> optionValue(int argc, char **argv,
                                                     std::string_view name) {
  for (int index = 2; index + 1 < argc; ++index) {
    if (argv[index] != nullptr && name == argv[index]) {
      return argv[index + 1] != nullptr
                 ? std::optional<std::string>{argv[index + 1]}
                 : std::nullopt;
    }
  }
  return std::nullopt;
}

} // namespace

int main(int argc, char **argv) {
  using namespace nuri::tools::benchmark;
  using namespace nuri::tools::core;
  if (argc < 2 || std::string_view(argv[1]) != "__run-child") {
    return static_cast<int>(BenchmarkExitCode::InvalidInput);
  }
  const auto caseId = optionValue(argc, argv, "--case");
  const auto artifactDirText = optionValue(argc, argv, "--artifact-dir");
  const auto repetitionIndexText =
      optionValue(argc, argv, "--repetition-index");
  if (!caseId.has_value() || !artifactDirText.has_value() ||
      !repetitionIndexText.has_value()) {
    return static_cast<int>(BenchmarkExitCode::InvalidInput);
  }
  const uint32_t repetitionIndex =
      static_cast<uint32_t>(std::stoul(*repetitionIndexText));
  const std::filesystem::path artifactDir(*artifactDirText);
  const std::filesystem::path reportPath = artifactDir / "report.json";
  const std::filesystem::path envelopePath = artifactDir / "run.json";

  std::cout << "fixture repetition " << repetitionIndex << '\n' << std::flush;
  std::cerr << "fixture diagnostic " << repetitionIndex << '\n' << std::flush;
  if (caseId->find("timeout") != std::string::npos) {
    std::this_thread::sleep_for(std::chrono::seconds(5));
    return 0;
  }
  if (caseId->find("failure") != std::string::npos) {
    return static_cast<int>(BenchmarkExitCode::RuntimeError);
  }

  BenchmarkReport report{};
  report.generatedAtUtc = utcTimestampIso8601();
  report.command = "deterministic benchmark child fixture";
  report.benchmarkCase.id = *caseId;
  report.benchmarkCase.suite = "test";
  report.benchmarkCase.samples = 1u;
  report.benchmarkCase.warmupFrames = 1u;
  report.benchmarkCase.measurementFrames = 2u;
  report.benchmarkCase.requiredMetrics = {"cpu.render_submit_ms"};
  report.run.samples = 1u;
  report.run.warmupFrames = 1u;
  report.run.measurementFrames = 2u;
  report.run.validForComparison = true;
  report.environment.osName = "TestOS";
  report.environment.osVersion = "1.0";
  report.environment.cpuName = "Fixture CPU";
  report.environment.cpuLogicalThreadCount = 8u;
  report.environment.gpuBackend = "nvrhi";
  report.environment.gpuBackendSource = "default";
  report.environment.gpuDeviceName = "Fixture GPU";
  report.environment.gpuVendorId = 0x10deu;
  report.environment.gpuDeviceId = 0x1234u;
  report.environment.gpuDriverVersion = "fixture-driver";
  report.environment.swapchainImageCount = 3u;
  report.environment.requestedPresentMode = "immediate";
  report.environment.resolvedPresentMode = "immediate";
  report.environment.windowMode = "windowed";
  report.environment.windowVisible = true;
  report.environment.renderGraphWorkerCount = 1u;
  report.environment.buildType = "Release";
  report.environment.cmakeToolProfile = "tools";
  const double value = 10.0 + static_cast<double>(repetitionIndex);
  report.frames.push_back(
      {.frameIndex = 1u,
       .sampleIndex = 0u,
       .measured = true,
       .measurements = {{"cpu.render_submit_ms", value - 0.5},
                        {"gpu.scopes_sum_ms", value * 0.8 - 0.4}}});
  report.frames.push_back(
      {.frameIndex = 2u,
       .sampleIndex = 0u,
       .measured = true,
       .measurements = {{"cpu.render_submit_ms", value + 0.5},
                        {"gpu.scopes_sum_ms", value * 0.8 + 0.4}}});
  computeBenchmarkReportStats(report);
  if (caseId->find("unknown_warmup") == std::string::npos) {
    report.sampleStats.front().warmupStable = true;
  }
  auto reportWrite = writeBenchmarkReportFile(report, reportPath, true);
  if (reportWrite.hasError()) {
    std::cerr << reportWrite.error() << '\n';
    return static_cast<int>(BenchmarkExitCode::RuntimeError);
  }

  auto payload = writeBenchmarkReportJson(report, true);
  auto digest = sha256File(reportPath);
  if (payload.hasError() || digest.hasError()) {
    return static_cast<int>(BenchmarkExitCode::RuntimeError);
  }
  ResultEnvelopeV2 envelope{};
  envelope.tool = ResultToolV2::Benchmark;
  envelope.runId = createRunId();
  envelope.status = ToolOutcome::Investigative;
  envelope.exitCode = 0;
  envelope.authoritative = false;
  envelope.selection = {.requested = *caseId,
                        .selected = 1u,
                        .attempted = 1u,
                        .completed = 1u,
                        .warned = 1u};
  envelope.artifacts.push_back({.role = "benchmark.case.report",
                                .path = "report.json",
                                .mediaType = "application/json",
                                .digest = "sha256:" + digest.value(),
                                .status = ResultArtifactStatusV2::Complete});
  envelope.children.push_back({.id = *caseId,
                               .status = "investigative",
                               .exitCode = 0,
                               .result = std::filesystem::path("report.json")});
  envelope.payloadJson = std::move(payload.value());
  auto envelopeWrite = writeResultEnvelopeV2(envelopePath, envelope);
  if (envelopeWrite.hasError()) {
    std::cerr << envelopeWrite.error() << '\n';
    return static_cast<int>(BenchmarkExitCode::RuntimeError);
  }
  return 0;
}
