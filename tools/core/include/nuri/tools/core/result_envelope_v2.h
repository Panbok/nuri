#pragma once

#include "nuri/core/result.h"
#include "nuri/tools/core/result_protocol.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace nuri::tools::core {

enum class ResultToolV2 : uint8_t {
  Benchmark,
  Snapshot,
  Autotest,
  Validate,
};

enum class ResultDiagnosticSeverityV2 : uint8_t {
  Info,
  Warning,
  Error,
};

enum class ResultArtifactStatusV2 : uint8_t {
  Complete,
  Missing,
  Invalid,
};

struct ResultSelectionV2 {
  std::optional<std::string> requested{};
  uint64_t selected = 0u;
  uint64_t attempted = 0u;
  uint64_t completed = 0u;
  uint64_t passed = 0u;
  uint64_t warned = 0u;
  uint64_t failed = 0u;
  uint64_t skipped = 0u;
  uint64_t unavailable = 0u;
  uint64_t notRun = 0u;
};

struct ResultProfileV2 {
  std::string id{};
  bool compatible = false;
  std::vector<std::string> incompatibilityReasons{};
};

struct ResultDiagnosticV2 {
  std::string code{};
  ResultDiagnosticSeverityV2 severity = ResultDiagnosticSeverityV2::Info;
  std::string message{};
  std::optional<std::string> contextJson{};
};

struct ResultArtifactV2 {
  std::string role{};
  std::filesystem::path path{};
  std::string mediaType{};
  std::string digest{};
  ResultArtifactStatusV2 status = ResultArtifactStatusV2::Complete;
};

struct ResultChildV2 {
  std::string id{};
  std::string status{};
  int exitCode = 0;
  std::optional<std::filesystem::path> result{};
};

struct ResultEnvelopeV2 {
  ResultToolV2 tool = ResultToolV2::Validate;
  std::optional<std::string> toolVersion{};
  std::string runId{};
  ToolOutcome status = ToolOutcome::RuntimeError;
  int exitCode = 4;
  bool authoritative = false;
  std::optional<std::string> startedAtUtc{};
  std::optional<double> durationMs{};
  std::optional<std::vector<std::string>> command{};
  std::optional<std::string> reproduceCommand{};
  ResultSelectionV2 selection{};
  std::optional<ResultProfileV2> profile{};
  std::optional<std::string> environmentFingerprint{};
  std::optional<std::string> workloadFingerprint{};
  std::vector<ResultDiagnosticV2> diagnostics{};
  std::vector<ResultArtifactV2> artifacts{};
  std::vector<ResultChildV2> children{};
  std::string payloadJson = "{}";
  std::optional<std::string> extensionsJson{};
};

[[nodiscard]] std::string_view resultToolV2Name(ResultToolV2 tool) noexcept;
[[nodiscard]] std::string_view
resultDiagnosticSeverityV2Name(ResultDiagnosticSeverityV2 severity) noexcept;
[[nodiscard]] std::string_view
resultArtifactStatusV2Name(ResultArtifactStatusV2 status) noexcept;

[[nodiscard]] Result<void, std::string>
validateResultEnvelopeV2(const ResultEnvelopeV2 &envelope);
[[nodiscard]] Result<std::string, std::string>
serializeResultEnvelopeV2(const ResultEnvelopeV2 &envelope);
[[nodiscard]] Result<ResultEnvelopeV2, std::string>
readResultEnvelopeV2(std::string_view json);
[[nodiscard]] Result<void, std::string>
writeResultEnvelopeV2(const std::filesystem::path &path,
                      const ResultEnvelopeV2 &envelope);

} // namespace nuri::tools::core
