#include "nuri/tools/core/result_envelope_v2.h"

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

namespace {

class ScopedResultEnvelopeTempDir {
public:
  ScopedResultEnvelopeTempDir() {
    const auto tick =
        std::chrono::steady_clock::now().time_since_epoch().count();
    path = std::filesystem::temp_directory_path() /
           ("nuri_result_envelope_" + std::to_string(tick));
    std::error_code error;
    std::filesystem::create_directories(path, error);
    EXPECT_FALSE(error);
  }

  ~ScopedResultEnvelopeTempDir() {
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }

  std::filesystem::path path{};
};

[[nodiscard]] nuri::tools::core::ResultEnvelopeV2 makeEnvelope() {
  using namespace nuri::tools::core;
  ResultEnvelopeV2 envelope{};
  envelope.tool = ResultToolV2::Snapshot;
  envelope.toolVersion = "2.0-test";
  envelope.runId = "20260710T123456.123Z-contract";
  envelope.status = ToolOutcome::Pass;
  envelope.exitCode = 0;
  envelope.authoritative = false;
  envelope.startedAtUtc = "2026-07-10T12:34:56.123Z";
  envelope.durationMs = 12.5;
  envelope.command = std::vector<std::string>{"nuri-snapshot", "run", "smoke"};
  envelope.reproduceCommand = "nuri-snapshot run smoke";
  envelope.selection = {.requested = "smoke",
                        .selected = 1u,
                        .attempted = 1u,
                        .completed = 1u,
                        .passed = 1u};
  envelope.profile = ResultProfileV2{.id = "local-nvrhi-visible",
                                     .compatible = true,
                                     .incompatibilityReasons = {}};
  envelope.environmentFingerprint =
      "sha256:"
      "1111111111111111111111111111111111111111111111111111111111111111";
  envelope.workloadFingerprint =
      "sha256:"
      "2222222222222222222222222222222222222222222222222222222222222222";
  envelope.diagnostics.push_back(
      {.code = "snapshot.captured",
       .severity = ResultDiagnosticSeverityV2::Info,
       .message = "captured",
       .contextJson = R"({"frame":42,"nested":{"ok":true}})"});
  envelope.artifacts.push_back(
      {.role = "snapshot.actual",
       .path = std::filesystem::path(u8"captures/кадр.png"),
       .mediaType = "image/png",
       .digest =
           "sha256:"
           "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
       .status = ResultArtifactStatusV2::Complete});
  envelope.children.push_back({.id = "snapshot.smoke.case",
                               .status = "pass",
                               .exitCode = 0,
                               .result = std::filesystem::path(
                                   "cases/snapshot.smoke.case/report.json")});
  envelope.payloadJson =
      R"({"capture":{"target":"final_color"},"values":[1,true,null]})";
  envelope.extensionsJson = R"({"vendor":{"opaque":[1,2,3]}})";
  return envelope;
}

TEST(NuriResultEnvelopeV2Test, StructuredRoundTripPreservesFullContract) {
  using namespace nuri::tools::core;
  const ResultEnvelopeV2 expected = makeEnvelope();
  auto serialized = serializeResultEnvelopeV2(expected);
  ASSERT_FALSE(serialized.hasError()) << serialized.error();

  auto parsed = readResultEnvelopeV2(serialized.value());
  ASSERT_FALSE(parsed.hasError()) << parsed.error();
  EXPECT_EQ(parsed->tool, ResultToolV2::Snapshot);
  EXPECT_EQ(parsed->runId, expected.runId);
  EXPECT_EQ(parsed->status, ToolOutcome::Pass);
  EXPECT_EQ(parsed->selection.selected, 1u);
  ASSERT_EQ(parsed->diagnostics.size(), 1u);
  ASSERT_TRUE(parsed->diagnostics.front().contextJson.has_value());
  EXPECT_NE(parsed->diagnostics.front().contextJson->find("\"frame\":42"),
            std::string::npos);
  ASSERT_EQ(parsed->artifacts.size(), 1u);
  EXPECT_TRUE(parsed->artifacts.front().path.generic_u8string() ==
              expected.artifacts.front().path.generic_u8string());
  ASSERT_TRUE(parsed->extensionsJson.has_value());
  EXPECT_NE(parsed->extensionsJson->find("\"opaque\""), std::string::npos);

  auto secondSerialization = serializeResultEnvelopeV2(parsed.value());
  ASSERT_FALSE(secondSerialization.hasError()) << secondSerialization.error();
  EXPECT_EQ(secondSerialization.value(), serialized.value());
}

TEST(NuriResultEnvelopeV2Test,
     FullReaderRejectsUnknownDuplicateAndInvalidFields) {
  using namespace nuri::tools::core;
  auto serialized = serializeResultEnvelopeV2(makeEnvelope());
  ASSERT_FALSE(serialized.hasError()) << serialized.error();

  std::string unknownRoot = serialized.value();
  unknownRoot.insert(unknownRoot.find('{') + 1u, "\n  \"unknown\": true,");
  EXPECT_TRUE(readResultEnvelopeV2(unknownRoot).hasError());

  std::string duplicateRoot = serialized.value();
  duplicateRoot.insert(duplicateRoot.find('{') + 1u,
                       "\n  \"kind\": \"nuri.tool.result\",");
  EXPECT_TRUE(readResultEnvelopeV2(duplicateRoot).hasError());

  std::string unknownSelection = serialized.value();
  const size_t selected = unknownSelection.find("\"selected\"");
  ASSERT_NE(selected, std::string::npos);
  unknownSelection.insert(selected, "\"unexpected\": 0,\n    ");
  EXPECT_TRUE(readResultEnvelopeV2(unknownSelection).hasError());

  ResultEnvelopeV2 invalid = makeEnvelope();
  invalid.artifacts.front().path = "../escape.png";
  EXPECT_TRUE(serializeResultEnvelopeV2(invalid).hasError());
  invalid = makeEnvelope();
  invalid.payloadJson = "[]";
  EXPECT_TRUE(serializeResultEnvelopeV2(invalid).hasError());
  invalid = makeEnvelope();
  invalid.startedAtUtc = "2026-02-30T12:00:00Z";
  EXPECT_TRUE(serializeResultEnvelopeV2(invalid).hasError());

  invalid = makeEnvelope();
  invalid.profile->compatible = false;
  invalid.profile->incompatibilityReasons = {"profile mismatch"};
  EXPECT_TRUE(serializeResultEnvelopeV2(invalid).hasError());

  invalid.status = ToolOutcome::Investigative;
  invalid.selection.passed = 0u;
  invalid.selection.warned = 1u;
  EXPECT_FALSE(serializeResultEnvelopeV2(invalid).hasError());

  invalid = makeEnvelope();
  invalid.selection.failed = 1u;
  EXPECT_TRUE(serializeResultEnvelopeV2(invalid).hasError());

  invalid = makeEnvelope();
  invalid.authoritative = true;
  invalid.status = ToolOutcome::Investigative;
  EXPECT_TRUE(serializeResultEnvelopeV2(invalid).hasError());

  invalid = makeEnvelope();
  invalid.environmentFingerprint = "sha256:not-a-digest";
  EXPECT_TRUE(serializeResultEnvelopeV2(invalid).hasError());

  invalid = makeEnvelope();
  invalid.selection = {};
  EXPECT_TRUE(serializeResultEnvelopeV2(invalid).hasError());
}

TEST(NuriResultEnvelopeV2Test, AtomicWriterSupportsUnicodePaths) {
  using namespace nuri::tools::core;
  ScopedResultEnvelopeTempDir temp;
  const std::filesystem::path path =
      temp.path / std::filesystem::path(u8"результат.json");
  auto written = writeResultEnvelopeV2(path, makeEnvelope());
  ASSERT_FALSE(written.hasError()) << written.error();

  std::ifstream file(path, std::ios::binary);
  const std::string json((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
  auto parsed = readResultEnvelopeV2(json);
  ASSERT_FALSE(parsed.hasError()) << parsed.error();
  EXPECT_EQ(parsed->runId, "20260710T123456.123Z-contract");
}

} // namespace
