#include "nuri/tools/core/atomic_file.h"
#include "nuri/tools/core/case_catalog.h"
#include "nuri/tools/core/environment_probe.h"
#include "nuri/tools/core/fingerprint.h"
#include "nuri/tools/core/identifier.h"
#include "nuri/tools/core/result_protocol.h"
#include "nuri/tools/core/run_workspace.h"
#include "nuri/tools/core/safe_path.h"
#include "nuri/tools/core/sha256.h"

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace {

class ScopedToolCoreTempDir {
public:
  ScopedToolCoreTempDir() {
    const auto tick =
        std::chrono::steady_clock::now().time_since_epoch().count();
    path = std::filesystem::temp_directory_path() /
           ("nuri_tool_core_" + std::to_string(tick));
    std::error_code error;
    std::filesystem::create_directories(path, error);
    EXPECT_FALSE(error);
  }

  ~ScopedToolCoreTempDir() {
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }

  std::filesystem::path path{};
};

TEST(NuriToolCoreTest, IdentifierGrammarRejectsUnsafeSegments) {
  using namespace nuri::tools::core;
  EXPECT_FALSE(validateIdentifier("renderer.damaged_helmet", "case",
                                  IdentifierShape::Dotted)
                   .hasError());
  EXPECT_FALSE(validateIdentifier("local-nvrhi-visible", "profile").hasError());
  EXPECT_TRUE(validateIdentifier("../outside", "case", IdentifierShape::Dotted)
                  .hasError());
  EXPECT_TRUE(
      validateIdentifier("Renderer.case", "case", IdentifierShape::Dotted)
          .hasError());
  EXPECT_TRUE(validateIdentifier("con", "profile").hasError());
  EXPECT_TRUE(validateIdentifier("case..name", "case", IdentifierShape::Dotted)
                  .hasError());
}

TEST(NuriToolCoreTest, FingerprintsAreCanonicalAndRejectAmbiguity) {
  using namespace nuri::tools::core;
  auto first = makeSha256Fingerprint(
      {{.name = "gpu", .value = "10:de"}, {.name = "os", .value = "win"}});
  auto reordered = makeSha256Fingerprint(
      {{.name = "os", .value = "win"}, {.name = "gpu", .value = "10:de"}});
  ASSERT_FALSE(first.hasError()) << first.error();
  ASSERT_FALSE(reordered.hasError()) << reordered.error();
  EXPECT_EQ(first.value(), reordered.value());
  EXPECT_EQ(first.value().size(), 71u);

  EXPECT_TRUE(makeSha256Fingerprint({{.name = "gpu", .value = "a"},
                                     {.name = "gpu", .value = "b"}})
                  .hasError());
  EXPECT_TRUE(makeSha256Fingerprint({{.name = "", .value = "x"}}).hasError());
}

TEST(NuriToolCoreTest, CaseManifestDiscoveryIsRecursiveAndDeterministic) {
  using namespace nuri::tools::core;
  ScopedToolCoreTempDir temp;
  std::filesystem::create_directories(temp.path / "nested");
  std::ofstream(temp.path / "z.json") << "{}";
  std::ofstream(temp.path / "nested" / "a.json") << "{}";
  std::ofstream(temp.path / "ignored.txt") << "{}";

  auto discovered = discoverCaseManifestPaths(temp.path);
  ASSERT_FALSE(discovered.hasError()) << discovered.error();
  ASSERT_EQ(discovered.value().size(), 2u);
  EXPECT_EQ(discovered.value()[0], temp.path / "nested" / "a.json");
  EXPECT_EQ(discovered.value()[1], temp.path / "z.json");

  auto missing = discoverCaseManifestPaths(temp.path / "missing");
  ASSERT_FALSE(missing.hasError()) << missing.error();
  EXPECT_TRUE(missing.value().empty());
}

TEST(NuriToolCoreTest, CaseCatalogRejectsDuplicatesAndInvalidMetadata) {
  using namespace nuri::tools::core;
  std::vector<CaseCatalogEntry> entries{
      {.id = "smoke.first", .suite = "smoke", .manifestPath = "z.json"},
      {.id = "smoke.first", .suite = "smoke", .manifestPath = "a.json"},
  };
  auto duplicate = validateCaseCatalog(entries, "test");
  ASSERT_TRUE(duplicate.hasError());
  EXPECT_NE(duplicate.error().find("duplicate test case id 'smoke.first'"),
            std::string::npos);
  EXPECT_LT(duplicate.error().find("a.json"), duplicate.error().find("z.json"));

  entries.resize(1u);
  entries[0].suite = "../unsafe";
  auto invalid = validateCaseCatalog(entries, "test");
  ASSERT_TRUE(invalid.hasError());
  EXPECT_NE(invalid.error().find("case suite"), std::string::npos);
}

TEST(NuriToolCoreTest, CaseCatalogSelectorsAreDeterministicAndExplicitOnZero) {
  using namespace nuri::tools::core;
  const std::vector<CaseCatalogEntry> entries{
      {.id = "stress.second",
       .suite = "stress",
       .tags = {"gpu", "slow"},
       .manifestPath = "second.json"},
      {.id = "smoke.first",
       .suite = "smoke",
       .tags = {"gpu", "fast"},
       .manifestPath = "first.json"},
      {.id = "stress.first",
       .suite = "stress",
       .tags = {"gpu", "fast"},
       .manifestPath = "third.json"},
  };

  auto suite =
      selectCaseCatalog(entries, CaseCatalogSelector{.suite = "stress"},
                        CaseCatalogZeroMatchPolicy::Reject, "test");
  ASSERT_FALSE(suite.hasError()) << suite.error();
  ASSERT_EQ(suite.value().size(), 2u);
  EXPECT_EQ(entries[suite.value()[0]].id, "stress.first");
  EXPECT_EQ(entries[suite.value()[1]].id, "stress.second");

  auto tags =
      selectCaseCatalog(entries, CaseCatalogSelector{.tags = {"gpu", "fast"}},
                        CaseCatalogZeroMatchPolicy::Reject, "test");
  ASSERT_FALSE(tags.hasError()) << tags.error();
  ASSERT_EQ(tags.value().size(), 2u);
  EXPECT_EQ(entries[tags.value()[0]].id, "smoke.first");
  EXPECT_EQ(entries[tags.value()[1]].id, "stress.first");

  auto rejected =
      selectCaseCatalog(entries, CaseCatalogSelector{.suite = "missing"},
                        CaseCatalogZeroMatchPolicy::Reject, "test");
  ASSERT_TRUE(rejected.hasError());
  EXPECT_NE(rejected.error().find("zero test cases"), std::string::npos);

  auto allowed =
      selectCaseCatalog(entries, CaseCatalogSelector{.suite = "missing"},
                        CaseCatalogZeroMatchPolicy::Allow, "test");
  ASSERT_FALSE(allowed.hasError()) << allowed.error();
  EXPECT_TRUE(allowed.value().empty());
}

TEST(NuriToolCoreTest, SafePathRejectsTraversalAndAbsolutePaths) {
  using namespace nuri::tools::core;
  ScopedToolCoreTempDir temp;
  auto valid = resolvePathUnder(temp.path, "cases/smoke.case");
  ASSERT_FALSE(valid.hasError());
  std::error_code canonicalError;
  const auto canonicalRoot =
      std::filesystem::weakly_canonical(temp.path, canonicalError);
  ASSERT_FALSE(canonicalError);
  EXPECT_EQ(valid.value().parent_path().parent_path(), canonicalRoot);
  EXPECT_TRUE(resolvePathUnder(temp.path, "../outside").hasError());
  EXPECT_TRUE(resolvePathUnder(temp.path, temp.path / "absolute").hasError());
}

TEST(NuriToolCoreTest, ConfinedRemovalCannotDeleteOutsideRoot) {
  using namespace nuri::tools::core;
  ScopedToolCoreTempDir temp;
  const std::filesystem::path outside =
      temp.path.parent_path() / (temp.path.filename().string() + "_outside");
  std::filesystem::create_directories(outside);
  std::ofstream(outside / "keep.txt") << "keep";
  EXPECT_TRUE(removeTreeUnder(temp.path, "../" + outside.filename().string())
                  .hasError());
  EXPECT_TRUE(std::filesystem::exists(outside / "keep.txt"));
  std::error_code error;
  std::filesystem::remove_all(outside, error);
}

TEST(NuriToolCoreTest, AtomicTextWriteReplacesCompleteFile) {
  using namespace nuri::tools::core;
  ScopedToolCoreTempDir temp;
  const std::filesystem::path path = temp.path / "report.json";
  ASSERT_FALSE(atomicWriteTextFile(path, "first").hasError());
  ASSERT_FALSE(atomicWriteTextFile(path, "second").hasError());
  std::ifstream file(path, std::ios::binary);
  std::string text((std::istreambuf_iterator<char>(file)),
                   std::istreambuf_iterator<char>());
  EXPECT_EQ(text, "second");
}

TEST(NuriToolCoreTest, RunWorkspacesAndCasePathsAreUniqueAndConfined) {
  using namespace nuri::tools::core;
  ScopedToolCoreTempDir temp;
  auto first = createRunWorkspace(temp.path);
  auto second = createRunWorkspace(temp.path);
  ASSERT_FALSE(first.hasError());
  ASSERT_FALSE(second.hasError());
  EXPECT_NE(first.value().root, second.value().root);
  auto caseDir = first.value().caseDirectory("smoke.procedural.case");
  ASSERT_FALSE(caseDir.hasError());
  EXPECT_TRUE(caseDir.value().string().find(first.value().root.string()) == 0u);
  EXPECT_TRUE(first.value().caseDirectory("../outside").hasError());
}

TEST(NuriToolCoreTest, OutcomeAggregationUsesSemanticPrecedence) {
  using namespace nuri::tools::core;
  EXPECT_EQ(
      aggregateOutcome(ToolOutcome::Failure, ToolOutcome::MissingBaseline),
      ToolOutcome::MissingBaseline);
  EXPECT_EQ(aggregateOutcome(ToolOutcome::RuntimeError, ToolOutcome::Invalid),
            ToolOutcome::RuntimeError);
  EXPECT_EQ(toolOutcomeExitCode(ToolOutcome::Invalid), 2);
  EXPECT_EQ(toolOutcomeExitCode(ToolOutcome::Investigative), 0);
  EXPECT_FALSE(selectionIsComplete(SelectionCounts{}));
  SelectionCounts complete{
      .selected = 2u, .attempted = 2u, .completed = 2u, .passed = 2u};
  EXPECT_TRUE(selectionIsComplete(complete));
}

TEST(NuriToolCoreTest, Sha256MatchesPublishedVectorsAndFiles) {
  using namespace nuri::tools::core;
  const std::string empty;
  EXPECT_EQ(sha256Hex(std::as_bytes(std::span(empty))),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  const std::string abc = "abc";
  EXPECT_EQ(sha256Hex(std::as_bytes(std::span(abc))),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");

  ScopedToolCoreTempDir temp;
  const auto path = temp.path / "abc.txt";
  std::ofstream(path, std::ios::binary) << abc;
  const auto fileDigest = sha256File(path);
  ASSERT_FALSE(fileDigest.hasError());
  EXPECT_EQ(fileDigest.value(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

TEST(NuriToolCoreTest, EnvironmentProbeUsesInjectedHostAndBuildFacts) {
  using namespace nuri::tools::core;
  const std::filesystem::path repo = "deterministic/repo";
  const EnvironmentBuildFacts build{
      .buildType = "Release",
      .cmakeToolProfile = "contract",
      .vcpkgManifestFeatures = "tools",
      .buildShared = true,
      .loggingEnabled = true,
      .assertsEnabled = false,
      .tracyEnabled = true,
      .devChecks = true,
  };
  const EnvironmentProbeSource source{
      .runCommand = [](std::string_view command, const std::filesystem::path &)
          -> std::optional<std::string> {
        if (command == "git rev-parse HEAD") {
          return " abc123\n";
        }
        if (command == "git rev-parse --abbrev-ref HEAD") {
          return "codex/tooling\r\n";
        }
        if (command == "git status --porcelain") {
          return " M tools/core/environment_probe.cpp\n";
        }
        return std::nullopt;
      },
      .osName = []() { return std::optional<std::string>("TestOS"); },
      .osVersion = []() { return std::optional<std::string>("1.2.3"); },
      .cpuName = []() { return std::optional<std::string>(" Test CPU "); },
      .cpuLogicalThreadCount = []() { return 12u; },
  };

  const EnvironmentProbeResult result =
      collectEnvironmentProbe(repo, build, source);
  EXPECT_EQ(result.host.repoRoot, repo);
  EXPECT_EQ(result.host.commitHash, "abc123");
  EXPECT_EQ(result.host.branchName, "codex/tooling");
  EXPECT_TRUE(result.host.dirty);
  EXPECT_EQ(result.host.osName, "TestOS");
  EXPECT_EQ(result.host.osVersion, "1.2.3");
  EXPECT_EQ(result.host.cpuName, "Test CPU");
  EXPECT_EQ(result.host.cpuLogicalThreadCount, 12u);
  EXPECT_EQ(result.build.buildType, "Release");
  EXPECT_EQ(result.build.cmakeToolProfile, "contract");
  EXPECT_EQ(result.build.vcpkgManifestFeatures, "tools");
  EXPECT_TRUE(result.build.buildShared);
  EXPECT_TRUE(result.build.loggingEnabled);
  EXPECT_TRUE(result.build.tracyEnabled);
  EXPECT_TRUE(result.build.devChecks);
}

TEST(NuriToolCoreTest, EnvironmentProbeReportsUnknownWithoutProbeSources) {
  using namespace nuri::tools::core;
  const EnvironmentProbeResult result = collectEnvironmentProbe(
      "missing/repo", EnvironmentBuildFacts{}, EnvironmentProbeSource{});
  EXPECT_EQ(result.host.commitHash, "unknown");
  EXPECT_EQ(result.host.branchName, "unknown");
  EXPECT_FALSE(result.host.dirty);
  EXPECT_EQ(result.host.osName, "unknown");
  EXPECT_EQ(result.host.osVersion, "unknown");
  EXPECT_EQ(result.host.cpuName, "unknown");
  EXPECT_EQ(result.host.cpuLogicalThreadCount, 0u);
  EXPECT_EQ(result.build.buildType, "unknown");
  EXPECT_EQ(result.build.cmakeToolProfile, "unknown");
}

} // namespace
