#include "nuri/tools/core/process.h"
#include "nuri_process_test_helper_path.h"

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

namespace {

using namespace std::chrono_literals;

[[nodiscard]] std::filesystem::path helperPath() {
  return std::filesystem::path(NURI_PROCESS_TEST_HELPER_PATH);
}

[[nodiscard]] std::string encodedArgument(const size_t index,
                                          const std::string &argument) {
  return std::to_string(index) + ':' + std::to_string(argument.size()) + ':' +
         argument + '\n';
}

TEST(NuriProcessTest, PreservesArgumentsWithoutShellExpansion) {
  using namespace nuri::tools::core;
  const std::vector<std::string> values{
      "plain",
      "argument with spaces",
      "\"quoted\"",
      "",
      ";&|<>$()^",
      "utf8-\xD0\xBF\xD1\x80\xD0\xBE\xD1\x86\xD0\xB5\xD1\x81\xD1\x81"};
  ProcessCommand command{.executable = helperPath(), .arguments = {"echo"}};
  command.arguments.insert(command.arguments.end(), values.begin(),
                           values.end());

  const ProcessResult result = runProcess(command, {.timeout = 5s});

  ASSERT_EQ(result.status, ProcessStatus::Exited) << result.errorMessage;
  ASSERT_EQ(result.exitCode, 0);
  std::string expected;
  for (size_t i = 0u; i < values.size(); ++i) {
    expected += encodedArgument(i, values[i]);
  }
  EXPECT_EQ(result.standardOutput, expected);
  EXPECT_EQ(result.standardError, expected);
  EXPECT_TRUE(result.errorMessage.empty());
  EXPECT_TRUE(result.exitedSuccessfully());
}

TEST(NuriProcessTest, TreatsSpecialCharacterExecutablePathAsLiteral) {
  using namespace nuri::tools::core;
  const std::filesystem::path workspace =
      std::filesystem::temp_directory_path() /
      ("nuri process & literal " +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  ASSERT_TRUE(std::filesystem::create_directory(workspace));
  const std::filesystem::path copiedHelper =
      workspace / ("helper (copy)&literal" + helperPath().extension().string());
  ASSERT_TRUE(std::filesystem::copy_file(helperPath(), copiedHelper));

  const ProcessResult result =
      runProcess({.executable = copiedHelper,
                  .arguments = {"echo", "trace path & (literal).tracy"}},
                 {.timeout = 5s});
  std::error_code cleanupError;
  std::filesystem::remove_all(workspace, cleanupError);

  ASSERT_EQ(result.status, ProcessStatus::Exited) << result.errorMessage;
  ASSERT_EQ(result.exitCode, 0);
  EXPECT_EQ(result.standardOutput,
            encodedArgument(0u, "trace path & (literal).tracy"));
  EXPECT_EQ(result.standardError,
            encodedArgument(0u, "trace path & (literal).tracy"));
  EXPECT_FALSE(cleanupError);
}

TEST(NuriProcessTest, ReportsNonzeroExitAndKeepsBothCaptures) {
  using namespace nuri::tools::core;
  const ProcessResult result =
      runProcess({.executable = helperPath(), .arguments = {"exit", "7"}},
                 {.timeout = 5s});

  ASSERT_EQ(result.status, ProcessStatus::Exited) << result.errorMessage;
  EXPECT_EQ(result.exitCode, 7);
  EXPECT_FALSE(result.terminationSignal.has_value());
  EXPECT_EQ(result.standardOutput, "before-exit\n");
  EXPECT_EQ(result.standardError, "exit-diagnostic\n");
  EXPECT_FALSE(result.exitedSuccessfully());
}

TEST(NuriProcessTest, ReportsSpawnFailureSeparatelyFromChildExit) {
  using namespace nuri::tools::core;
  const ProcessResult result =
      runProcess({.executable = helperPath().parent_path() / "does-not-exist",
                  .arguments = {}},
                 {.timeout = 5s});

  EXPECT_EQ(result.status, ProcessStatus::SpawnFailed);
  EXPECT_FALSE(result.exitCode.has_value());
  EXPECT_FALSE(result.errorMessage.empty());
}

TEST(NuriProcessTest, TerminatesAtTimeoutAndPreservesPriorOutput) {
  using namespace nuri::tools::core;
  const ProcessResult result =
      runProcess({.executable = helperPath(), .arguments = {"sleep", "5000"}},
                 {.timeout = 50ms});

  EXPECT_EQ(result.status, ProcessStatus::TimedOut) << result.errorMessage;
  EXPECT_FALSE(result.exitCode.has_value());
  EXPECT_EQ(result.standardOutput, "before-sleep\n");
  EXPECT_EQ(result.standardError, "sleep-diagnostic\n");
  EXPECT_LT(result.elapsed, 2s);
}

TEST(NuriProcessTest, AppliesWorkingDirectoryWithoutChangingParent) {
  using namespace nuri::tools::core;
  const std::filesystem::path original = std::filesystem::current_path();
  const std::filesystem::path workingDirectory =
      std::filesystem::temp_directory_path();
  const ProcessResult result =
      runProcess({.executable = helperPath(), .arguments = {"cwd"}},
                 {.workingDirectory = workingDirectory, .timeout = 5s});

  ASSERT_EQ(result.status, ProcessStatus::Exited) << result.errorMessage;
  EXPECT_TRUE(std::filesystem::equivalent(
      std::filesystem::path(result.standardOutput), workingDirectory));
  EXPECT_EQ(std::filesystem::current_path(), original);
}

TEST(NuriProcessTest, DrainsLargeStdoutAndStderrWithoutDeadlock) {
  using namespace nuri::tools::core;
  constexpr size_t kBytes = 256u * 1024u;
  const ProcessResult result =
      runProcess({.executable = helperPath(),
                  .arguments = {"bulk", std::to_string(kBytes)}},
                 {.timeout = 5s});

  ASSERT_EQ(result.status, ProcessStatus::Exited) << result.errorMessage;
  EXPECT_EQ(result.exitCode, 0);
  EXPECT_EQ(result.standardOutput, std::string(kBytes, 'o'));
  EXPECT_EQ(result.standardError, std::string(kBytes, 'e'));
}

} // namespace
