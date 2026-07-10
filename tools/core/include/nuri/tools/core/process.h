#pragma once

#include <chrono>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace nuri::tools::core {

enum class ProcessStatus {
  Exited,
  TimedOut,
  SpawnFailed,
  InternalError,
};

struct ProcessCommand {
  std::filesystem::path executable{};
  std::vector<std::string> arguments{};
};

struct ProcessOptions {
  std::optional<std::filesystem::path> workingDirectory{};
  std::chrono::milliseconds timeout = (std::chrono::milliseconds::max)();
};

struct ProcessResult {
  ProcessStatus status = ProcessStatus::SpawnFailed;
  std::optional<int> exitCode{};
  std::optional<int> terminationSignal{};
  std::string standardOutput{};
  std::string standardError{};
  std::string errorMessage{};
  std::chrono::milliseconds elapsed{};

  [[nodiscard]] bool exitedSuccessfully() const noexcept {
    return status == ProcessStatus::Exited && exitCode == 0;
  }
};

// Launches executable directly with the supplied argument vector. No command
// interpreter or shell expansion is involved.
[[nodiscard]] ProcessResult runProcess(const ProcessCommand &command,
                                       const ProcessOptions &options = {});

} // namespace nuri::tools::core
