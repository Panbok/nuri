#include "nuri/tools/core/run_workspace.h"

#include "nuri/tools/core/identifier.h"
#include "nuri/tools/core/safe_path.h"

#include <atomic>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

#if defined(_WIN32)
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#else
#include <unistd.h>
#endif

namespace nuri::tools::core {
namespace {

std::atomic<uint64_t> gRunSequence{0u};

[[nodiscard]] uint64_t processId() noexcept {
#if defined(_WIN32)
  return static_cast<uint64_t>(GetCurrentProcessId());
#else
  return static_cast<uint64_t>(getpid());
#endif
}

[[nodiscard]] std::string formatRunId(uint64_t sequence) {
  const auto now = std::chrono::system_clock::now();
  const auto milliseconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          now.time_since_epoch());
  const std::time_t time = std::chrono::system_clock::to_time_t(now);
  std::tm utc{};
#if defined(_WIN32)
  gmtime_s(&utc, &time);
#else
  gmtime_r(&time, &utc);
#endif
  std::ostringstream out;
  out << std::put_time(&utc, "%Y%m%dT%H%M%S") << '.' << std::setw(3)
      << std::setfill('0') << (milliseconds.count() % 1000) << "Z-p"
      << processId() << '-' << sequence;
  return out.str();
}

} // namespace

std::string createRunId() { return formatRunId(gRunSequence.fetch_add(1u)); }

Result<std::filesystem::path, std::string>
RunWorkspace::caseDirectory(std::string_view caseId) const {
  auto valid = validateIdentifier(caseId, "case id", IdentifierShape::Dotted);
  if (valid.hasError()) {
    return Result<std::filesystem::path, std::string>::makeError(valid.error());
  }
  return resolvePathUnder(root,
                          std::filesystem::path("cases") / std::string(caseId));
}

Result<RunWorkspace, std::string>
createRunWorkspace(const std::filesystem::path &toolArtifactRoot) {
  if (toolArtifactRoot.empty()) {
    return Result<RunWorkspace, std::string>::makeError(
        "tool artifact root must not be empty");
  }
  std::error_code error;
  std::filesystem::create_directories(toolArtifactRoot, error);
  if (error) {
    return Result<RunWorkspace, std::string>::makeError(
        "failed to create tool artifact root: " + error.message());
  }

  for (uint32_t attempt = 0u; attempt < 128u; ++attempt) {
    const std::string runId = createRunId();
    auto resolved = resolvePathUnder(toolArtifactRoot, runId);
    if (resolved.hasError()) {
      return Result<RunWorkspace, std::string>::makeError(resolved.error());
    }
    if (std::filesystem::create_directory(resolved.value(), error)) {
      return Result<RunWorkspace, std::string>::makeResult(
          RunWorkspace{.runId = runId, .root = resolved.value()});
    }
    if (error && error != std::errc::file_exists) {
      return Result<RunWorkspace, std::string>::makeError(
          "failed to create run workspace: " + error.message());
    }
    error.clear();
  }
  return Result<RunWorkspace, std::string>::makeError(
      "failed to allocate a unique run workspace");
}

} // namespace nuri::tools::core
