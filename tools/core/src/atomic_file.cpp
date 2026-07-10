#include "nuri/tools/core/atomic_file.h"

#include <atomic>
#include <chrono>
#include <fstream>
#include <system_error>

#if defined(_WIN32)
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#endif

namespace nuri::tools::core {
namespace {

std::atomic<uint64_t> gTemporaryFileSequence{0u};

[[nodiscard]] std::filesystem::path
temporarySibling(const std::filesystem::path &path) {
  const uint64_t tick = static_cast<uint64_t>(
      std::chrono::steady_clock::now().time_since_epoch().count());
  const uint64_t sequence = gTemporaryFileSequence.fetch_add(1u);
  std::filesystem::path temporary = path;
  temporary += ".tmp." + std::to_string(tick) + "." +
               std::to_string(sequence);
  return temporary;
}

#if defined(_WIN32)
[[nodiscard]] Result<std::filesystem::path, std::string>
extendedLengthPath(const std::filesystem::path &path) {
  std::error_code error;
  const std::filesystem::path absolute = std::filesystem::absolute(path, error);
  if (error) {
    return Result<std::filesystem::path, std::string>::makeError(
        "atomicWriteTextFile: failed to resolve path: " + error.message());
  }
  const std::wstring native = absolute.native();
  if (native.starts_with(L"\\\\?\\")) {
    return Result<std::filesystem::path, std::string>::makeResult(absolute);
  }
  if (native.starts_with(L"\\\\")) {
    return Result<std::filesystem::path, std::string>::makeResult(
        std::filesystem::path(L"\\\\?\\UNC\\" + native.substr(2u)));
  }
  return Result<std::filesystem::path, std::string>::makeResult(
      std::filesystem::path(L"\\\\?\\" + native));
}
#endif

} // namespace

Result<void, std::string>
atomicWriteTextFile(const std::filesystem::path &path, std::string_view text) {
  if (path.empty()) {
    return Result<void, std::string>::makeError(
        "atomicWriteTextFile: path must not be empty");
  }
  std::error_code error;
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path(), error);
    if (error) {
      return Result<void, std::string>::makeError(
          "atomicWriteTextFile: failed to create parent: " + error.message());
    }
  }

  const std::filesystem::path temporary = temporarySibling(path);
#if defined(_WIN32)
  auto nativeTemporary = extendedLengthPath(temporary);
  auto nativeTarget = extendedLengthPath(path);
  if (nativeTemporary.hasError() || nativeTarget.hasError()) {
    return Result<void, std::string>::makeError(
        nativeTemporary.hasError() ? nativeTemporary.error()
                                   : nativeTarget.error());
  }
#endif
  {
#if defined(_WIN32)
    std::ofstream file(nativeTemporary.value(),
                       std::ios::binary | std::ios::trunc);
#else
    std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
#endif
    if (!file) {
      return Result<void, std::string>::makeError(
          "atomicWriteTextFile: failed to open temporary file");
    }
    file.write(text.data(), static_cast<std::streamsize>(text.size()));
    file.flush();
    if (!file) {
      file.close();
#if defined(_WIN32)
      DeleteFileW(nativeTemporary.value().c_str());
#else
      std::filesystem::remove(temporary, error);
#endif
      return Result<void, std::string>::makeError(
          "atomicWriteTextFile: failed to write temporary file");
    }
  }

#if defined(_WIN32)
  if (!MoveFileExW(nativeTemporary.value().c_str(), nativeTarget.value().c_str(),
                   MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
    const DWORD lastError = GetLastError();
    DeleteFileW(nativeTemporary.value().c_str());
    return Result<void, std::string>::makeError(
        "atomicWriteTextFile: replacement failed: " +
        std::error_code(static_cast<int>(lastError), std::system_category())
            .message());
  }
#else
  std::filesystem::rename(temporary, path, error);
  if (error) {
    std::filesystem::remove(temporary, error);
    return Result<void, std::string>::makeError(
        "atomicWriteTextFile: replacement failed: " + error.message());
  }
#endif
  return Result<void, std::string>::makeResult();
}

} // namespace nuri::tools::core
