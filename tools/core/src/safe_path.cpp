#include "nuri/tools/core/safe_path.h"

#include <algorithm>
#include <cwctype>
#include <system_error>

#if defined(_WIN32)
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#endif

namespace nuri::tools::core {
namespace {

[[nodiscard]] bool componentEquals(const std::filesystem::path &lhs,
                                   const std::filesystem::path &rhs) {
#if defined(_WIN32)
  std::wstring left = lhs.native();
  std::wstring right = rhs.native();
  std::transform(left.begin(), left.end(), left.begin(),
                 [](wchar_t value) { return std::towlower(value); });
  std::transform(right.begin(), right.end(), right.begin(),
                 [](wchar_t value) { return std::towlower(value); });
  return left == right;
#else
  return lhs == rhs;
#endif
}

[[nodiscard]] bool isContained(const std::filesystem::path &root,
                               const std::filesystem::path &candidate) {
  auto rootIt = root.begin();
  auto candidateIt = candidate.begin();
  for (; rootIt != root.end(); ++rootIt, ++candidateIt) {
    if (candidateIt == candidate.end() ||
        !componentEquals(*rootIt, *candidateIt)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool isReparseOrSymlink(const std::filesystem::path &path,
                                      std::error_code &error) {
#if defined(_WIN32)
  const DWORD attributes = GetFileAttributesW(path.c_str());
  if (attributes == INVALID_FILE_ATTRIBUTES) {
    const DWORD lastError = GetLastError();
    if (lastError == ERROR_FILE_NOT_FOUND ||
        lastError == ERROR_PATH_NOT_FOUND) {
      error.clear();
      return false;
    }
    error =
        std::error_code(static_cast<int>(lastError), std::system_category());
    return false;
  }
  error.clear();
  return (attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0u;
#else
  const std::filesystem::file_status status =
      std::filesystem::symlink_status(path, error);
  return !error && std::filesystem::is_symlink(status);
#endif
}

[[nodiscard]] Result<void, std::string>
rejectReparseComponents(const std::filesystem::path &root,
                        const std::filesystem::path &candidate) {
  std::filesystem::path current;
  for (const auto &component : candidate) {
    current /= component;
    if (!isContained(root, current) && current != root) {
      continue;
    }
    std::error_code error;
    if (isReparseOrSymlink(current, error)) {
      return Result<void, std::string>::makeError(
          "path contains a symlink, junction, or reparse point: " +
          current.string());
    }
    if (error) {
      return Result<void, std::string>::makeError(
          "failed to inspect path component: " + current.string() + ": " +
          error.message());
    }
  }
  return Result<void, std::string>::makeResult();
}

} // namespace

Result<std::filesystem::path, std::string>
resolvePathUnder(const std::filesystem::path &root,
                 const std::filesystem::path &relative) {
  if (root.empty()) {
    return Result<std::filesystem::path, std::string>::makeError(
        "path root must not be empty");
  }
  if (relative.empty() || relative.is_absolute() || relative.has_root_path()) {
    return Result<std::filesystem::path, std::string>::makeError(
        "path must be a non-empty relative path");
  }
  for (const auto &component : relative) {
    if (component.empty() || component == "." || component == "..") {
      return Result<std::filesystem::path, std::string>::makeError(
          "path contains an unsafe component");
    }
  }

  std::error_code error;
  const std::filesystem::path absoluteRoot =
      std::filesystem::absolute(root, error);
  if (error) {
    return Result<std::filesystem::path, std::string>::makeError(
        "failed to resolve path root: " + error.message());
  }
  const std::filesystem::path canonicalRoot =
      std::filesystem::weakly_canonical(absoluteRoot, error);
  if (error) {
    return Result<std::filesystem::path, std::string>::makeError(
        "failed to canonicalize path root: " + error.message());
  }
  const std::filesystem::path candidate =
      (canonicalRoot / relative).lexically_normal();
  if (!isContained(canonicalRoot, candidate) || candidate == canonicalRoot) {
    return Result<std::filesystem::path, std::string>::makeError(
        "resolved path escapes or aliases its root");
  }
  auto reparseCheck = rejectReparseComponents(canonicalRoot, candidate);
  if (reparseCheck.hasError()) {
    return Result<std::filesystem::path, std::string>::makeError(
        reparseCheck.error());
  }
  return Result<std::filesystem::path, std::string>::makeResult(candidate);
}

Result<uintmax_t, std::string>
removeTreeUnder(const std::filesystem::path &root,
                const std::filesystem::path &relative) {
  auto resolved = resolvePathUnder(root, relative);
  if (resolved.hasError()) {
    return Result<uintmax_t, std::string>::makeError(resolved.error());
  }
  auto finalCheck = resolvePathUnder(root, relative);
  if (finalCheck.hasError() || finalCheck.value() != resolved.value()) {
    return Result<uintmax_t, std::string>::makeError(
        "path changed during containment validation");
  }
  std::error_code error;
  const uintmax_t removed =
      std::filesystem::remove_all(resolved.value(), error);
  if (error) {
    return Result<uintmax_t, std::string>::makeError(
        "failed to remove confined tree: " + error.message());
  }
  return Result<uintmax_t, std::string>::makeResult(removed);
}

} // namespace nuri::tools::core
