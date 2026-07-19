#include "nuri/resources/storage/cache_utils.h"
#include "nuri/pch.h"
#include "nuri/resources/storage/binary_io.h"
#if defined(_WIN32)
#include <Windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
namespace nuri {

void Fnv1a64::add(std::span<const std::byte> bytes) noexcept {
  for (const std::byte byte : bytes) {
    value_ = (value_ ^ static_cast<uint8_t>(byte)) * 1099511628211ull;
  }
}

RandomAccessFile::~RandomAccessFile() { close(); }

bool RandomAccessFile::open(const std::filesystem::path &path) {
  close();
#if defined(_WIN32)
  const HANDLE handle =
      CreateFileW(path.c_str(), GENERIC_READ,
                  FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                  nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (handle == INVALID_HANDLE_VALUE) {
    return false;
  }
  file_ = reinterpret_cast<intptr_t>(handle);
  LARGE_INTEGER size{};
  if (!GetFileSizeEx(handle, &size) || size.QuadPart <= 0) {
    close();
    return false;
  }
  size_ = static_cast<size_t>(size.QuadPart);
#else
  file_ = ::open(path.c_str(), O_RDONLY);
  struct stat info{};
  if (file_ < 0 || ::fstat(static_cast<int>(file_), &info) != 0 ||
      info.st_size <= 0) {
    close();
    return false;
  }
  size_ = static_cast<size_t>(info.st_size);
#endif
  return true;
}

bool RandomAccessFile::readAt(uint64_t offset,
                              std::span<std::byte> destination) const {
  if (!binaryRangeValid(size_, offset, destination.size())) {
    return false;
  }
  size_t completed = 0;
  while (completed < destination.size()) {
    const size_t remaining = destination.size() - completed;
#if defined(_WIN32)
    const DWORD chunk = static_cast<DWORD>(
        std::min<size_t>(remaining, std::numeric_limits<DWORD>::max()));
    const uint64_t chunkOffset = offset + completed;
    OVERLAPPED overlapped{};
    overlapped.Offset = static_cast<DWORD>(chunkOffset);
    overlapped.OffsetHigh = static_cast<DWORD>(chunkOffset >> 32u);
    DWORD bytesRead = 0;
    if (!ReadFile(reinterpret_cast<HANDLE>(file_),
                  destination.data() + completed, chunk, &bytesRead,
                  &overlapped) ||
        bytesRead != chunk) {
      return false;
    }
#else
    const size_t chunk = std::min<size_t>(remaining, SSIZE_MAX);
    if (::pread(static_cast<int>(file_), destination.data() + completed, chunk,
                static_cast<off_t>(offset + completed)) !=
        static_cast<ssize_t>(chunk)) {
      return false;
    }
#endif
    completed += chunk;
  }
  return true;
}

void RandomAccessFile::close() noexcept {
  if (file_ < 0) {
    return;
  }
#if defined(_WIN32)
  CloseHandle(reinterpret_cast<HANDLE>(file_));
#else
  ::close(static_cast<int>(file_));
#endif
  file_ = -1;
  size_ = 0;
}

std::filesystem::path normalizeSourcePath(const std::filesystem::path &path) {
  std::error_code ec;
  auto normalized = std::filesystem::weakly_canonical(path, ec);
  if (ec) {
    normalized = std::filesystem::absolute(path, ec);
  }
  return (ec ? path : normalized).lexically_normal();
}

std::string hexU64(uint64_t value) { return std::format("{:016x}", value); }

SourceFingerprint querySourceFingerprint(const std::filesystem::path &path) {
  std::error_code ec;
  if (!std::filesystem::is_regular_file(path, ec) || ec) {
    return {};
  }
  const uint64_t size = std::filesystem::file_size(path, ec);
  if (ec) {
    return {};
  }
  const auto writeTime = std::filesystem::last_write_time(path, ec);
  if (ec) {
    return {};
  }
  return SourceFingerprint{
      .exists = true,
      .sizeBytes = size,
      .mtimeNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
                     writeTime.time_since_epoch())
                     .count(),
  };
}

std::filesystem::path temporarySiblingPath(const std::filesystem::path &path) {
  static std::atomic<uint64_t> counter{0};
  auto temporary = path;
  temporary += std::format(
      ".tmp.{:x}.{}", std::hash<std::thread::id>{}(std::this_thread::get_id()),
      counter.fetch_add(1, std::memory_order_relaxed));
  return temporary;
}

bool replaceFileAtomic(const std::filesystem::path &temporaryPath,
                       const std::filesystem::path &destinationPath) noexcept {
#if defined(_WIN32)
  if (MoveFileExW(temporaryPath.c_str(), destinationPath.c_str(),
                  MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
    return true;
  }
  static std::atomic<uint64_t> counter{0};
  auto retired = destinationPath;
  retired += std::format(".retired.{}",
                         counter.fetch_add(1, std::memory_order_relaxed));
  if (!MoveFileExW(destinationPath.c_str(), retired.c_str(),
                   MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
    return false;
  }
  if (MoveFileExW(temporaryPath.c_str(), destinationPath.c_str(),
                  MOVEFILE_WRITE_THROUGH)) {
    DeleteFileW(retired.c_str());
    return true;
  }
  MoveFileExW(retired.c_str(), destinationPath.c_str(),
              MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH);
  return false;
#else
  std::error_code ec;
  std::filesystem::rename(temporaryPath, destinationPath, ec);
  return !ec;
#endif
}

Result<std::vector<std::byte>, std::string>
readBinaryFile(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  const auto size = input.tellg();
  if (!input || size < 0) {
    return Result<std::vector<std::byte>, std::string>::makeError(
        "failed to open binary file '" + path.string() + "'");
  }
  std::vector<std::byte> bytes(static_cast<size_t>(size));
  input.seekg(0);
  input.read(reinterpret_cast<char *>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  if (!input) {
    return Result<std::vector<std::byte>, std::string>::makeError(
        "failed to read binary file '" + path.string() + "'");
  }
  return Result<std::vector<std::byte>, std::string>::makeResult(
      std::move(bytes));
}

Result<bool, std::string>
writeBinaryFileAtomic(const std::filesystem::path &path,
                      std::span<const std::byte> bytes) {
  if (path.empty()) {
    return Result<bool, std::string>::makeError("empty binary destination");
  }
  std::error_code ec;
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path(), ec);
  }
  if (ec) {
    return Result<bool, std::string>::makeError(
        "failed to prepare binary destination '" + path.string() + "'");
  }
  const auto temp = temporarySiblingPath(path);
  {
    std::ofstream output(temp, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char *>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    if (!output) {
      std::filesystem::remove(temp, ec);
      return Result<bool, std::string>::makeError(
          "failed to write binary file '" + path.string() + "'");
    }
  }
  if (!replaceFileAtomic(temp, path)) {
    std::filesystem::remove(temp, ec);
    return Result<bool, std::string>::makeError(
        "failed to publish binary file '" + path.string() + "'");
  }
  return Result<bool, std::string>::makeResult(true);
}

} // namespace nuri
