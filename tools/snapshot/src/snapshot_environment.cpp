#include "nuri/tools/snapshot/snapshot_environment.h"

#include "nuri/tools/snapshot/build_config.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <format>
#include <fstream>
#include <sstream>
#include <thread>

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/utsname.h>
#endif

namespace nuri::tools::snapshot {
namespace {

[[nodiscard]] std::string trim(std::string text) {
  while (!text.empty() && (text.back() == '\n' || text.back() == '\r' ||
                           text.back() == ' ' || text.back() == '\t')) {
    text.pop_back();
  }
  size_t first = 0u;
  while (first < text.size() && (text[first] == ' ' || text[first] == '\t' ||
                                 text[first] == '\n' || text[first] == '\r')) {
    ++first;
  }
  return text.substr(first);
}

[[nodiscard]] std::string runCommand(std::string_view command,
                                     const std::filesystem::path &cwd) {
  std::string full =
      "cd /d \"" + cwd.string() + "\" && " + std::string(command) + " 2>nul";
#if !defined(_WIN32)
  full =
      "cd \"" + cwd.string() + "\" && " + std::string(command) + " 2>/dev/null";
#endif
  std::array<char, 256> buffer{};
#if defined(_WIN32)
  FILE *pipe = _popen(full.c_str(), "r");
#else
  FILE *pipe = popen(full.c_str(), "r");
#endif
  if (pipe == nullptr) {
    return "unknown";
  }
  std::string output;
  while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) !=
         nullptr) {
    output += buffer.data();
  }
#if defined(_WIN32)
  (void)_pclose(pipe);
#else
  (void)pclose(pipe);
#endif
  output = trim(output);
  return output.empty() ? std::string("unknown") : output;
}

[[nodiscard]] std::string detectOsName() {
#if defined(_WIN32)
  return "Windows";
#elif defined(__APPLE__)
  return "macOS";
#elif defined(__linux__)
  return "Linux";
#else
  return "unknown";
#endif
}

[[nodiscard]] std::string detectOsVersion() {
#if defined(_WIN32)
  using RtlGetVersionFn = LONG(WINAPI *)(OSVERSIONINFOW *);
  HMODULE ntdll = GetModuleHandleA("ntdll.dll");
  if (ntdll != nullptr) {
    auto rtlGetVersion = reinterpret_cast<RtlGetVersionFn>(
        GetProcAddress(ntdll, "RtlGetVersion"));
    if (rtlGetVersion != nullptr) {
      OSVERSIONINFOW version{};
      version.dwOSVersionInfoSize = sizeof(version);
      if (rtlGetVersion(&version) == 0) {
        return std::format("{}.{}.{}", version.dwMajorVersion,
                           version.dwMinorVersion, version.dwBuildNumber);
      }
    }
  }
  return "unknown";
#else
  utsname name{};
  if (uname(&name) == 0) {
    return name.release;
  }
  return "unknown";
#endif
}

[[nodiscard]] std::string detectCpuName() {
#if defined(_WIN32)
  char buffer[256]{};
  DWORD size = sizeof(buffer);
  if (RegGetValueA(HKEY_LOCAL_MACHINE,
                   "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                   "ProcessorNameString", RRF_RT_REG_SZ, nullptr, buffer,
                   &size) == ERROR_SUCCESS) {
    return trim(buffer);
  }
  return "unknown";
#elif defined(__linux__)
  std::ifstream cpuInfo("/proc/cpuinfo");
  std::string line;
  while (std::getline(cpuInfo, line)) {
    constexpr std::string_view kPrefix = "model name";
    if (line.rfind(kPrefix, 0) == 0) {
      const size_t colon = line.find(':');
      if (colon != std::string::npos) {
        return trim(line.substr(colon + 1u));
      }
    }
  }
  return "unknown";
#else
  return "unknown";
#endif
}

[[nodiscard]] bool macroBool(int value) { return value != 0; }

} // namespace

std::filesystem::path snapshotRepoRoot() {
#if defined(PROJECT_SOURCE_DIR)
  return std::filesystem::path(PROJECT_SOURCE_DIR).lexically_normal();
#else
  return std::filesystem::current_path();
#endif
}

std::string readProcessEnvironment(std::string_view name) {
  std::string key(name);
#if defined(_WIN32)
  size_t size = 0u;
  getenv_s(&size, nullptr, 0u, key.c_str());
  if (size == 0u) {
    return {};
  }
  std::string value(size, '\0');
  getenv_s(&size, value.data(), value.size(), key.c_str());
  if (!value.empty() && value.back() == '\0') {
    value.pop_back();
  }
  return value;
#else
  const char *value = std::getenv(key.c_str());
  return value != nullptr ? std::string(value) : std::string();
#endif
}

SnapshotEnvironment collectSnapshotEnvironment(
    std::string_view backend, std::string_view backendSource,
    std::string_view requestedPresentMode, std::string_view presentModeSource,
    std::string_view requestedWindowMode, std::string_view resolvedWindowMode) {
  SnapshotEnvironment env{};
  env.repoRoot = snapshotRepoRoot();
  env.commitHash = runCommand("git rev-parse HEAD", env.repoRoot);
  env.branchName = runCommand("git rev-parse --abbrev-ref HEAD", env.repoRoot);
  const std::string dirty = runCommand("git status --porcelain", env.repoRoot);
  env.dirty = dirty != "unknown" && !dirty.empty();
  env.osName = detectOsName();
  env.osVersion = detectOsVersion();
  env.cpuName = detectCpuName();
  env.cpuLogicalThreadCount =
      static_cast<uint32_t>(std::thread::hardware_concurrency());
  env.gpuBackend = std::string(backend);
  env.gpuBackendSource = std::string(backendSource);
  env.requestedPresentMode = std::string(requestedPresentMode);
  env.resolvedPresentMode = std::string(requestedPresentMode);
  env.presentModeSource = std::string(presentModeSource);
  env.requestedWindowMode = std::string(requestedWindowMode);
  env.resolvedWindowMode = std::string(resolvedWindowMode);
  env.windowVisible = resolvedWindowMode == "visible";
  env.cmakeToolProfile = NURI_SNAPSHOT_TOOL_PROFILE;
  env.vcpkgManifestFeatures = NURI_SNAPSHOT_VCPKG_MANIFEST_FEATURES;
  env.buildType = NURI_SNAPSHOT_BUILD_TYPE;
  env.buildShared = macroBool(NURI_SNAPSHOT_BUILD_SHARED);
  env.loggingEnabled = macroBool(NURI_SNAPSHOT_WITH_LOGGING);
  env.assertsEnabled = macroBool(NURI_SNAPSHOT_WITH_ASSERTS);
  env.tracyEnabled = macroBool(NURI_SNAPSHOT_WITH_TRACY);
  env.tracyGpuEnabled = macroBool(NURI_SNAPSHOT_WITH_TRACY_GPU);
  env.tracyGpuDrawZonesEnabled =
      macroBool(NURI_SNAPSHOT_WITH_TRACY_GPU_DRAW_ZONES);
  env.devChecks = macroBool(NURI_SNAPSHOT_DEV_CHECKS);
  return env;
}

std::string joinCommandLine(int argc, char **argv) {
  std::ostringstream out;
  for (int i = 0; i < argc; ++i) {
    if (i != 0) {
      out << ' ';
    }
    std::string arg = argv[i] != nullptr ? argv[i] : "";
    const bool needsQuotes = arg.find(' ') != std::string::npos;
    if (needsQuotes) {
      out << '"';
    }
    out << arg;
    if (needsQuotes) {
      out << '"';
    }
  }
  return out.str();
}

std::string utcTimestampIso8601() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t nowTime = std::chrono::system_clock::to_time_t(now);
  std::tm utc{};
#if defined(_WIN32)
  gmtime_s(&utc, &nowTime);
#else
  gmtime_r(&nowTime, &utc);
#endif
  char buffer[32]{};
  std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &utc);
  return buffer;
}

std::string utcTimestampForPath() {
  std::string text = utcTimestampIso8601();
  for (char &c : text) {
    if (c == ':' || c == '-') {
      c = '\0';
    }
  }
  std::string out;
  out.reserve(text.size());
  for (const char c : text) {
    if (c != '\0') {
      out.push_back(c);
    }
  }
  return out;
}

} // namespace nuri::tools::snapshot
