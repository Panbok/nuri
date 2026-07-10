#include "nuri/tools/core/environment_probe.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <format>
#include <fstream>
#include <sstream>
#include <thread>
#include <utility>

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/utsname.h>
#endif

namespace nuri::tools::core {
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

[[nodiscard]] std::optional<std::string>
runCommand(std::string_view command, const std::filesystem::path &cwd) {
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
    return std::nullopt;
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
  return output;
}

[[nodiscard]] std::optional<std::string> detectOsName() {
#if defined(_WIN32)
  return "Windows";
#elif defined(__APPLE__)
  return "macOS";
#elif defined(__linux__)
  return "Linux";
#else
  return std::nullopt;
#endif
}

[[nodiscard]] std::optional<std::string> detectOsVersion() {
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
  return std::nullopt;
#else
  utsname name{};
  if (uname(&name) == 0) {
    return name.release;
  }
  return std::nullopt;
#endif
}

[[nodiscard]] std::optional<std::string> detectCpuName() {
#if defined(_WIN32)
  char buffer[256]{};
  DWORD size = sizeof(buffer);
  if (RegGetValueA(HKEY_LOCAL_MACHINE,
                   "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                   "ProcessorNameString", RRF_RT_REG_SZ, nullptr, buffer,
                   &size) == ERROR_SUCCESS) {
    return buffer;
  }
  return std::nullopt;
#elif defined(__linux__)
  std::ifstream cpuInfo("/proc/cpuinfo");
  std::string line;
  while (std::getline(cpuInfo, line)) {
    constexpr std::string_view kPrefix = "model name";
    if (line.rfind(kPrefix, 0) == 0) {
      const size_t colon = line.find(':');
      if (colon != std::string::npos) {
        return line.substr(colon + 1u);
      }
    }
  }
  return std::nullopt;
#else
  return std::nullopt;
#endif
}

[[nodiscard]] std::string
valueOrUnknown(const std::optional<std::string> &value) {
  if (!value.has_value()) {
    return "unknown";
  }
  std::string normalized = trim(*value);
  return normalized.empty() ? std::string("unknown") : normalized;
}

} // namespace

EnvironmentProbeSource defaultEnvironmentProbeSource() {
  return EnvironmentProbeSource{
      .runCommand = runCommand,
      .osName = detectOsName,
      .osVersion = detectOsVersion,
      .cpuName = detectCpuName,
      .cpuLogicalThreadCount =
          []() {
            return static_cast<uint32_t>(std::thread::hardware_concurrency());
          },
  };
}

EnvironmentProbeResult
collectEnvironmentProbe(const std::filesystem::path &repoRoot,
                        EnvironmentBuildFacts build) {
  return collectEnvironmentProbe(repoRoot, std::move(build),
                                 defaultEnvironmentProbeSource());
}

EnvironmentProbeResult
collectEnvironmentProbe(const std::filesystem::path &repoRoot,
                        EnvironmentBuildFacts build,
                        const EnvironmentProbeSource &source) {
  EnvironmentProbeResult result{};
  result.host.repoRoot = repoRoot;
  result.build = std::move(build);

  const auto command = [&](std::string_view text) {
    return source.runCommand ? source.runCommand(text, repoRoot)
                             : std::optional<std::string>{};
  };
  result.host.commitHash = valueOrUnknown(command("git rev-parse HEAD"));
  result.host.branchName =
      valueOrUnknown(command("git rev-parse --abbrev-ref HEAD"));
  const std::optional<std::string> status = command("git status --porcelain");
  if (status.has_value()) {
    const std::string normalized = trim(*status);
    result.host.dirty = !normalized.empty() && normalized != "unknown";
  }
  result.host.osName =
      valueOrUnknown(source.osName ? source.osName() : std::nullopt);
  result.host.osVersion =
      valueOrUnknown(source.osVersion ? source.osVersion() : std::nullopt);
  result.host.cpuName =
      valueOrUnknown(source.cpuName ? source.cpuName() : std::nullopt);
  result.host.cpuLogicalThreadCount =
      source.cpuLogicalThreadCount ? source.cpuLogicalThreadCount() : 0u;
  return result;
}

std::string readEnvironmentVariable(std::string_view name) {
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

} // namespace nuri::tools::core
