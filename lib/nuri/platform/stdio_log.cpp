#include "nuri/platform/stdio_log.h"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <limits>

#if defined(_WIN32)
#include <io.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace nuri {

namespace {

[[nodiscard]] constexpr const char *levelName(LogLevel level) noexcept {
  switch (level) {
  case LogLevel::Trace:
    return "trace";
  case LogLevel::Debug:
    return "debug";
  case LogLevel::Info:
    return "info";
  case LogLevel::Warning:
    return "warning";
  case LogLevel::Error:
    return "error";
  case LogLevel::Fatal:
    return "fatal";
  }
  return "unknown";
}

[[nodiscard]] constexpr bool includes(LogLevel threshold,
                                      LogLevel level) noexcept {
  return static_cast<uint8_t>(level) >= static_cast<uint8_t>(threshold);
}

[[nodiscard]] constexpr const char *colorCode(LogLevel level) noexcept {
  switch (level) {
  case LogLevel::Trace:
    return "\x1b[90m";
  case LogLevel::Debug:
    return "\x1b[36m";
  case LogLevel::Info:
    return "\x1b[32m";
  case LogLevel::Warning:
    return "\x1b[33m";
  case LogLevel::Error:
    return "\x1b[31m";
  case LogLevel::Fatal:
    return "\x1b[1;91m";
  }
  return "";
}

[[nodiscard]] bool enableTerminalColor(std::FILE *stream) noexcept {
  if (stream == nullptr) {
    return false;
  }
#if defined(_WIN32)
  const int descriptor = _fileno(stream);
  if (descriptor < 0) {
    return false;
  }
  const intptr_t osHandle = _get_osfhandle(descriptor);
  if (osHandle == -1) {
    return false;
  }
  const HANDLE handle = reinterpret_cast<HANDLE>(osHandle);
  DWORD mode = 0u;
  if (GetConsoleMode(handle, &mode) == 0) {
    return false;
  }
  if ((mode & ENABLE_VIRTUAL_TERMINAL_PROCESSING) == 0u &&
      SetConsoleMode(handle, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING) == 0) {
    return false;
  }
  return true;
#else
  return ::isatty(::fileno(stream)) != 0;
#endif
}

struct LocalTimestamp {
  std::tm calendar{};
  uint32_t milliseconds = 0u;
};

[[nodiscard]] LocalTimestamp localTimestamp() noexcept {
  const auto now = std::chrono::system_clock::now();
  const auto milliseconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          now.time_since_epoch());
  const std::time_t time = std::chrono::system_clock::to_time_t(now);

  LocalTimestamp timestamp{
      .milliseconds = static_cast<uint32_t>(milliseconds.count() % 1000),
  };
#if defined(_WIN32)
  (void)localtime_s(&timestamp.calendar, &time);
#else
  (void)localtime_r(&time, &timestamp.calendar);
#endif
  return timestamp;
}

void writeLine(std::FILE *stream, LogLevel level, std::string_view message,
               bool forceFlush, bool colored) noexcept {
  if (stream == nullptr) {
    return;
  }

  const LocalTimestamp timestamp = localTimestamp();
  const int length = static_cast<int>(std::min(
      message.size(), static_cast<size_t>(std::numeric_limits<int>::max())));
  if (colored) {
    std::fputs(colorCode(level), stream);
  }
  std::fprintf(stream, "%02d:%02d:%02d.%03u %-7s %.*s",
               timestamp.calendar.tm_hour, timestamp.calendar.tm_min,
               timestamp.calendar.tm_sec, timestamp.milliseconds,
               levelName(level), length, message.data());
  if (colored) {
    std::fputs("\x1b[0m", stream);
  }
  std::fputc('\n', stream);
  if (forceFlush) {
    std::fflush(stream);
  }
}

} // namespace

StdioLog::StdioLog(const LogConfig &config)
    : config_(config),
      stdoutColor_(config.coloredConsole && enableTerminalColor(stdout)),
      stderrColor_(config.coloredConsole && enableTerminalColor(stderr)) {
  if (config_.filePath.empty()) {
    initialized_ = true;
    return;
  }

#if defined(_WIN32)
  initialized_ = fopen_s(&file_, config_.filePath.c_str(), "wb") == 0;
#else
  file_ = std::fopen(config_.filePath.c_str(), "wb");
  initialized_ = file_ != nullptr;
#endif
}

StdioLog::~StdioLog() {
  std::scoped_lock lock(mutex_);
  if (file_ != nullptr) {
    std::fclose(file_);
    file_ = nullptr;
  }
}

std::unique_ptr<StdioLog> StdioLog::create(const LogConfig &config) {
  auto log = std::unique_ptr<StdioLog>(new StdioLog(config));
  if (!log->initialized_) {
    return nullptr;
  }
  return log;
}

void StdioLog::write(LogLevel level, std::string_view message) {
  std::scoped_lock lock(mutex_);
  if (includes(config_.logLevel, level)) {
    writeLine(file_, level, message, config_.forceFlush, false);
  }
  if (includes(config_.consoleLevel, level)) {
    std::FILE *console = level >= LogLevel::Error ? stderr : stdout;
    const bool colored = console == stderr ? stderrColor_ : stdoutColor_;
    writeLine(console, level, message, config_.forceFlush, colored);
  }
}

std::unique_ptr<Log> Log::create() {
  LogConfig config{};
  return StdioLog::create(config);
}

std::unique_ptr<Log> Log::create(const LogConfig &config) {
  return StdioLog::create(config);
}

} // namespace nuri
