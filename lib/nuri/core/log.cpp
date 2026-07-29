#include "nuri/core/log.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <ctime>
#include <limits>
#include <memory>
#include <mutex>
#if defined(_WIN32)
#include <io.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace nuri {
namespace {
[[nodiscard]] constexpr const char *levelName(LogLevel level) noexcept {
  constexpr std::array names{"trace",   "debug", "info",
                             "warning", "error", "fatal"};
  return names[static_cast<size_t>(level)];
}

[[nodiscard]] constexpr const char *colorCode(LogLevel level) noexcept {
  constexpr std::array colors{"\x1b[90m", "\x1b[36m", "\x1b[32m",
                              "\x1b[33m", "\x1b[31m", "\x1b[1;91m"};
  return colors[static_cast<size_t>(level)];
}

[[nodiscard]] constexpr bool includes(LogLevel threshold,
                                      LogLevel level) noexcept {
  return static_cast<uint8_t>(level) >= static_cast<uint8_t>(threshold);
}

[[nodiscard]] bool enableTerminalColor(std::FILE *stream) noexcept {
  if (!stream) {
    return false;
  }
#if defined(_WIN32)
  const int descriptor = _fileno(stream);
  const intptr_t osHandle = descriptor < 0 ? -1 : _get_osfhandle(descriptor);
  if (osHandle == -1) {
    return false;
  }
  const HANDLE handle = reinterpret_cast<HANDLE>(osHandle);
  DWORD mode = 0u;
  return GetConsoleMode(handle, &mode) != 0 &&
         (((mode & ENABLE_VIRTUAL_TERMINAL_PROCESSING) != 0u) ||
          SetConsoleMode(handle, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING) !=
              0);
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
  const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      now.time_since_epoch());
  const std::time_t time = std::chrono::system_clock::to_time_t(now);
  LocalTimestamp timestamp{
      .milliseconds = static_cast<uint32_t>(elapsed.count() % 1000),
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
  if (!stream) {
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

class StdioLog final {
public:
  [[nodiscard]] static std::shared_ptr<StdioLog>
  create(const LogConfig &config) {
    auto log = std::shared_ptr<StdioLog>(new StdioLog(config));
    return log->initialized_ ? std::move(log) : nullptr;
  }

  ~StdioLog() {
    std::scoped_lock lock(mutex_);
    if (file_) {
      std::fclose(file_);
    }
  }

  void write(LogLevel level, std::string_view message) {
    std::scoped_lock lock(mutex_);
    if (includes(config_.logLevel, level)) {
      writeLine(file_, level, message, config_.forceFlush, false);
    }
    if (includes(config_.consoleLevel, level)) {
      std::FILE *console = level >= LogLevel::Error ? stderr : stdout;
      writeLine(console, level, message, config_.forceFlush,
                console == stderr ? stderrColor_ : stdoutColor_);
    }
  }

private:
  explicit StdioLog(const LogConfig &config)
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

  LogConfig config_{};
  std::FILE *file_ = nullptr;
  std::mutex mutex_{};
  bool stdoutColor_ = false;
  bool stderrColor_ = false;
  bool initialized_ = false;
};

struct LoggerState {
  [[nodiscard]] std::shared_ptr<StdioLog> loadOrCreate() {
    std::shared_ptr<StdioLog> current =
        instance.load(std::memory_order_acquire);
    if (current) {
      return current;
    }
    std::scoped_lock lock(controlMutex);
    current = instance.load(std::memory_order_acquire);
    if (!current) {
      current = StdioLog::create(hasConfig ? config : LogConfig{});
      instance.store(current, std::memory_order_release);
    }
    return current;
  }

  void initialize(const LogConfig *newConfig) {
    std::scoped_lock lock(controlMutex);
    if (instance.load(std::memory_order_acquire)) {
      return;
    }
    if (newConfig) {
      config = *newConfig;
      hasConfig = true;
    }
    instance.store(StdioLog::create(hasConfig ? config : LogConfig{}),
                   std::memory_order_release);
  }

  void shutdown() {
    std::shared_ptr<StdioLog> oldInstance;
    {
      std::scoped_lock lock(controlMutex);
      oldInstance = instance.exchange({}, std::memory_order_acq_rel);
      config = {};
      hasConfig = false;
    }
  }

  std::mutex controlMutex;
  std::atomic<std::shared_ptr<StdioLog>> instance;
  LogConfig config;
  bool hasConfig = false;
};

struct RecentLogRing {
  static constexpr size_t kCapacity = 2000;

  void append(LogLevel level, std::string_view message) {
    if (message.empty()) {
      return;
    }
    std::scoped_lock lock(mutex);
    LogEntry &entry = slots[(nextSequence - 1) % kCapacity];
    entry.level = level;
    entry.message.assign(message);
    entry.sequence = nextSequence++;
  }

  LogReadResult readSince(std::uint64_t afterSequence,
                          std::vector<LogEntry> &out) const {
    std::scoped_lock lock(mutex);
    out.clear();
    const std::uint64_t claimed = nextSequence - 1;
    if (claimed == 0) {
      return {};
    }
    const std::uint64_t first =
        claimed > kCapacity ? claimed - kCapacity + 1 : 1;
    LogReadResult result{
        .firstSequence = first,
        .lastSequence = claimed,
        .truncated = afterSequence != 0 && afterSequence < first,
    };
    const std::uint64_t requested =
        afterSequence == std::numeric_limits<std::uint64_t>::max()
            ? afterSequence
            : afterSequence + 1;
    for (std::uint64_t sequence = std::max(requested, first);
         sequence <= claimed; ++sequence) {
      out.push_back(slots[(sequence - 1) % kCapacity]);
    }
    return result;
  }

  mutable std::mutex mutex;
  std::array<LogEntry, kCapacity> slots{};
  std::uint64_t nextSequence = 1;
};

LoggerState &loggerState() {
  static LoggerState state;
  return state;
}

RecentLogRing &recentLogRing() {
  static RecentLogRing ring;
  return ring;
}
} // namespace

void Log::initialize() { loggerState().initialize(nullptr); }

void Log::initialize(const LogConfig &config) {
  loggerState().initialize(&config);
}

void Log::shutdown() { loggerState().shutdown(); }

void logMessage(LogLevel level, std::string_view message) {
  recentLogRing().append(level, message);
  if (std::shared_ptr<StdioLog> log = loggerState().loadOrCreate()) {
    log->write(level, message);
  } else if (!message.empty()) {
    std::fwrite(message.data(), sizeof(char), message.size(), stderr);
    std::fputc('\n', stderr);
  }
}

void logMessagef(LogLevel level, const char *fmt, ...) {
  if (!fmt) {
    return;
  }
  va_list args;
  va_start(args, fmt);
  va_list copy;
  va_copy(copy, args);
  const int required = std::vsnprintf(nullptr, 0, fmt, copy);
  va_end(copy);
  if (required < 0) {
    va_end(args);
    return;
  }
  std::string buffer(static_cast<size_t>(required) + 1, '\0');
  std::vsnprintf(buffer.data(), buffer.size(), fmt, args);
  va_end(args);
  buffer.resize(static_cast<size_t>(required));
  logMessage(level, buffer);
}

LogReadResult readLogEntriesSince(std::uint64_t afterSequence,
                                  std::vector<LogEntry> &out) {
  return recentLogRing().readSince(afterSequence, out);
}
} // namespace nuri
