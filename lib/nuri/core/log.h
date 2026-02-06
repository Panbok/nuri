#pragma once

#include "nuri/defines.h"
#include "nuri/pch.h"

#include <cstdint>
#include <string>
#include <vector>

namespace nuri {

enum class LogLevel : uint8_t {
  Trace,
  Debug,
  Info,
  Warning,
  Fatal,
};

struct LogConfig {
  std::string filePath;
  LogLevel logLevel = LogLevel::Debug;
  LogLevel consoleLevel = LogLevel::Info;
  bool forceFlush = true;
  bool writeIntro = true;
  bool writeOutro = true;
  bool coloredConsole = true;
  bool htmlLog = false;
  bool threadNames = false;
};

class NURI_API Log {
public:
  static std::unique_ptr<Log> create();
  static std::unique_ptr<Log> create(const LogConfig &config);
  static void initialize();
  static void initialize(const LogConfig &config);
  static void shutdown();
  static Log *get();

  virtual ~Log() = default;
  Log(const Log &) = delete;
  Log &operator=(const Log &) = delete;
  Log(Log &&) = delete;
  Log &operator=(Log &&) = delete;

  virtual void write(LogLevel level, std::string_view message) = 0;

protected:
  Log() = default;
};

NURI_API void logMessage(LogLevel level, std::string_view message);
NURI_API void logMessagef(LogLevel level, const char *fmt, ...);

struct LogEntry {
  LogLevel level = LogLevel::Info;
  std::string message;
  std::uint64_t sequence = 0;
};

struct LogReadResult {
  std::uint64_t firstSequence = 0;
  std::uint64_t lastSequence = 0;
  bool truncated = false;
};

NURI_API LogReadResult readLogEntriesSince(std::uint64_t afterSequence,
                                           std::vector<LogEntry> &out);

#define NURI_LOG_TRACE(fmt, ...)                                               \
  do {                                                                         \
    nuri::logMessagef(nuri::LogLevel::Trace, fmt __VA_OPT__(,) __VA_ARGS__);   \
  } while (false)

#define NURI_LOG_DEBUG(fmt, ...)                                               \
  do {                                                                         \
    nuri::logMessagef(nuri::LogLevel::Debug, fmt __VA_OPT__(,) __VA_ARGS__);   \
  } while (false)

#define NURI_LOG_INFO(fmt, ...)                                                \
  do {                                                                         \
    nuri::logMessagef(nuri::LogLevel::Info, fmt __VA_OPT__(,) __VA_ARGS__);    \
  } while (false)

#define NURI_LOG_WARNING(fmt, ...)                                             \
  do {                                                                         \
    nuri::logMessagef(nuri::LogLevel::Warning,                                 \
                      fmt __VA_OPT__(,) __VA_ARGS__);                          \
  } while (false)

#define NURI_LOG_FATAL(fmt, ...)                                               \
  do {                                                                         \
    nuri::logMessagef(nuri::LogLevel::Fatal, fmt __VA_OPT__(,) __VA_ARGS__);   \
  } while (false)

#define NURI_ASSERT(condition, fmt, ...)                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      NURI_LOG_FATAL("Assertion failed: " #condition " " fmt                   \
                         __VA_OPT__(,) __VA_ARGS__);                           \
      std::terminate();                                                        \
    }                                                                          \
  } while (false)

} // namespace nuri
