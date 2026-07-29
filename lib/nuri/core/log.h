#pragma once
#include "nuri/defines.h"
#include <cstdint>
#include <exception>
#include <string>
#include <string_view>
#include <vector>
namespace nuri {

enum class LogLevel : uint8_t {
  Trace,
  Debug,
  Info,
  Warning,
  Error,
  Fatal,
};

struct LogConfig {
  std::string filePath;
  LogLevel logLevel = LogLevel::Debug;
  LogLevel consoleLevel = LogLevel::Info;
  bool forceFlush = true;
  bool coloredConsole = true;
};

class NURI_API Log final {
public:
  static void initialize();
  static void initialize(const LogConfig &config);
  static void shutdown();
  Log() = delete;
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

#if defined(NURI_LOGGING) && NURI_LOGGING
#define NURI_LOG_TRACE(fmt, ...)                                               \
  do {                                                                         \
    nuri::logMessagef(nuri::LogLevel::Trace, fmt __VA_OPT__(, ) __VA_ARGS__);  \
  } while (false)

#define NURI_LOG_DEBUG(fmt, ...)                                               \
  do {                                                                         \
    nuri::logMessagef(nuri::LogLevel::Debug, fmt __VA_OPT__(, ) __VA_ARGS__);  \
  } while (false)

#define NURI_LOG_INFO(fmt, ...)                                                \
  do {                                                                         \
    nuri::logMessagef(nuri::LogLevel::Info, fmt __VA_OPT__(, ) __VA_ARGS__);   \
  } while (false)

#define NURI_LOG_WARNING(fmt, ...)                                             \
  do {                                                                         \
    nuri::logMessagef(nuri::LogLevel::Warning,                                 \
                      fmt __VA_OPT__(, ) __VA_ARGS__);                         \
  } while (false)

#define NURI_LOG_ERROR(fmt, ...)                                               \
  do {                                                                         \
    nuri::logMessagef(nuri::LogLevel::Error, fmt __VA_OPT__(, ) __VA_ARGS__);  \
  } while (false)

#define NURI_LOG_FATAL(fmt, ...)                                               \
  do {                                                                         \
    nuri::logMessagef(nuri::LogLevel::Fatal, fmt __VA_OPT__(, ) __VA_ARGS__);  \
  } while (false)
#else
#define NURI_LOG_TRACE(...)                                                    \
  do {                                                                         \
  } while (false)

#define NURI_LOG_DEBUG(...)                                                    \
  do {                                                                         \
  } while (false)

#define NURI_LOG_INFO(...)                                                     \
  do {                                                                         \
  } while (false)

#define NURI_LOG_WARNING(...)                                                  \
  do {                                                                         \
  } while (false)

#define NURI_LOG_ERROR(...)                                                    \
  do {                                                                         \
  } while (false)

#define NURI_LOG_FATAL(...)                                                    \
  do {                                                                         \
  } while (false)
#endif

#if defined(NURI_ASSERTS) && NURI_ASSERTS
#define NURI_ASSERT(condition, fmt, ...)                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      NURI_LOG_FATAL("Assertion failed: " #condition " " fmt __VA_OPT__(, )    \
                         __VA_ARGS__);                                         \
      std::terminate();                                                        \
    }                                                                          \
  } while (false)
#else
#define NURI_ASSERT(...)                                                       \
  do {                                                                         \
  } while (false)
#endif

} // namespace nuri
