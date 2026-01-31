#pragma once

#include "nuri/defines.h"
#include "nuri/pch.h"

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

#define NURI_ASSERT(condition, fmt, ...)                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      nuri::logMessagef(nuri::LogLevel::Fatal,                                 \
                        "Assertion failed: " #condition " " fmt,               \
                        ##__VA_ARGS__);                                        \
      std::terminate();                                                        \
    }                                                                          \
  } while (false)

} // namespace nuri
