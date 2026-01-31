#include "nuri/core/log.h"

namespace nuri {

namespace {

std::unique_ptr<Log> g_log;
LogConfig g_config;
bool g_has_config = false;
std::string g_file_path_storage;

void writeFallback(std::string_view message) {
  if (message.empty()) {
    return;
  }
  std::fwrite(message.data(), sizeof(char), message.size(), stderr);
  std::fputc('\n', stderr);
}

} // namespace

void Log::initialize() { (void)Log::get(); }

void Log::initialize(const LogConfig &config) {
  if (g_log) {
    return;
  }

  g_config = config;
  if (!config.filePath.empty()) {
    g_file_path_storage = std::string(config.filePath);
    g_config.filePath = g_file_path_storage;
  }

  g_has_config = true;
  (void)Log::get();
}

void Log::shutdown() {
  g_log.reset();
  g_config = {};
  g_has_config = false;
  g_file_path_storage.clear();
}

Log *Log::get() {
  if (!g_log) {
    g_log = g_has_config ? Log::create(g_config) : Log::create();
  }
  return g_log.get();
}

void logMessage(LogLevel level, std::string_view message) {
  Log *log = Log::get();
  if (!log) {
    writeFallback(message);
    return;
  }
  log->write(level, message);
}

void logMessagef(LogLevel level, const char *fmt, ...) {
  if (!fmt) {
    return;
  }

  va_list args;
  va_start(args, fmt);

  va_list argsCopy;
  va_copy(argsCopy, args);
  const int required = std::vsnprintf(nullptr, 0, fmt, argsCopy);
  va_end(argsCopy);

  if (required < 0) {
    va_end(args);
    return;
  }

  std::string buffer;
  buffer.resize(static_cast<size_t>(required) + 1);
  std::vsnprintf(buffer.data(), buffer.size(), fmt, args);
  buffer.resize(static_cast<size_t>(required));
  va_end(args);

  logMessage(level, buffer);
}

} // namespace nuri
