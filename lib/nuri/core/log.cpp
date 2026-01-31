#include "nuri/core/log.h"

namespace nuri {

namespace {

std::mutex g_log_mutex;
std::unique_ptr<Log> g_log;
LogConfig g_config;
bool g_has_config = false;

void writeFallback(std::string_view message) {
  if (message.empty()) {
    return;
  }
  std::fwrite(message.data(), sizeof(char), message.size(), stderr);
  std::fputc('\n', stderr);
}

} // namespace

static Log *getOrCreateUnlocked() {
  if (!g_log) {
    g_log = g_has_config ? Log::create(g_config) : Log::create();
  }
  return g_log.get();
}

void Log::initialize() {
  std::scoped_lock lock(g_log_mutex);
  getOrCreateUnlocked();
}

void Log::initialize(const LogConfig &config) {
  std::scoped_lock lock(g_log_mutex);
  if (g_log) {
    return;
  }

  g_config = config;
  g_has_config = true;
  getOrCreateUnlocked();
}

void Log::shutdown() {
  std::scoped_lock lock(g_log_mutex);
  g_log.reset();
  g_config = {};
  g_has_config = false;
}

Log *Log::get() {
  std::scoped_lock lock(g_log_mutex);
  return getOrCreateUnlocked();
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
