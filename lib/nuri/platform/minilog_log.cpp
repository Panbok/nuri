#include "nuri/platform/minilog_log.h"

#include <minilog/minilog.h>

namespace nuri {

namespace {

minilog::eLogLevel toMinilogLevel(LogLevel level) {
  switch (level) {
  case LogLevel::Trace:
    return minilog::Paranoid;
  case LogLevel::Debug:
    return minilog::Debug;
  case LogLevel::Info:
    return minilog::Log;
  case LogLevel::Warning:
    return minilog::Warning;
  case LogLevel::Fatal:
    return minilog::FatalError;
  }
  return minilog::Log;
}

} // namespace

struct MinilogLog::Impl {
  bool initialized = false;
  std::string filePath;
};

MinilogLog::MinilogLog(const LogConfig &userConfig)
    : impl_(std::make_unique<Impl>()) {
  minilog::LogConfig config{};
  config.logLevel = toMinilogLevel(userConfig.logLevel);
  config.logLevelPrintToConsole = toMinilogLevel(userConfig.consoleLevel);
  if (config.logLevelPrintToConsole < config.logLevel) {
    config.logLevelPrintToConsole = config.logLevel;
  }
  config.forceFlush = userConfig.forceFlush;
  config.writeIntro = userConfig.writeIntro;
  config.writeOutro = userConfig.writeOutro;
  config.coloredConsole = userConfig.coloredConsole;
  config.htmlLog = userConfig.htmlLog;
  config.threadNames = userConfig.threadNames;

  const char *fileName = nullptr;
  if (!userConfig.filePath.empty()) {
    impl_->filePath = std::string(userConfig.filePath);
    fileName = impl_->filePath.c_str();
  }

  impl_->initialized = minilog::initialize(fileName, config);
}

MinilogLog::~MinilogLog() {
  if (impl_ && impl_->initialized) {
    minilog::deinitialize();
  }
}

std::unique_ptr<MinilogLog> MinilogLog::create(const LogConfig &config) {
  auto log = std::unique_ptr<MinilogLog>(new MinilogLog(config));
  if (!log->impl_ || !log->impl_->initialized) {
    return nullptr;
  }
  return log;
}

void MinilogLog::write(LogLevel level, std::string_view message) {
  const int length = static_cast<int>(message.size());
  minilog::log(toMinilogLevel(level), "%.*s", length, message.data());
}

std::unique_ptr<Log> Log::create() {
  LogConfig config{};
  return MinilogLog::create(config);
}

std::unique_ptr<Log> Log::create(const LogConfig &config) {
  return MinilogLog::create(config);
}

} // namespace nuri
