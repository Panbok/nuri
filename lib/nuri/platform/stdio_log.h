#pragma once

#include "nuri/core/log.h"

#include <cstdio>
#include <mutex>

namespace nuri {

class StdioLog final : public Log {
public:
  static std::unique_ptr<StdioLog> create(const LogConfig &config);
  ~StdioLog() override;

  StdioLog(const StdioLog &) = delete;
  StdioLog &operator=(const StdioLog &) = delete;
  StdioLog(StdioLog &&) = delete;
  StdioLog &operator=(StdioLog &&) = delete;

  void write(LogLevel level, std::string_view message) override;

private:
  explicit StdioLog(const LogConfig &config);

  LogConfig config_{};
  std::FILE *file_ = nullptr;
  std::mutex mutex_{};
  bool stdoutColor_ = false;
  bool stderrColor_ = false;
  bool initialized_ = false;
};

} // namespace nuri
