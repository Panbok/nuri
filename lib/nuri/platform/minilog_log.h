#pragma once

#include "nuri/core/log.h"

namespace nuri {

class MinilogLog final : public Log {
public:
  static std::unique_ptr<MinilogLog> create(const LogConfig &config);
  ~MinilogLog() override;

  MinilogLog(const MinilogLog &) = delete;
  MinilogLog &operator=(const MinilogLog &) = delete;
  MinilogLog(MinilogLog &&) = delete;
  MinilogLog &operator=(MinilogLog &&) = delete;

  void write(LogLevel level, std::string_view message) override;

private:
  explicit MinilogLog(const LogConfig &config);
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace nuri
