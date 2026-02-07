#include "nuri/core/log.h"

#include <array>
#include <atomic>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>

namespace nuri {

namespace {

void writeFallback(std::string_view message) {
  if (message.empty()) {
    return;
  }
  std::fwrite(message.data(), sizeof(char), message.size(), stderr);
  std::fputc('\n', stderr);
}

struct LoggerState {
  std::shared_ptr<Log> loadOrCreate() {
    std::shared_ptr<Log> current = instance.load(std::memory_order_acquire);
    if (current) {
      return current;
    }

    std::scoped_lock lock(controlMutex);
    current = instance.load(std::memory_order_acquire);
    if (current) {
      return current;
    }

    std::unique_ptr<Log> createdUnique = hasConfig ? Log::create(config) : Log::create();
    std::shared_ptr<Log> created = std::move(createdUnique);
    instance.store(created, std::memory_order_release);
    return created;
  }

  void initializeDefault() { (void)loadOrCreate(); }

  void initializeWithConfig(const LogConfig &newConfig) {
    std::scoped_lock lock(controlMutex);
    if (instance.load(std::memory_order_acquire)) {
      return;
    }

    config = newConfig;
    hasConfig = true;

    std::unique_ptr<Log> createdUnique = Log::create(config);
    std::shared_ptr<Log> created = std::move(createdUnique);
    instance.store(created, std::memory_order_release);
  }

  void shutdown() {
    std::shared_ptr<Log> oldInstance;
    {
      std::scoped_lock lock(controlMutex);
      oldInstance =
          instance.exchange(std::shared_ptr<Log>{}, std::memory_order_acq_rel);
      config = {};
      hasConfig = false;
    }
    oldInstance.reset();
  }

  Log *getRawLegacy() { return loadOrCreate().get(); }

  std::mutex controlMutex;
  std::atomic<std::shared_ptr<Log>> instance;
  LogConfig config;
  bool hasConfig = false;
};

struct RecentLogRing {
  static constexpr size_t kCapacity = 2000;

  void append(LogLevel level, std::string_view message) {
    if (message.empty()) {
      return;
    }

    auto entry = std::make_shared<LogEntry>();
    entry->level = level;
    entry->message.assign(message.data(), message.size());
    entry->sequence = nextSequence.fetch_add(1, std::memory_order_acq_rel);

    const size_t index = static_cast<size_t>((entry->sequence - 1) % kCapacity);
    std::shared_ptr<const LogEntry> published = std::move(entry);
    slots[index].store(std::move(published), std::memory_order_release);
  }

  LogReadResult readSince(std::uint64_t afterSequence,
                          std::vector<LogEntry> &out) const {
    out.clear();

    LogReadResult result{};
    const std::uint64_t claimed =
        nextSequence.load(std::memory_order_acquire) - 1;
    if (claimed == 0) {
      return result;
    }

    const std::uint64_t capacity = static_cast<std::uint64_t>(kCapacity);
    const std::uint64_t firstSequence =
        claimed > capacity ? (claimed - capacity + 1) : 1;
    result.firstSequence = firstSequence;

    if (afterSequence != 0 && afterSequence < firstSequence) {
      result.truncated = true;
    }

    const std::uint64_t nextRequested =
        afterSequence == std::numeric_limits<std::uint64_t>::max()
            ? afterSequence
            : (afterSequence + 1);
    const std::uint64_t start =
        nextRequested > firstSequence ? nextRequested : firstSequence;

    std::uint64_t contiguousLast = afterSequence;
    for (std::uint64_t sequence = start; sequence <= claimed; ++sequence) {
      const size_t index = static_cast<size_t>((sequence - 1) % kCapacity);
      std::shared_ptr<const LogEntry> entry =
          slots[index].load(std::memory_order_acquire);
      if (!entry || entry->sequence != sequence) {
        break;
      }

      out.push_back(*entry);
      contiguousLast = sequence;
    }

    if (contiguousLast >= firstSequence) {
      result.lastSequence = contiguousLast;
    } else {
      result.lastSequence = firstSequence - 1;
    }
    return result;
  }

  std::array<std::atomic<std::shared_ptr<const LogEntry>>, kCapacity> slots{};
  std::atomic<std::uint64_t> nextSequence{1};
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

void Log::initialize() { loggerState().initializeDefault(); }

void Log::initialize(const LogConfig &config) {
  loggerState().initializeWithConfig(config);
}

void Log::shutdown() { loggerState().shutdown(); }

Log *Log::get() { return loggerState().getRawLegacy(); }

void logMessage(LogLevel level, std::string_view message) {
  recentLogRing().append(level, message);
  std::shared_ptr<Log> log = loggerState().loadOrCreate();
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

LogReadResult readLogEntriesSince(std::uint64_t afterSequence,
                                  std::vector<LogEntry> &out) {
  return recentLogRing().readSince(afterSequence, out);
}

} // namespace nuri
