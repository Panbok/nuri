#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory_resource>
#include <optional>

namespace nuri {

// Tracks recording lifetimes independently from queue submission lifetimes.
// Resource destruction snapshots latestIssuedSerial(); the resource cannot be
// assigned a queue completion value until every recording at or before that
// serial has either been submitted or abandoned.
class RecordingRetirementTracker {
public:
  explicit RecordingRetirementTracker(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : resolvedOutOfOrder_(memory) {}

  [[nodiscard]] uint64_t beginRecording() noexcept {
    ++latestIssuedSerial_;
    if (latestIssuedSerial_ == 0u) {
      ++latestIssuedSerial_;
    }
    return latestIssuedSerial_;
  }

  // submissionInstance is zero for an abandoned recording.
  [[nodiscard]] bool resolveRecording(uint64_t serial,
                                      uint64_t submissionInstance) {
    if (serial == 0u || serial <= resolvedThroughSerial_ ||
        serial > latestIssuedSerial_) {
      return false;
    }
    const auto [_, inserted] =
        resolvedOutOfOrder_.emplace(serial, submissionInstance);
    if (!inserted) {
      return false;
    }

    for (;;) {
      const auto next = resolvedOutOfOrder_.find(resolvedThroughSerial_ + 1u);
      if (next == resolvedOutOfOrder_.end()) {
        break;
      }
      resolvedSubmissionMax_ = std::max(resolvedSubmissionMax_, next->second);
      resolvedThroughSerial_ = next->first;
      resolvedOutOfOrder_.erase(next);
    }
    return true;
  }

  [[nodiscard]] uint64_t latestIssuedSerial() const noexcept {
    return latestIssuedSerial_;
  }

  [[nodiscard]] uint64_t resolvedThroughSerial() const noexcept {
    return resolvedThroughSerial_;
  }

  [[nodiscard]] uint64_t resolvedSubmissionMax() const noexcept {
    return resolvedSubmissionMax_;
  }

  [[nodiscard]] bool
  areRecordingsResolvedThrough(uint64_t requiredSerial) const noexcept {
    return requiredSerial <= resolvedThroughSerial_;
  }

  // Returns a conservative graphics-queue last-use value after every recording
  // that existed at logical destruction has resolved. A later resolved
  // recording may raise the returned value, which delays reuse but remains
  // safe.
  [[nodiscard]] std::optional<uint64_t>
  tryResolveLastUse(uint64_t requiredSerial,
                    uint64_t lastSubmittedAtDestruction) const noexcept {
    if (!areRecordingsResolvedThrough(requiredSerial)) {
      return std::nullopt;
    }
    return std::max(lastSubmittedAtDestruction, resolvedSubmissionMax_);
  }

private:
  uint64_t latestIssuedSerial_ = 0u;
  uint64_t resolvedThroughSerial_ = 0u;
  uint64_t resolvedSubmissionMax_ = 0u;
  std::pmr::map<uint64_t, uint64_t> resolvedOutOfOrder_;
};

} // namespace nuri
