#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory_resource>
#include <optional>
#include <vector>
namespace nuri {

class RecordingRetirementTracker {
public:
  explicit RecordingRetirementTracker(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : resolutions_(memory) {}
  [[nodiscard]] uint64_t beginRecording() {
    resolutions_.push_back(kUnresolved);
    return ++latestIssuedSerial_;
  }
  [[nodiscard]] bool resolveRecording(uint64_t serial,
                                      uint64_t submissionInstance) {
    if (serial < baseSerial_ || serial > latestIssuedSerial_) {
      return false;
    }
    uint64_t &resolution =
        resolutions_[static_cast<size_t>(serial - baseSerial_)];
    if (resolution != kUnresolved) {
      return false;
    }
    resolution = submissionInstance;
    while (head_ < resolutions_.size() && resolutions_[head_] != kUnresolved) {
      resolvedSubmissionMax_ =
          std::max(resolvedSubmissionMax_, resolutions_[head_]);
      resolvedThroughSerial_ = baseSerial_ + head_++;
    }
    if (head_ >= 256u && head_ * 2u >= resolutions_.size()) {
      resolutions_.erase(resolutions_.begin(), resolutions_.begin() + head_);
      baseSerial_ += head_;
      head_ = 0u;
    }
    return true;
  }
  [[nodiscard]] uint64_t latestIssuedSerial() const {
    return latestIssuedSerial_;
  }
  [[nodiscard]] uint64_t resolvedThroughSerial() const {
    return resolvedThroughSerial_;
  }
  [[nodiscard]] uint64_t resolvedSubmissionMax() const {
    return resolvedSubmissionMax_;
  }
  [[nodiscard]] std::optional<uint64_t>
  tryResolveLastUse(uint64_t requiredSerial,
                    uint64_t lastSubmittedAtDestruction) const {
    if (requiredSerial > resolvedThroughSerial_) {
      return std::nullopt;
    }
    return std::max(lastSubmittedAtDestruction, resolvedSubmissionMax_);
  }

private:
  static constexpr uint64_t kUnresolved = std::numeric_limits<uint64_t>::max();
  std::pmr::vector<uint64_t> resolutions_;
  uint64_t baseSerial_ = 1u;
  size_t head_ = 0u;
  uint64_t latestIssuedSerial_ = 0u;
  uint64_t resolvedThroughSerial_ = 0u;
  uint64_t resolvedSubmissionMax_ = 0u;
};

} // namespace nuri
