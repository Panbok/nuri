#pragma once
#include "nuri/core/result.h"
#include "nuri/defines.h"
#include <cstdint>
#include <string>
namespace nuri {

enum class HistoryInvalidationReason : uint8_t {
  None = 0u,
  ResourceRecreation,
  BackendRecreation,
  ExplicitReset,
};

struct HistoryLease {
  uint64_t frameIndex = 0u;
  uint64_t generation = 0u;
  uint32_t readSlot = 0u;
  uint32_t writeSlot = 0u;
  bool readValid = false;
  bool pendingCommit = false;
};

class NURI_API HistoryRegistry final {
public:
  [[nodiscard]] Result<HistoryLease, std::string>
  prepareFrame(uint64_t frameIndex, uint32_t slotCount);
  [[nodiscard]] bool commitFrame(uint64_t frameIndex) noexcept;
  void abandonFrame(uint64_t frameIndex) noexcept;
  void invalidate(HistoryInvalidationReason reason) noexcept;
  void reset() noexcept;
  [[nodiscard]] const HistoryLease &lease() const noexcept { return lease_; }
  [[nodiscard]] uint64_t committedFrameIndex() const noexcept {
    return committedFrameIndex_;
  }
  [[nodiscard]] HistoryInvalidationReason
  lastInvalidationReason() const noexcept {
    return lastInvalidationReason_;
  }

private:
  HistoryLease lease_{};
  uint64_t committedFrameIndex_ = 0u;
  uint64_t generation_ = 0u;
  uint32_t committedSlot_ = 0u;
  bool committedValid_ = false;
  HistoryInvalidationReason lastInvalidationReason_ =
      HistoryInvalidationReason::None;
};

} // namespace nuri
