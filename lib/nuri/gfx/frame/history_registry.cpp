#include "nuri/pch.h"

#include "nuri/gfx/frame/history_registry.h"

#include <algorithm>

namespace nuri {

Result<HistoryLease, std::string>
HistoryRegistry::prepareFrame(uint64_t frameIndex, uint32_t slotCount) {
  if (lease_.pendingCommit) {
    if (lease_.frameIndex == frameIndex) {
      return Result<HistoryLease, std::string>::makeResult(lease_);
    }
    return Result<HistoryLease, std::string>::makeError(
        "HistoryRegistry::prepareFrame: prior frame is still pending");
  }

  const uint32_t safeSlotCount = std::max(slotCount, 1u);
  uint32_t writeSlot = static_cast<uint32_t>(frameIndex % safeSlotCount);
  if (committedValid_ && safeSlotCount > 1u && writeSlot == committedSlot_) {
    writeSlot = (writeSlot + 1u) % safeSlotCount;
  }
  lease_ = HistoryLease{
      .frameIndex = frameIndex,
      .generation = generation_,
      .readSlot = committedSlot_ % safeSlotCount,
      .writeSlot = writeSlot,
      .readValid = committedValid_,
      .pendingCommit = true,
  };
  return Result<HistoryLease, std::string>::makeResult(lease_);
}

bool HistoryRegistry::commitFrame(uint64_t frameIndex) noexcept {
  if (!lease_.pendingCommit || lease_.frameIndex != frameIndex) {
    return false;
  }
  committedSlot_ = lease_.writeSlot;
  committedFrameIndex_ = frameIndex;
  committedValid_ = true;
  lease_.pendingCommit = false;
  lease_.readValid = true;
  lastInvalidationReason_ = HistoryInvalidationReason::None;
  return true;
}

void HistoryRegistry::abandonFrame(uint64_t frameIndex) noexcept {
  if (lease_.pendingCommit && lease_.frameIndex == frameIndex) {
    lease_.pendingCommit = false;
  }
}

void HistoryRegistry::invalidate(HistoryInvalidationReason reason) noexcept {
  ++generation_;
  committedValid_ = false;
  lease_.pendingCommit = false;
  lease_.readValid = false;
  lastInvalidationReason_ = reason;
}

void HistoryRegistry::reset() noexcept { *this = HistoryRegistry{}; }

} // namespace nuri
