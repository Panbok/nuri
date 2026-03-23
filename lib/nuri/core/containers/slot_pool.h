#pragma once

#include "nuri/core/log.h"

#include <cstdint>
#include <memory_resource>
#include <vector>

namespace nuri {

struct SlotReservation {
  uint32_t index = 0;
  uint32_t generation = 0;
  bool appended = false;
};

template <uint32_t Mask> struct MaskedNonZeroGenerationPolicy {
  [[nodiscard]] static constexpr uint32_t next(uint32_t current) noexcept {
    const uint32_t nextGeneration = (current + 1u) & Mask;
    return nextGeneration == 0u ? 1u : nextGeneration;
  }
};

struct UnmaskedNonZeroGenerationPolicy {
  [[nodiscard]] static constexpr uint32_t next(uint32_t current) noexcept {
    const uint32_t nextGeneration = current + 1u;
    return nextGeneration == 0u ? 1u : nextGeneration;
  }
};

template <typename GenerationPolicy = UnmaskedNonZeroGenerationPolicy>
// SlotPool is not thread-safe. Concurrent acquire()/release() calls must be
// externally synchronized; the intended default usage is single-threaded.
class SlotPool {
public:
  explicit SlotPool(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : generations_(memory), live_(memory), freeSlots_(memory) {}

  void reserve(uint32_t capacity) {
    generations_.reserve(capacity);
    live_.reserve(capacity);
    freeSlots_.reserve(capacity);
  }

  void clear() {
    generations_.clear();
    live_.clear();
    freeSlots_.clear();
    liveCount_ = 0;
  }

  [[nodiscard]] uint32_t slotCount() const noexcept {
    return static_cast<uint32_t>(generations_.size());
  }

  [[nodiscard]] uint32_t liveCount() const noexcept { return liveCount_; }

  [[nodiscard]] SlotReservation acquire() {
    SlotReservation reservation{};
    if (!freeSlots_.empty()) {
      reservation.index = freeSlots_.back();
      freeSlots_.pop_back();
    } else {
      reservation.index = static_cast<uint32_t>(generations_.size());
      generations_.push_back(0u);
      live_.push_back(0u);
      reservation.appended = true;
    }

    generations_[reservation.index] =
        GenerationPolicy::next(generations_[reservation.index]);
    live_[reservation.index] = 1u;
    reservation.generation = generations_[reservation.index];
    ++liveCount_;
    return reservation;
  }

  void release(uint32_t index) {
    NURI_ASSERT(index < generations_.size(),
                "SlotPool::release: index out of range");
    NURI_ASSERT(live_[index] != 0u, "SlotPool::release: slot is not live");
    NURI_ASSERT(liveCount_ > 0u, "SlotPool::release: live count underflow");
    live_[index] = 0u;
    freeSlots_.push_back(index);
    --liveCount_;
  }

  [[nodiscard]] bool isLive(uint32_t index) const noexcept {
    return index < live_.size() && live_[index] != 0u;
  }

  [[nodiscard]] uint32_t generation(uint32_t index) const noexcept {
    return index < generations_.size() ? generations_[index] : 0u;
  }

  [[nodiscard]] bool isValid(uint32_t index,
                             uint32_t expectedGeneration) const noexcept {
    return isLive(index) && generation(index) == expectedGeneration;
  }

private:
  std::pmr::vector<uint32_t> generations_;
  std::pmr::vector<uint8_t> live_;
  std::pmr::vector<uint32_t> freeSlots_;
  uint32_t liveCount_ = 0;
};

} // namespace nuri
