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
class SlotPool {
public:
  explicit SlotPool(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : generations_(memory), states_(memory), freeSlots_(memory) {}
  void reserve(uint32_t capacity) {
    generations_.reserve(capacity);
    states_.reserve(capacity);
    freeSlots_.reserve(capacity);
  }
  void clear() {
    generations_.clear();
    states_.clear();
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
      states_.push_back(SlotState::Free);
      reservation.appended = true;
    }
    generations_[reservation.index] =
        GenerationPolicy::next(generations_[reservation.index]);
    states_[reservation.index] = SlotState::Live;
    reservation.generation = generations_[reservation.index];
    ++liveCount_;
    return reservation;
  }
  void release(uint32_t index) {
    retire(index);
    recycle(index);
  }
  void retire(uint32_t index) {
    states_[index] = SlotState::Retired;
    --liveCount_;
  }
  void recycle(uint32_t index) {
    states_[index] = SlotState::Free;
    freeSlots_.push_back(index);
  }
  [[nodiscard]] bool isLive(uint32_t index) const noexcept {
    return index < states_.size() && states_[index] == SlotState::Live;
  }
  [[nodiscard]] bool isRetired(uint32_t index) const noexcept {
    return index < states_.size() && states_[index] == SlotState::Retired;
  }
  [[nodiscard]] uint32_t generation(uint32_t index) const noexcept {
    return index < generations_.size() ? generations_[index] : 0u;
  }
  [[nodiscard]] bool isValid(uint32_t index,
                             uint32_t expectedGeneration) const noexcept {
    return isLive(index) && generation(index) == expectedGeneration;
  }

private:
  enum class SlotState : uint8_t {
    Free,
    Live,
    Retired,
  };
  std::pmr::vector<uint32_t> generations_;
  std::pmr::vector<SlotState> states_;
  std::pmr::vector<uint32_t> freeSlots_;
  uint32_t liveCount_ = 0;
};

} // namespace nuri
