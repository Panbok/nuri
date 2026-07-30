#pragma once
#include "nuri/core/containers/slot_pool.h"
#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/resources/gpu/resource_handles.h"
#include "nuri/sim/simulation_bindings.h"
#include "nuri/sim/simulation_desc.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory_resource>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>
namespace nuri {

struct NURI_API SimulationStats {
  explicit SimulationStats(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : lastFaultReason(memory) {}
  uint64_t creationOrder = 0u;
  uint64_t controlVersion = 0u;
  uint64_t executedStepCount = 0u;
  uint64_t executedSubstepCount = 0u;
  std::array<uint64_t, static_cast<size_t>(SimulationPhase::Count)>
      phaseExecutionCounts{};
  uint64_t lastExecutedFrameIndex = std::numeric_limits<uint64_t>::max();
  uint64_t paramsSizeBytes = 0u;
  bool enabled = false;
  bool paused = false;
  bool faulted = false;
  std::pmr::string lastFaultReason;
};

class NURI_API SimulationRegistry {
public:
  struct Record {
    explicit Record(
        std::pmr::memory_resource *memory = std::pmr::get_default_resource())
        : debugName(memory), binding(memory), params(memory), stats(memory) {}
    SimulationKind kind = SimulationKind::Unknown;
    SimulationState state = SimulationState::Stopped;
    bool enabled = false;
    bool singleStepRequested = false;
    float timeScale = 1.0f;
    uint32_t priority = 0u;
    uint32_t substepCount = 1u;
    uint32_t solverIterationCount = 1u;
    uint64_t creationOrder = 0u;
    std::pmr::string debugName;
    SimulationBindingDesc binding;
    std::pmr::vector<std::byte> params;
    SimulationStats stats;
  };
  explicit SimulationRegistry(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  void clear();
  [[nodiscard]] Result<SimulationHandle, std::string>
  create(const SimulationDesc &desc);
  [[nodiscard]] bool destroy(SimulationHandle handle);
  [[nodiscard]] Record *tryGet(SimulationHandle handle) noexcept;
  [[nodiscard]] const Record *tryGet(SimulationHandle handle) const noexcept;
  [[nodiscard]] uint32_t liveCount() const noexcept {
    return slots_.liveCount();
  }
  template <typename Fn> void forEachLive(Fn &&fn) {
    for (uint32_t index = 0; index < slots_.slotCount(); ++index) {
      if (!slots_.isLive(index)) {
        continue;
      }
      if constexpr (std::is_same_v<
                        std::invoke_result_t<Fn &, SimulationHandle, Record &>,
                        bool>) {
        if (!std::invoke(
                fn,
                SimulationHandle::fromParts(index, slots_.generation(index)),
                records_[index])) {
          break;
        }
      } else {
        std::invoke(
            fn, SimulationHandle::fromParts(index, slots_.generation(index)),
            records_[index]);
      }
    }
  }
  template <typename Fn> void forEachLive(Fn &&fn) const {
    for (uint32_t index = 0; index < slots_.slotCount(); ++index) {
      if (!slots_.isLive(index)) {
        continue;
      }
      if constexpr (std::is_same_v<std::invoke_result_t<Fn &, SimulationHandle,
                                                        const Record &>,
                                   bool>) {
        if (!std::invoke(
                fn,
                SimulationHandle::fromParts(index, slots_.generation(index)),
                records_[index])) {
          break;
        }
      } else {
        std::invoke(
            fn, SimulationHandle::fromParts(index, slots_.generation(index)),
            records_[index]);
      }
    }
  }
  [[nodiscard]] bool setEnabled(SimulationHandle handle, bool enabled);
  [[nodiscard]] bool pause(SimulationHandle handle);
  [[nodiscard]] bool resume(SimulationHandle handle);
  [[nodiscard]] bool requestSingleStep(SimulationHandle handle);
  [[nodiscard]] bool setTimeScale(SimulationHandle handle, float timeScale);
  [[nodiscard]] bool setSubstepCount(SimulationHandle handle, uint32_t count);
  [[nodiscard]] bool setSolverIterationCount(SimulationHandle handle,
                                             uint32_t count);
  [[nodiscard]] bool setParams(SimulationHandle handle,
                               std::span<const std::byte> params);
  [[nodiscard]] bool getState(SimulationHandle handle,
                              SimulationState &out) const noexcept;
  [[nodiscard]] bool getDesc(SimulationHandle handle,
                             SimulationDesc &out) const;
  [[nodiscard]] bool getStats(SimulationHandle handle,
                              SimulationStats &out) const;
  [[nodiscard]] bool markFaulted(SimulationHandle handle,
                                 std::string_view reason);
  [[nodiscard]] bool clearSingleStepRequest(SimulationHandle handle);
  [[nodiscard]] bool notePhaseExecution(SimulationHandle handle,
                                        SimulationPhase phase,
                                        uint64_t frameIndex);

private:
  [[nodiscard]] static Result<void, std::string>
  validateCreateDesc(const SimulationDesc &desc);
  [[nodiscard]] bool slotValid(SimulationHandle handle) const noexcept;
  std::pmr::memory_resource *memory_ = std::pmr::get_default_resource();
  SlotPool<MaskedNonZeroGenerationPolicy<kResourceHandleGenerationMask>> slots_;
  std::pmr::vector<Record> records_;
  uint64_t nextCreationOrder_ = 1u;
};

} // namespace nuri
