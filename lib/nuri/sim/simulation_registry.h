#pragma once

#include "nuri/core/containers/slot_pool.h"
#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/resources/gpu/resource_handles.h"
#include "nuri/sim/simulation_bindings.h"
#include "nuri/sim/simulation_desc.h"
#include "nuri/sim/simulation_stats.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory_resource>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace nuri {

class NURI_API SimulationRegistry {
public:
  struct Record {
    explicit Record(
        std::pmr::memory_resource *memory = std::pmr::get_default_resource())
        : debugName(memory), binding(memory), params(memory),
          faultReason(memory), stats(memory) {}

    SimulationKind kind = SimulationKind::Unknown;
    SimulationBackendPreference backendPreference =
        SimulationBackendPreference::Auto;
    SimulationState state = SimulationState::Stopped;
    bool enabled = false;
    bool allowGpuExecution = true;
    bool singleStepRequested = false;
    bool faulted = false;
    float timeScale = 1.0f;
    uint32_t priority = 0u;
    uint32_t substepCount = 1u;
    uint32_t solverIterationCount = 1u;
    uint64_t creationOrder = 0u;
    std::pmr::string debugName;
    SimulationBindingDesc binding;
    std::pmr::vector<std::byte> params;
    std::pmr::string faultReason;
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
      fn(makeSimulationHandle(index, slots_.generation(index)),
         records_[index]);
    }
  }

  template <typename Fn> void forEachLive(Fn &&fn) const {
    for (uint32_t index = 0; index < slots_.slotCount(); ++index) {
      if (!slots_.isLive(index)) {
        continue;
      }
      fn(makeSimulationHandle(index, slots_.generation(index)),
         records_[index]);
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
  [[nodiscard]] static Result<bool, std::string>
  validateCreateDesc(const SimulationDesc &desc);
  [[nodiscard]] bool slotValid(SimulationHandle handle) const noexcept;

  std::pmr::memory_resource *memory_ = std::pmr::get_default_resource();
  SlotPool<MaskedNonZeroGenerationPolicy<kResourceHandleGenerationMask>> slots_;
  std::pmr::vector<Record> records_;
  uint64_t nextCreationOrder_ = 1u;
};

} // namespace nuri
