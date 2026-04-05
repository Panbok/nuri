#pragma once

#include "nuri/defines.h"
#include "nuri/gfx/sim/simulation_gpu_types.h"

#include <cstdint>

namespace nuri {

class NURI_API SimulationGpuContext {
public:
  void reset() noexcept;
  void beginFrame(uint64_t frameIndex) noexcept;
  void publishFrameResources(const SimulationFrameGpuResources &resources);

  [[nodiscard]] uint64_t frameIndex() const noexcept { return frameIndex_; }
  [[nodiscard]] uint64_t resourceVersion() const noexcept {
    return resourceVersion_;
  }
  [[nodiscard]] const SimulationFrameGpuResources &
  frameResources() const noexcept {
    return frameResources_;
  }

private:
  uint64_t frameIndex_ = 0u;
  uint64_t resourceVersion_ = 0u;
  SimulationFrameGpuResources frameResources_{};
};

} // namespace nuri
