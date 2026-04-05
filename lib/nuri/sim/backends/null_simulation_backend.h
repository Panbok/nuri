#pragma once

#include "nuri/sim/backends/simulation_backend.h"

namespace nuri {

class NURI_API NullSimulationBackend final : public ISimulationBackend {
public:
  [[nodiscard]] SimulationKind kind() const noexcept override {
    return SimulationKind::Unknown;
  }

  [[nodiscard]] Result<bool, std::string>
  createInstance(SceneRuntimeHost &host, SimulationHandle handle,
                 const SimulationDesc &desc) override;
  [[nodiscard]] bool destroyInstance(SceneRuntimeHost &host,
                                     SimulationHandle handle) override;
  [[nodiscard]] Result<bool, std::string>
  updateParams(SceneRuntimeHost &host, SimulationHandle handle,
               std::span<const std::byte> params) override;
  [[nodiscard]] Result<bool, std::string>
  executePhase(SceneRuntimeHost &host, SimulationHandle handle,
               SimulationPhase phase,
               const SimulationExecutionContext &context) override;
};

} // namespace nuri
