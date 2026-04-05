#pragma once

#include <cstddef>
#include <span>
#include <string>

#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/sim/simulation_desc.h"
#include "nuri/sim/simulation_execution_context.h"

namespace nuri {

class SceneRuntimeHost;

class NURI_API ISimulationBackend {
public:
  virtual ~ISimulationBackend() = default;

  [[nodiscard]] virtual SimulationKind kind() const noexcept = 0;
  [[nodiscard]] virtual Result<bool, std::string>
  createInstance(SceneRuntimeHost &host, SimulationHandle handle,
                 const SimulationDesc &desc) = 0;
  [[nodiscard]] virtual Result<bool, std::string>
  destroyInstance(SceneRuntimeHost &host, SimulationHandle handle) = 0;
  [[nodiscard]] virtual Result<bool, std::string>
  updateParams(SceneRuntimeHost &host, SimulationHandle handle,
               std::span<const std::byte> params) = 0;
  [[nodiscard]] virtual Result<bool, std::string>
  executePhase(SceneRuntimeHost &host, SimulationHandle handle,
               SimulationPhase phase,
               const SimulationExecutionContext &context) = 0;
};

} // namespace nuri
