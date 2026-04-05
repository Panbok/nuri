#include "nuri/pch.h"

#include "nuri/sim/backends/null_simulation_backend.h"

#include "nuri/scene_runtime/scene_runtime_host.h"

namespace nuri {

Result<bool, std::string>
NullSimulationBackend::createInstance(SceneRuntimeHost &, SimulationHandle,
                                      const SimulationDesc &) {
  return Result<bool, std::string>::makeResult(true);
}

bool NullSimulationBackend::destroyInstance(SceneRuntimeHost &,
                                            SimulationHandle) {
  return true;
}

Result<bool, std::string>
NullSimulationBackend::updateParams(SceneRuntimeHost &, SimulationHandle,
                                    std::span<const std::byte>) {
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
NullSimulationBackend::executePhase(SceneRuntimeHost &, SimulationHandle,
                                    SimulationPhase,
                                    const SimulationExecutionContext &) {
  return Result<bool, std::string>::makeResult(true);
}

} // namespace nuri
