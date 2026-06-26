#pragma once

#include "nuri/core/result.h"
#include "nuri/core/runtime_config.h"
#include "nuri/defines.h"

#include <memory_resource>
#include <string>

namespace nuri {

class GPUDevice;
class RenderPipeline;

[[nodiscard]] NURI_API Result<bool, std::string>
registerDefaultRenderPipeline(RenderPipeline &pipeline, GPUDevice &gpu,
                              const RuntimeShaderConfig &shaderConfig,
                              std::pmr::memory_resource *memory);

} // namespace nuri
