#pragma once
#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_types.h"
#include <string>
#include <string_view>
namespace nuri {

class GPUDevice;

[[nodiscard]] NURI_API Result<ShaderHandle, std::string>
compileShader(GPUDevice &gpu, std::string_view moduleName,
              std::string_view code, ShaderStage stage);

[[nodiscard]] NURI_API Result<ShaderHandle, std::string>
compileShaderFile(GPUDevice &gpu, std::string_view moduleName,
                  std::string_view path, ShaderStage stage,
                  std::string_view preamble = {});

} // namespace nuri
