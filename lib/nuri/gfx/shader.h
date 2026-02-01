#pragma once

#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_types.h"

#include <array>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>


namespace nuri {

class GPUDevice;

class NURI_API Shader {
public:
  Shader(std::string_view moduleName, GPUDevice &gpu);
  ~Shader();

  Shader(const Shader &) = delete;
  Shader &operator=(const Shader &) = delete;
  Shader(Shader &&) = delete;
  Shader &operator=(Shader &&) = delete;

  static std::unique_ptr<Shader> create(std::string_view moduleName,
                                        GPUDevice &gpu) {
    return std::make_unique<Shader>(moduleName, gpu);
  }

  Result<std::string, std::string> load(std::string_view path);

  Result<ShaderHandle, std::string> compile(const std::string &code,
                                            ShaderStage stage);

  Result<ShaderHandle, std::string> compileFromFile(std::string_view path,
                                                    ShaderStage stage);

  [[nodiscard]] ShaderHandle getHandle(ShaderStage stage) const;

private:
  std::string moduleName_;
  GPUDevice &gpu_;
  std::array<ShaderHandle, static_cast<size_t>(ShaderStage::Count)>
      shaderHandles_{};
  std::unordered_map<ShaderStage, std::string> debug_glsl_source_code_;
};

} // namespace nuri
