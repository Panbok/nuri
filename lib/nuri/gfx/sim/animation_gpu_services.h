#pragma once

#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/pipeline.h"
#include "nuri/gfx/shader.h"
#include "nuri/resources/gpu/buffer.h"

#include <filesystem>
#include <memory>
#include <memory_resource>
#include <string>

namespace nuri {

class NURI_API AnimationGpuServices {
public:
  explicit AnimationGpuServices(
      GPUDevice &gpu, std::filesystem::path shaderRoot,
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  ~AnimationGpuServices();

  AnimationGpuServices(const AnimationGpuServices &) = delete;
  AnimationGpuServices &operator=(const AnimationGpuServices &) = delete;
  AnimationGpuServices(AnimationGpuServices &&) = delete;
  AnimationGpuServices &operator=(AnimationGpuServices &&) = delete;

  [[nodiscard]] Result<bool, std::string> ensureInitialized();
  [[nodiscard]] GPUDevice &gpu() noexcept { return gpu_; }
  [[nodiscard]] const std::filesystem::path &shaderRoot() const noexcept {
    return shaderRoot_;
  }
  [[nodiscard]] ComputePipelineHandle samplePipeline() const noexcept {
    return samplePipelineHandle_;
  }
  [[nodiscard]] ComputePipelineHandle worldPipeline() const noexcept {
    return worldPipelineHandle_;
  }
  [[nodiscard]] ComputePipelineHandle scatterPipeline() const noexcept {
    return scatterPipelineHandle_;
  }
  [[nodiscard]] ComputePipelineHandle morphPipeline() const noexcept {
    return morphPipelineHandle_;
  }
  [[nodiscard]] ComputePipelineHandle skinPalettePipeline() const noexcept {
    return skinPalettePipelineHandle_;
  }
  [[nodiscard]] ComputePipelineHandle skinPipeline() const noexcept {
    return skinPipelineHandle_;
  }

  [[nodiscard]] Result<std::unique_ptr<Buffer>, std::string>
  createStorageBuffer(size_t sizeBytes, std::string_view debugName);
  [[nodiscard]] Result<std::unique_ptr<Buffer>, std::string>
  createStorageVertexBuffer(size_t sizeBytes, std::string_view debugName);

private:
  [[nodiscard]] Result<bool, std::string> createShaders();
  [[nodiscard]] Result<bool, std::string> createPipelines();
  void destroyPipelines() noexcept;
  void destroyShaders() noexcept;

  GPUDevice &gpu_;
  std::filesystem::path shaderRoot_;
  std::pmr::memory_resource *memory_ = std::pmr::get_default_resource();
  std::unique_ptr<Shader> shader_;
  std::unique_ptr<Pipeline> samplePipeline_;
  std::unique_ptr<Pipeline> worldPipeline_;
  std::unique_ptr<Pipeline> scatterPipeline_;
  std::unique_ptr<Pipeline> morphPipeline_;
  std::unique_ptr<Pipeline> skinPalettePipeline_;
  std::unique_ptr<Pipeline> skinPipeline_;
  ShaderHandle sampleShaderHandle_{};
  ShaderHandle worldShaderHandle_{};
  ShaderHandle scatterShaderHandle_{};
  ShaderHandle morphShaderHandle_{};
  ShaderHandle skinPaletteShaderHandle_{};
  ShaderHandle skinShaderHandle_{};
  ComputePipelineHandle samplePipelineHandle_{};
  ComputePipelineHandle worldPipelineHandle_{};
  ComputePipelineHandle scatterPipelineHandle_{};
  ComputePipelineHandle morphPipelineHandle_{};
  ComputePipelineHandle skinPalettePipelineHandle_{};
  ComputePipelineHandle skinPipelineHandle_{};
  bool initialized_ = false;
};

} // namespace nuri
