#pragma once

#include "nuri/core/log.h"
#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/owned_gpu_resource.h"
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

  [[nodiscard]] Result<void, std::string> ensureInitialized();
  [[nodiscard]] GPUDevice &gpu() noexcept { return gpu_; }
  [[nodiscard]] const std::filesystem::path &shaderRoot() const noexcept {
    return shaderRoot_;
  }
  [[nodiscard]] ComputePipelineHandle samplePipeline() const noexcept {
    assertPipelineHandle(samplePipelineHandle_.get(), "samplePipeline");
    return samplePipelineHandle_.get();
  }
  [[nodiscard]] ComputePipelineHandle blendPipeline() const noexcept {
    assertPipelineHandle(blendPipelineHandle_.get(), "blendPipeline");
    return blendPipelineHandle_.get();
  }
  [[nodiscard]] ComputePipelineHandle worldPipeline() const noexcept {
    assertPipelineHandle(worldPipelineHandle_.get(), "worldPipeline");
    return worldPipelineHandle_.get();
  }
  [[nodiscard]] ComputePipelineHandle scatterPipeline() const noexcept {
    assertPipelineHandle(scatterPipelineHandle_.get(), "scatterPipeline");
    return scatterPipelineHandle_.get();
  }
  [[nodiscard]] ComputePipelineHandle morphPipeline() const noexcept {
    assertPipelineHandle(morphPipelineHandle_.get(), "morphPipeline");
    return morphPipelineHandle_.get();
  }
  [[nodiscard]] ComputePipelineHandle skinPalettePipeline() const noexcept {
    assertPipelineHandle(skinPalettePipelineHandle_.get(),
                         "skinPalettePipeline");
    return skinPalettePipelineHandle_.get();
  }
  [[nodiscard]] ComputePipelineHandle skinPipeline() const noexcept {
    assertPipelineHandle(skinPipelineHandle_.get(), "skinPipeline");
    return skinPipelineHandle_.get();
  }

  [[nodiscard]] Result<std::unique_ptr<Buffer>, std::string>
  createStorageBuffer(size_t sizeBytes, std::string_view debugName);
  [[nodiscard]] Result<std::unique_ptr<Buffer>, std::string>
  createStorageVertexBuffer(size_t sizeBytes, std::string_view debugName);

private:
  void assertPipelineHandle(ComputePipelineHandle handle,
                            std::string_view accessorName) const noexcept {
    NURI_ASSERT(initialized_,
                "AnimationGpuServices::%.*s: ensureInitialized() must succeed "
                "before accessing pipelines",
                static_cast<int>(accessorName.size()), accessorName.data());
    NURI_ASSERT(nuri::isValid(handle),
                "AnimationGpuServices::%.*s: pipeline handle is invalid",
                static_cast<int>(accessorName.size()), accessorName.data());
  }

  [[nodiscard]] Result<bool, std::string> createShaders();
  [[nodiscard]] Result<bool, std::string> createPipelines();
  void destroyPipelines() noexcept;
  void destroyShaders() noexcept;

  GPUDevice &gpu_;
  std::filesystem::path shaderRoot_;
  std::pmr::memory_resource *memory_ = std::pmr::get_default_resource();
  std::unique_ptr<Shader> shader_;
  OwnedShaderHandle sampleShaderHandle_{};
  OwnedShaderHandle blendShaderHandle_{};
  OwnedShaderHandle worldShaderHandle_{};
  OwnedShaderHandle scatterShaderHandle_{};
  OwnedShaderHandle morphShaderHandle_{};
  OwnedShaderHandle skinPaletteShaderHandle_{};
  OwnedShaderHandle skinShaderHandle_{};
  OwnedComputePipelineHandle samplePipelineHandle_{};
  OwnedComputePipelineHandle blendPipelineHandle_{};
  OwnedComputePipelineHandle worldPipelineHandle_{};
  OwnedComputePipelineHandle scatterPipelineHandle_{};
  OwnedComputePipelineHandle morphPipelineHandle_{};
  OwnedComputePipelineHandle skinPalettePipelineHandle_{};
  OwnedComputePipelineHandle skinPipelineHandle_{};
  bool initialized_ = false;
};

} // namespace nuri
