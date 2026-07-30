#pragma once
#include "nuri/core/log.h"
#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/pipeline/owned_program_bundle.h"
#include "nuri/resources/gpu/buffer.h"
#include <array>
#include <filesystem>
#include <memory>
#include <memory_resource>
#include <string>
namespace nuri {

class NURI_API AnimationGpuServices {
  enum class Program : uint8_t {
    Sample,
    Blend,
    World,
    Scatter,
    Morph,
    SkinPalette,
    Skin,
    Count,
  };

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
    return pipeline(Program::Sample);
  }
  [[nodiscard]] ComputePipelineHandle blendPipeline() const noexcept {
    return pipeline(Program::Blend);
  }
  [[nodiscard]] ComputePipelineHandle worldPipeline() const noexcept {
    return pipeline(Program::World);
  }
  [[nodiscard]] ComputePipelineHandle scatterPipeline() const noexcept {
    return pipeline(Program::Scatter);
  }
  [[nodiscard]] ComputePipelineHandle morphPipeline() const noexcept {
    return pipeline(Program::Morph);
  }
  [[nodiscard]] ComputePipelineHandle skinPalettePipeline() const noexcept {
    return pipeline(Program::SkinPalette);
  }
  [[nodiscard]] ComputePipelineHandle skinPipeline() const noexcept {
    return pipeline(Program::Skin);
  }
  [[nodiscard]] Result<std::unique_ptr<Buffer>, std::string>
  createStorageBuffer(size_t sizeBytes, std::string_view debugName);
  [[nodiscard]] Result<std::unique_ptr<Buffer>, std::string>
  createStorageVertexBuffer(size_t sizeBytes, std::string_view debugName);

private:
  [[nodiscard]] ComputePipelineHandle pipeline(Program program) const noexcept;
  GPUDevice &gpu_;
  std::filesystem::path shaderRoot_;
  std::pmr::memory_resource *memory_ = std::pmr::get_default_resource();
  static constexpr size_t kProgramCount = static_cast<size_t>(Program::Count);
  OwnedProgramBundle programs_{};
  bool initialized_ = false;
};

} // namespace nuri
