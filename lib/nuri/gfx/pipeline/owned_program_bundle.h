#pragma once

#include "nuri/core/result.h"
#include "nuri/gfx/owned_gpu_resource.h"
#include "nuri/gfx/shader.h"

#include <filesystem>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace nuri {

struct ShaderSpec {
  std::string_view debugName{};
  std::filesystem::path path{};
  ShaderStage stage = ShaderStage::Vertex;
};

struct GraphicsPipelineSpec {
  std::string_view debugName{};
  RenderPipelineDesc desc{};
};

struct ComputePipelineSpec {
  std::string_view debugName{};
  ComputePipelineDesc desc{};
};

class OwnedProgramBundle final {
public:
  OwnedProgramBundle() = default;
  ~OwnedProgramBundle() { reset(); }
  OwnedProgramBundle(const OwnedProgramBundle &) = delete;
  OwnedProgramBundle &operator=(const OwnedProgramBundle &) = delete;
  OwnedProgramBundle(OwnedProgramBundle &&) = default;
  OwnedProgramBundle &operator=(OwnedProgramBundle &&other) noexcept {
    if (this != &other) {
      reset();
      shaders_ = std::move(other.shaders_);
      graphicsPipelines_ = std::move(other.graphicsPipelines_);
      computePipelines_ = std::move(other.computePipelines_);
    }
    return *this;
  }

  [[nodiscard]] Result<bool, std::string>
  compileShaders(GPUDevice &gpu, std::span<const ShaderSpec> specs) {
    std::vector<OwnedShaderHandle> created;
    created.reserve(specs.size());
    for (const ShaderSpec &spec : specs) {
      auto shader = compileShaderFile(gpu, spec.debugName, spec.path.string(),
                                      spec.stage);
      if (shader.hasError()) {
        resetReverse(created);
        return Result<bool, std::string>::makeError(shader.error());
      }
      created.emplace_back(gpu, shader.value());
    }
    resetReverse(shaders_);
    shaders_ = std::move(created);
    return Result<bool, std::string>::makeResult(true);
  }

  [[nodiscard]] Result<bool, std::string>
  createShaders(GPUDevice &gpu, std::span<const ShaderDesc> descs) {
    std::vector<OwnedShaderHandle> created;
    created.reserve(descs.size());
    for (const ShaderDesc &desc : descs) {
      auto shader = gpu.createShaderModule(desc);
      if (shader.hasError()) {
        resetReverse(created);
        return Result<bool, std::string>::makeError(shader.error());
      }
      created.emplace_back(gpu, shader.value());
    }
    resetReverse(shaders_);
    shaders_ = std::move(created);
    return Result<bool, std::string>::makeResult(true);
  }

  [[nodiscard]] Result<bool, std::string>
  replaceGraphicsPipelines(GPUDevice &gpu,
                           std::span<const GraphicsPipelineSpec> specs) {
    std::vector<OwnedRenderPipelineHandle> created;
    created.reserve(specs.size());
    for (const GraphicsPipelineSpec &spec : specs) {
      auto pipeline = gpu.createRenderPipeline(spec.desc, spec.debugName);
      if (pipeline.hasError()) {
        resetReverse(created);
        return Result<bool, std::string>::makeError(pipeline.error());
      }
      created.emplace_back(gpu, pipeline.value());
    }
    resetReverse(graphicsPipelines_);
    graphicsPipelines_ = std::move(created);
    return Result<bool, std::string>::makeResult(true);
  }

  [[nodiscard]] Result<bool, std::string>
  replaceComputePipelines(GPUDevice &gpu,
                          std::span<const ComputePipelineSpec> specs) {
    std::vector<OwnedComputePipelineHandle> created;
    created.reserve(specs.size());
    for (const ComputePipelineSpec &spec : specs) {
      auto pipeline = gpu.createComputePipeline(spec.desc, spec.debugName);
      if (pipeline.hasError()) {
        resetReverse(created);
        return Result<bool, std::string>::makeError(pipeline.error());
      }
      created.emplace_back(gpu, pipeline.value());
    }
    resetReverse(computePipelines_);
    computePipelines_ = std::move(created);
    return Result<bool, std::string>::makeResult(true);
  }

  [[nodiscard]] ShaderHandle shader(size_t index) const noexcept {
    return shaders_[index].get();
  }
  [[nodiscard]] RenderPipelineHandle
  graphicsPipeline(size_t index) const noexcept {
    return graphicsPipelines_[index].get();
  }
  [[nodiscard]] ComputePipelineHandle
  computePipeline(size_t index) const noexcept {
    return computePipelines_[index].get();
  }
  [[nodiscard]] size_t shaderCount() const noexcept { return shaders_.size(); }
  [[nodiscard]] size_t graphicsPipelineCount() const noexcept {
    return graphicsPipelines_.size();
  }

  void resetPipelines() noexcept {
    resetReverse(computePipelines_);
    resetReverse(graphicsPipelines_);
  }
  void reset() noexcept {
    resetPipelines();
    resetReverse(shaders_);
  }

private:
  template <typename Owned>
  static void resetReverse(std::vector<Owned> &items) {
    for (auto item = items.rbegin(); item != items.rend(); ++item) {
      item->reset();
    }
    items.clear();
  }

  std::vector<OwnedShaderHandle> shaders_{};
  std::vector<OwnedRenderPipelineHandle> graphicsPipelines_{};
  std::vector<OwnedComputePipelineHandle> computePipelines_{};
};

} // namespace nuri
