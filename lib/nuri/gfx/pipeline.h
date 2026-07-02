#pragma once

#include "nuri/core/log.h"
#include "nuri/core/result.h"
#include "nuri/gfx/gpu_device.h"

#include <memory>
#include <string>
#include <string_view>

namespace nuri {

enum class PipelineType : uint8_t {
  Graphics,
  Compute,
  Meshlet,
  Count,
};

class Pipeline {
public:
  explicit Pipeline(GPUDevice &gpu)
      : gpu_(gpu), type_(PipelineType::Count), renderPipeline_{},
        computePipeline_{}, meshletPipeline_{} {}

  ~Pipeline() {
    if (type_ == PipelineType::Graphics && nuri::isValid(renderPipeline_)) {
      gpu_.destroyRenderPipeline(renderPipeline_);
    } else if (type_ == PipelineType::Compute &&
               nuri::isValid(computePipeline_)) {
      gpu_.destroyComputePipeline(computePipeline_);
    } else if (type_ == PipelineType::Meshlet &&
               nuri::isValid(meshletPipeline_)) {
      gpu_.destroyMeshletPipeline(meshletPipeline_);
    }
  }

  Pipeline(const Pipeline &) = delete;
  Pipeline &operator=(const Pipeline &) = delete;
  Pipeline(Pipeline &&) = delete;
  Pipeline &operator=(Pipeline &&) = delete;

  static std::unique_ptr<Pipeline> create(GPUDevice &gpu) {
    return std::make_unique<Pipeline>(gpu);
  }

  Result<RenderPipelineHandle, std::string>
  createRenderPipeline(const RenderPipelineDesc &desc,
                       std::string_view debugName = {}) {
    if (isCreated()) {
      return Result<RenderPipelineHandle, std::string>::makeError(
          "Pipeline already created");
    }

    auto result = gpu_.createRenderPipeline(desc, debugName);
    if (result.hasError()) {
      return Result<RenderPipelineHandle, std::string>::makeError(
          result.error());
    }

    renderPipeline_ = result.value();
    type_ = PipelineType::Graphics;
    return Result<RenderPipelineHandle, std::string>::makeResult(
        renderPipeline_);
  }

  Result<ComputePipelineHandle, std::string>
  createComputePipeline(const ComputePipelineDesc &desc,
                        std::string_view debugName = {}) {
    if (isCreated()) {
      return Result<ComputePipelineHandle, std::string>::makeError(
          "Pipeline already created");
    }

    auto result = gpu_.createComputePipeline(desc, debugName);
    if (result.hasError()) {
      return Result<ComputePipelineHandle, std::string>::makeError(
          result.error());
    }

    computePipeline_ = result.value();
    type_ = PipelineType::Compute;
    return Result<ComputePipelineHandle, std::string>::makeResult(
        computePipeline_);
  }

  Result<MeshletPipelineHandle, std::string>
  createMeshletPipeline(const MeshletPipelineDesc &desc,
                        std::string_view debugName = {}) {
    if (isCreated()) {
      return Result<MeshletPipelineHandle, std::string>::makeError(
          "Pipeline already created");
    }

    auto result = gpu_.createMeshletPipeline(desc, debugName);
    if (result.hasError()) {
      return Result<MeshletPipelineHandle, std::string>::makeError(
          result.error());
    }

    meshletPipeline_ = result.value();
    type_ = PipelineType::Meshlet;
    return Result<MeshletPipelineHandle, std::string>::makeResult(
        meshletPipeline_);
  }

  [[nodiscard]] RenderPipelineHandle getRenderPipeline() const {
    NURI_ASSERT(type_ == PipelineType::Graphics,
                "Pipeline type is not Graphics");
    return renderPipeline_;
  }

  [[nodiscard]] ComputePipelineHandle getComputePipeline() const {
    NURI_ASSERT(type_ == PipelineType::Compute, "Pipeline type is not Compute");
    return computePipeline_;
  }

  [[nodiscard]] MeshletPipelineHandle getMeshletPipeline() const {
    NURI_ASSERT(type_ == PipelineType::Meshlet, "Pipeline type is not Meshlet");
    return meshletPipeline_;
  }

  [[nodiscard]] PipelineType getType() const { return type_; }
  [[nodiscard]] bool isCreated() const { return type_ != PipelineType::Count; }

private:
  GPUDevice &gpu_;
  PipelineType type_ = PipelineType::Count;
  RenderPipelineHandle renderPipeline_{};
  ComputePipelineHandle computePipeline_{};
  MeshletPipelineHandle meshletPipeline_{};
};

} // namespace nuri
