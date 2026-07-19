#pragma once
#include "nuri/core/runtime_config.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/pipeline/render_pipeline.h"
#include <array>
#include <cstdint>
#include <filesystem>
#include <span>
namespace nuri {

class RenderPipeline;

enum class TemporalInputPlacement : uint8_t {
  EarlyGtao = 0u,
  ColorReconstruction = 1u,
};

enum class TemporalInput : uint8_t { MotionVectors, ReactiveMask };

struct TemporalAAPipelineState {
  std::array<ShaderHandle, 2> shaders{};
  RenderPipelineHandle pipeline{};
};

class NURI_API TemporalAAClearPass final {
public:
  TemporalAAClearPass(TemporalInput input, TemporalInputPlacement placement)
      : input_(input), placement_(placement) {}
  [[nodiscard]] bool isEnabled(const FrameBuildContext &ctx) const;
  Result<bool, std::string> publishFrameData(FrameBuildContext &ctx);
  Result<bool, std::string> prepare(FrameBuildContext &ctx);
  Result<bool, std::string> build(FrameBuildContext &ctx);

private:
  TemporalInput input_ = TemporalInput::MotionVectors;
  TemporalInputPlacement placement_ =
      TemporalInputPlacement::ColorReconstruction;
};

class NURI_API TemporalAABackgroundMotionPass final {
public:
  explicit TemporalAABackgroundMotionPass(GPUDevice &gpu,
                                          RuntimeCompositeConfig config,
                                          TemporalInputPlacement placement);
  ~TemporalAABackgroundMotionPass();
  [[nodiscard]] bool isEnabled(const FrameBuildContext &ctx) const;
  Result<bool, std::string> prepare(FrameBuildContext &ctx);
  Result<bool, std::string> build(FrameBuildContext &ctx);

private:
  GPUDevice &gpu_;
  TemporalAAPipelineState pipeline_{};
  std::filesystem::path vertexPath_{};
  std::filesystem::path fragmentPath_{};
  bool initialized_ = false;
  TemporalInputPlacement placement_ =
      TemporalInputPlacement::ColorReconstruction;
};

class NURI_API TemporalAAMotionClassPass final {
public:
  explicit TemporalAAMotionClassPass(GPUDevice &gpu,
                                     RuntimeCompositeConfig config,
                                     TemporalInputPlacement placement);
  ~TemporalAAMotionClassPass();
  [[nodiscard]] bool isEnabled(const FrameBuildContext &ctx) const;
  Result<bool, std::string> prepare(FrameBuildContext &ctx);
  Result<bool, std::string> build(FrameBuildContext &ctx);

private:
  GPUDevice &gpu_;
  TemporalAAPipelineState pipeline_{};
  std::filesystem::path vertexPath_{};
  std::filesystem::path fragmentPath_{};
  bool initialized_ = false;
  TemporalInputPlacement placement_ =
      TemporalInputPlacement::ColorReconstruction;
};

class NURI_API TemporalAAResolvePass final {
public:
  explicit TemporalAAResolvePass(GPUDevice &gpu, RuntimeCompositeConfig config);
  ~TemporalAAResolvePass();
  [[nodiscard]] bool isEnabled(const FrameBuildContext &ctx) const;
  Result<bool, std::string> prepare(FrameBuildContext &ctx);
  Result<bool, std::string> build(FrameBuildContext &ctx);

private:
  GPUDevice &gpu_;
  TemporalAAPipelineState pipeline_{};
  SamplerHandle linearClampSampler_{};
  std::filesystem::path vertexPath_{};
  std::filesystem::path fragmentPath_{};
  bool initialized_ = false;
};

NURI_API void registerTemporalInputStages(RenderPipeline &pipeline,
                                          GPUDevice &gpu,
                                          RuntimeCompositeConfig config);
NURI_API void registerTemporalAAStages(RenderPipeline &pipeline, GPUDevice &gpu,
                                       RuntimeCompositeConfig config);

} // namespace nuri
