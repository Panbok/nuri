#pragma once

#include "nuri/core/result.h"
#include "nuri/core/runtime_config.h"
#include "nuri/defines.h"
#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/pipeline/render_feature.h"
#include "nuri/gfx/pipeline/render_feature_pass.h"
#include "nuri/resources/gpu/buffer.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>

namespace nuri {

using CompositeFeatureConfig = RuntimeOpaqueShaderConfig;

class Shader;

class NURI_API CompositePass final : public RenderFeaturePass {
public:
  explicit CompositePass(GPUDevice &gpu, CompositeFeatureConfig config);
  ~CompositePass() override;

  CompositePass(const CompositePass &) = delete;
  CompositePass &operator=(const CompositePass &) = delete;
  CompositePass(CompositePass &&) = delete;
  CompositePass &operator=(CompositePass &&) = delete;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "CompositePass";
  }
  [[nodiscard]] bool isEnabled(const FrameBuildContext &ctx) const override;
  Result<bool, std::string> prepare(FrameBuildContext &ctx) override;
  Result<bool, std::string> build(FrameBuildContext &ctx) override;

private:
  struct PushConstants {
    uint64_t frameDataAddress = 0;
    uint64_t vertexBufferAddress = 0;
    uint64_t instanceMatricesAddress = 0;
    uint64_t instanceRemapAddress = 0;
    uint64_t materialBufferAddress = 0;
    uint64_t instanceCentersPhaseAddress = 0;
    uint64_t instanceBaseMatricesAddress = 0;
    uint32_t instanceCount = 0;
    uint32_t materialIndex = 0;
    float timeSeconds = 0.0f;
    float tessNearDistance = 1.0f;
    float tessFarDistance = 8.0f;
    float tessMinFactor = 1.0f;
    float tessMaxFactor = 1.0f;
    uint32_t debugVisualizationMode = 0;
  };
  static_assert(sizeof(PushConstants) <= 128,
                "CompositePass::PushConstants exceeds Vulkan guarantee");

  Result<bool, std::string> ensureInitialized();
  Result<bool, std::string> createShaders();
  Result<bool, std::string> ensurePipeline();
  Result<bool, std::string> ensureFrameBufferCapacity(size_t requiredBytes);
  void destroyPipelineState();
  void destroyShaders();
  void destroyBuffers();

  GPUDevice &gpu_;
  std::unique_ptr<Shader> shader_;
  std::unique_ptr<Buffer> frameBuffer_;

  ShaderHandle vertexShader_{};
  ShaderHandle fragmentShader_{};
  RenderPipelineHandle pipelineHandle_{};
  Format pipelineColorFormat_ = Format::Count;

  size_t frameBufferCapacityBytes_ = 0;
  bool initialized_ = false;
  bool frameDataUploadValid_ = false;
  bool hasPreparedDraw_ = false;

  ForwardSceneFrameData frameData_{};
  ForwardSceneFrameData uploadedFrameData_{};
  PushConstants pushConstants_{};
  DrawItem drawItem_{};
  TextureHandle sourceFrameColor_{};
  std::filesystem::path vertexPath_{};
  std::filesystem::path fragmentPath_{};
};

class NURI_API CompositeFeature final : public RenderFeature {
public:
  explicit CompositeFeature(GPUDevice &gpu, CompositeFeatureConfig config);
  ~CompositeFeature() override = default;

  CompositeFeature(const CompositeFeature &) = delete;
  CompositeFeature &operator=(const CompositeFeature &) = delete;
  CompositeFeature(CompositeFeature &&) = delete;
  CompositeFeature &operator=(CompositeFeature &&) = delete;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "CompositeFeature";
  }
  [[nodiscard]] std::span<RenderFeaturePass *const> passes() noexcept override;

private:
  CompositePass pass_;
  std::array<RenderFeaturePass *, 1> passes_{&pass_};
};

} // namespace nuri
