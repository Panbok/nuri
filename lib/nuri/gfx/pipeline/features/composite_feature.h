#pragma once

#include "nuri/core/result.h"
#include "nuri/core/runtime_config.h"
#include "nuri/defines.h"
#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/pipeline/render_feature.h"
#include "nuri/gfx/pipeline/render_feature_pass.h"
#include "nuri/gfx/shader.h"
#include "nuri/resources/gpu/texture.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace nuri {

using FrameCompositionFeatureConfig = RuntimeCompositeConfig;

struct NURI_API FullscreenPassResources {
  std::unique_ptr<Shader> shader{};
  ShaderHandle vertexShader{};
  ShaderHandle fragmentShader{};
  RenderPipelineHandle pipelineHandle{};
  Format pipelineColorFormat = Format::Count;
  bool initialized = false;
  std::filesystem::path vertexPath{};
  std::filesystem::path fragmentPath{};
};

struct NURI_API CopyPushConstants {
  uint32_t sourceTexId = 0u;
  uint32_t sourceSamplerId = 0u;
  uint32_t flags = 0u;
  uint32_t reserved0 = 0u;
};
static_assert(sizeof(CopyPushConstants) <= 128);

class NURI_API FullscreenRenderPass : public RenderFeaturePass {
public:
  ~FullscreenRenderPass() override;

protected:
  explicit FullscreenRenderPass(GPUDevice &gpu);

  Result<bool, std::string> ensureInitialized(std::string_view shaderName,
                                              std::string_view errorContext);
  Result<bool, std::string> ensurePipeline(Format colorFormat,
                                           std::string_view debugName,
                                           std::string_view errorContext);
  Result<bool, std::string> createShaders(std::string_view shaderName,
                                          std::string_view errorContext);
  void destroyPipeline();
  void destroyShaders();

  GPUDevice &gpu_;
  FullscreenPassResources resources_{};
};

class NURI_API SceneColorDownsamplePass final : public FullscreenRenderPass {
public:
  explicit SceneColorDownsamplePass(GPUDevice &gpu,
                                    FrameCompositionFeatureConfig config);
  ~SceneColorDownsamplePass() override = default;

  SceneColorDownsamplePass(const SceneColorDownsamplePass &) = delete;
  SceneColorDownsamplePass &
  operator=(const SceneColorDownsamplePass &) = delete;
  SceneColorDownsamplePass(SceneColorDownsamplePass &&) = delete;
  SceneColorDownsamplePass &operator=(SceneColorDownsamplePass &&) = delete;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "SceneColorDownsamplePass";
  }
  [[nodiscard]] bool isEnabled(const FrameBuildContext &ctx) const override;
  Result<bool, std::string> prepare(FrameBuildContext &ctx) override;
  Result<bool, std::string> build(FrameBuildContext &ctx) override;
};

class NURI_API SceneResolvePass final : public FullscreenRenderPass {
public:
  explicit SceneResolvePass(GPUDevice &gpu,
                            FrameCompositionFeatureConfig config);
  ~SceneResolvePass() override = default;

  SceneResolvePass(const SceneResolvePass &) = delete;
  SceneResolvePass &operator=(const SceneResolvePass &) = delete;
  SceneResolvePass(SceneResolvePass &&) = delete;
  SceneResolvePass &operator=(SceneResolvePass &&) = delete;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "SceneResolvePass";
  }
  [[nodiscard]] bool isEnabled(const FrameBuildContext &ctx) const override;
  Result<bool, std::string> prepare(FrameBuildContext &ctx) override;
  Result<bool, std::string> build(FrameBuildContext &ctx) override;
};

class NURI_API PresentToneMapPass final : public FullscreenRenderPass {
public:
  explicit PresentToneMapPass(GPUDevice &gpu,
                              FrameCompositionFeatureConfig config);
  ~PresentToneMapPass() override;

  PresentToneMapPass(const PresentToneMapPass &) = delete;
  PresentToneMapPass &operator=(const PresentToneMapPass &) = delete;
  PresentToneMapPass(PresentToneMapPass &&) = delete;
  PresentToneMapPass &operator=(PresentToneMapPass &&) = delete;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "PresentToneMapPass";
  }
  [[nodiscard]] bool isEnabled(const FrameBuildContext &ctx) const override;
  Result<bool, std::string> prepare(FrameBuildContext &ctx) override;
  Result<bool, std::string> build(FrameBuildContext &ctx) override;

private:
  static constexpr float kShaperMinLog2 = -10.0f;
  static constexpr float kShaperMaxLog2 = 6.5f;
  static constexpr float kShaperInvRange =
      1.0f / (kShaperMaxLog2 - kShaperMinLog2);

  struct PushConstants {
    uint32_t sourceTexId = 0u;
    uint32_t sourceSamplerId = 0u;
    uint32_t acesLutTexId = 0u;
    uint32_t agxLutTexId = 0u;
    uint32_t lutSamplerId = 0u;
    uint32_t flags = 0u;
    float acesExposureScale = 1.0f;
    float agxExposureScale = 1.0f;
    float compareSplit = 0.5f;
    float shaperMinLog2 = kShaperMinLog2;
    float shaperInvRange = kShaperInvRange;
  };
  static_assert(sizeof(PushConstants) <= 128);

  struct ToneMapLutResource {
    std::filesystem::path path{};
    std::unique_ptr<Texture> texture{};
    bool loadAttempted = false;
    bool warned = false;
  };

  Result<bool, std::string> ensureToneMapAssetsLoaded();
  Result<bool, std::string> ensureToneMapSampler();
  Result<bool, std::string> ensureToneMapLutLoaded(ToneMapLutResource &resource,
                                                   std::string_view debugName);
  void destroyToneMapAssets();

  ToneMapLutResource aces2SdrLut_{};
  ToneMapLutResource agxLut_{};
  SamplerHandle lutSampler_{};
};

class NURI_API FrameCompositionFeature final : public RenderFeature {
public:
  explicit FrameCompositionFeature(GPUDevice &gpu,
                                   FrameCompositionFeatureConfig config);
  ~FrameCompositionFeature() override = default;

  FrameCompositionFeature(const FrameCompositionFeature &) = delete;
  FrameCompositionFeature &operator=(const FrameCompositionFeature &) = delete;
  FrameCompositionFeature(FrameCompositionFeature &&) = delete;
  FrameCompositionFeature &operator=(FrameCompositionFeature &&) = delete;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "FrameCompositionFeature";
  }
  [[nodiscard]] std::span<RenderFeaturePass *const> passes() noexcept override;

private:
  SceneColorDownsamplePass downsamplePass_;
  SceneResolvePass resolvePass_;
  std::array<RenderFeaturePass *, 2> passes_{&downsamplePass_, &resolvePass_};
};

class NURI_API HDRExposureAdaptPass final : public FullscreenRenderPass {
public:
  explicit HDRExposureAdaptPass(GPUDevice &gpu,
                                FrameCompositionFeatureConfig config);
  ~HDRExposureAdaptPass() override = default;

  HDRExposureAdaptPass(const HDRExposureAdaptPass &) = delete;
  HDRExposureAdaptPass &operator=(const HDRExposureAdaptPass &) = delete;
  HDRExposureAdaptPass(HDRExposureAdaptPass &&) = delete;
  HDRExposureAdaptPass &operator=(HDRExposureAdaptPass &&) = delete;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "HDRExposureAdaptPass";
  }
  [[nodiscard]] bool isEnabled(const FrameBuildContext &ctx) const override;
  Result<bool, std::string> prepare(FrameBuildContext &ctx) override;
  Result<bool, std::string> build(FrameBuildContext &ctx) override;
};

class NURI_API HDRBloomCompositePass final : public FullscreenRenderPass {
public:
  explicit HDRBloomCompositePass(GPUDevice &gpu,
                                 FrameCompositionFeatureConfig config);
  ~HDRBloomCompositePass() override;

  HDRBloomCompositePass(const HDRBloomCompositePass &) = delete;
  HDRBloomCompositePass &operator=(const HDRBloomCompositePass &) = delete;
  HDRBloomCompositePass(HDRBloomCompositePass &&) = delete;
  HDRBloomCompositePass &operator=(HDRBloomCompositePass &&) = delete;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "HDRBloomCompositePass";
  }
  [[nodiscard]] bool isEnabled(const FrameBuildContext &ctx) const override;
  Result<bool, std::string> prepare(FrameBuildContext &ctx) override;
  Result<bool, std::string> build(FrameBuildContext &ctx) override;

private:
  Result<bool, std::string> ensureOutputTextures();
  Result<bool, std::string> ensureBloomTextures(uint32_t requestedMipCount);
  void destroyOutputTextures();
  void destroyBloomTextures();

  FullscreenPassResources bloomResources_{};
  std::vector<TextureHandle> outputTextures_{};
  std::vector<std::vector<TextureHandle>> bloomDownsampleTextures_{};
  std::vector<std::vector<TextureHandle>> bloomUpsampleTextures_{};
  uint32_t outputWidth_ = 0u;
  uint32_t outputHeight_ = 0u;
  uint32_t outputRingCount_ = 0u;
  uint32_t bloomWidth_ = 0u;
  uint32_t bloomHeight_ = 0u;
  uint32_t bloomRingCount_ = 0u;
  uint32_t bloomMipCount_ = 0u;
};

class NURI_API HDRPostProcessFeature final : public RenderFeature {
public:
  explicit HDRPostProcessFeature(GPUDevice &gpu,
                                 FrameCompositionFeatureConfig config);
  ~HDRPostProcessFeature() override = default;

  HDRPostProcessFeature(const HDRPostProcessFeature &) = delete;
  HDRPostProcessFeature &operator=(const HDRPostProcessFeature &) = delete;
  HDRPostProcessFeature(HDRPostProcessFeature &&) = delete;
  HDRPostProcessFeature &operator=(HDRPostProcessFeature &&) = delete;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "HDRPostProcessFeature";
  }
  [[nodiscard]] Result<bool, std::string>
  publishFrameData(FrameBuildContext &ctx) override;
  [[nodiscard]] std::span<RenderFeaturePass *const> passes() noexcept override;

private:
  HDRExposureAdaptPass exposurePass_;
  HDRBloomCompositePass compositePass_;
  std::array<RenderFeaturePass *, 2> passes_{&exposurePass_, &compositePass_};
};

class NURI_API FramePresentFeature final : public RenderFeature {
public:
  explicit FramePresentFeature(GPUDevice &gpu,
                               FrameCompositionFeatureConfig config);
  ~FramePresentFeature() override = default;

  FramePresentFeature(const FramePresentFeature &) = delete;
  FramePresentFeature &operator=(const FramePresentFeature &) = delete;
  FramePresentFeature(FramePresentFeature &&) = delete;
  FramePresentFeature &operator=(FramePresentFeature &&) = delete;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "FramePresentFeature";
  }
  [[nodiscard]] bool isTerminalFeature() const noexcept override {
    return true;
  }
  [[nodiscard]] std::span<RenderFeaturePass *const> passes() noexcept override;

private:
  PresentToneMapPass presentPass_;
  std::array<RenderFeaturePass *, 1> passes_{&presentPass_};
};

using CompositeFeature = FramePresentFeature;

} // namespace nuri
