#pragma once
#include "nuri/core/result.h"
#include "nuri/core/runtime_config.h"
#include "nuri/defines.h"
#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/pipeline/render_pipeline.h"
#include "nuri/resources/gpu/texture.h"
#include <array>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <span>
#include <vector>
namespace nuri {

class RenderPipeline;

using FrameCompositionFeatureConfig = RuntimeCompositeConfig;

struct NURI_API FullscreenPassResources {
  ShaderHandle vertexShader{};
  ShaderHandle fragmentShader{};
  RenderPipelineHandle pipelineHandle{};
  Format pipelineColorFormat = Format::Count;
  std::filesystem::path vertexPath{};
  std::filesystem::path fragmentPath{};
};

struct NURI_API CopyPushConstants {
  uint32_t sourceTexId = 0u;
  uint32_t sourceSamplerId = 0u;
  uint32_t flags = 0u;
};
static_assert(sizeof(CopyPushConstants) <= 128);

class NURI_API FullscreenRenderPass {
public:
  ~FullscreenRenderPass();
  FullscreenRenderPass(const FullscreenRenderPass &) = delete;
  FullscreenRenderPass &operator=(const FullscreenRenderPass &) = delete;

protected:
  explicit FullscreenRenderPass(GPUDevice &gpu);
  Result<bool, std::string> ensureInitialized(std::string_view shaderName);
  Result<bool, std::string> ensurePipeline(Format colorFormat,
                                           std::string_view debugName);
  GPUDevice &gpu_;
  FullscreenPassResources resources_{};
};

class NURI_API SceneColorDownsamplePass final : public FullscreenRenderPass {
public:
  explicit SceneColorDownsamplePass(GPUDevice &gpu,
                                    FrameCompositionFeatureConfig config);
  [[nodiscard]] bool isEnabled(const FrameBuildContext &ctx) const;
  Result<bool, std::string> prepare(FrameBuildContext &ctx);
  Result<bool, std::string> build(FrameBuildContext &ctx);
};

class NURI_API SceneResolvePass final : public FullscreenRenderPass {
public:
  explicit SceneResolvePass(GPUDevice &gpu,
                            FrameCompositionFeatureConfig config);
  [[nodiscard]] bool isEnabled(const FrameBuildContext &ctx) const;
  Result<bool, std::string> prepare(FrameBuildContext &ctx);
  Result<bool, std::string> build(FrameBuildContext &ctx);
};

class NURI_API PresentToneMapPass final : public FullscreenRenderPass {
public:
  explicit PresentToneMapPass(GPUDevice &gpu,
                              FrameCompositionFeatureConfig config);
  ~PresentToneMapPass();
  [[nodiscard]] bool isEnabled(const FrameBuildContext &ctx) const;
  Result<bool, std::string> prepare(FrameBuildContext &ctx);
  Result<bool, std::string> build(FrameBuildContext &ctx);

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
  };
  Result<bool, std::string> ensureToneMapSampler();
  void ensureToneMapLutLoaded(ToneMapLutResource &resource,
                              std::string_view debugName);
  std::array<ToneMapLutResource, 2> toneMapLuts_{};
  SamplerHandle lutSampler_{};
  FullscreenPassResources captureResources_{};
};

class NURI_API HDRExposureAdaptPass final : public FullscreenRenderPass {
public:
  explicit HDRExposureAdaptPass(GPUDevice &gpu,
                                FrameCompositionFeatureConfig config);
  ~HDRExposureAdaptPass();
  [[nodiscard]] bool isEnabled(const FrameBuildContext &ctx) const;
  Result<bool, std::string> publishFrameData(FrameBuildContext &ctx);
  Result<bool, std::string> prepare(FrameBuildContext &ctx);
  Result<bool, std::string> build(FrameBuildContext &ctx);
  void onFrameSubmitted(const RenderFrameContext &frame) noexcept;
  void onFrameAbandoned(const RenderFrameContext &frame) noexcept;

private:
  enum class TelemetrySlotState : uint8_t {
    Free = 0,
    Recording,
    Pending,
    Consumed,
    Dropped,
  };
  struct TelemetrySlot {
    OwnedBufferHandle device{};
    OwnedBufferHandle readback{};
    SubmissionHandle submission{};
    TelemetrySlotState state = TelemetrySlotState::Free;
    uint64_t sceneId = 0u;
    uint64_t sourceFrame = 0u;
  };
  Result<bool, std::string> ensureTelemetryRing();
  void collectCompletedTelemetry(FrameBuildContext &ctx);
  TelemetrySlot *acquireTelemetrySlot(FrameBuildContext &ctx);
  Result<bool, std::string> ensureLuminanceTextures();
  void destroyLuminanceTextures();
  FullscreenPassResources luminanceResources_{};
  std::vector<std::vector<TextureHandle>> luminanceTextures_{};
  uint32_t luminanceWidth_ = 0u;
  uint32_t luminanceHeight_ = 0u;
  uint32_t luminanceRingCount_ = 0u;
  uint32_t luminanceMipCount_ = 0u;
  std::vector<TelemetrySlot> telemetrySlots_{};
  uint64_t latestTelemetrySourceFrame_ = 0u;
  uint64_t latestTelemetrySceneId_ = 0u;
  float latestAdaptedExposureEv_ = 0.0f;
  float latestAutomaticExposureEv_ = 0.0f;
  float latestTargetExposureEv_ = 0.0f;
  float latestMeteredLuminance_ = 0.0f;
  float latestInvalidFraction_ = 0.0f;
  uint32_t telemetryDroppedSamples_ = 0u;
  size_t activeTelemetrySlot_ = std::numeric_limits<size_t>::max();
  bool latestTelemetryAvailable_ = false;
  bool telemetryRingReady_ = false;
  bool telemetryScheduled_ = false;
};

class NURI_API HDRBloomCompositePass final : public FullscreenRenderPass {
public:
  explicit HDRBloomCompositePass(GPUDevice &gpu,
                                 FrameCompositionFeatureConfig config);
  ~HDRBloomCompositePass();
  [[nodiscard]] bool isEnabled(const FrameBuildContext &ctx) const;
  Result<bool, std::string> prepare(FrameBuildContext &ctx);
  Result<bool, std::string> build(FrameBuildContext &ctx);

private:
  FullscreenPassResources bloomResources_{};
};

NURI_API void
registerFrameCompositionStages(RenderPipeline &pipeline, GPUDevice &gpu,
                               FrameCompositionFeatureConfig config);
NURI_API void
registerHDRPostProcessStages(RenderPipeline &pipeline, GPUDevice &gpu,
                             FrameCompositionFeatureConfig config);
NURI_API void registerFramePresentStage(RenderPipeline &pipeline,
                                        GPUDevice &gpu,
                                        FrameCompositionFeatureConfig config);

} // namespace nuri
