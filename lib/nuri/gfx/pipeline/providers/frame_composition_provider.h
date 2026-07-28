#pragma once
#include "nuri/defines.h"
#include "nuri/gfx/frame/history_registry.h"
#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/pipeline/render_pipeline.h"
#include <array>
#include <cstdint>
#include <memory_resource>
#include <string>
#include <string_view>
#include <vector>
namespace nuri {

class NURI_API FrameCompositionProvider final {
public:
  explicit FrameCompositionProvider(
      GPUDevice &gpu,
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  ~FrameCompositionProvider();
  FrameCompositionProvider(const FrameCompositionProvider &) = delete;
  FrameCompositionProvider &
  operator=(const FrameCompositionProvider &) = delete;
  FrameCompositionProvider(FrameCompositionProvider &&) = delete;
  FrameCompositionProvider &operator=(FrameCompositionProvider &&) = delete;
  [[nodiscard]] Result<bool, std::string> prepare(FrameBuildContext &ctx);
  void onFrameSubmitted(const RenderFrameContext &frame) noexcept;
  void onFrameAbandoned(const RenderFrameContext &frame) noexcept;

private:
  using TextureRing = std::pmr::vector<TextureHandle>;
  enum class Ring : uint8_t {
    SceneColor,
    SceneColorHalf,
    SceneColorQuarter,
    FrameColor,
    SceneDepth,
    MsaaSceneColor,
    MsaaSceneDepth,
    MotionVectors,
    ReactiveMask,
    MotionClass,
    Normals,
    AmbientOcclusion,
    DDGIOpaqueSurfaceCache,
    Exposure,
    PresentCapture,
    HistoryColor,
    Count,
  };
  struct RingDesc {
    Ring ring;
    FrameTextureRequirementFlags requirement;
    Format format;
    TextureUsage usage;
    std::string_view name;
    uint8_t mipLevel = 0u;
    uint8_t samples = 1u;
    bool fixedSize = false;
  };
  Result<bool, std::string>
  ensureTextures(FrameTextureRequirementFlags requirements,
                 CoverageMode coverage);
  Result<bool, std::string> recreateTextureRing(const RingDesc &desc);
  void invalidateAllocationState() noexcept;
  void destroyTextureRing(Ring ring);
  [[nodiscard]] TextureHandle currentTexture(Ring ring,
                                             uint64_t index) const noexcept;
  GPUDevice &gpu_;
  std::pmr::memory_resource *memory_ = std::pmr::get_default_resource();
  std::pmr::vector<TextureRing> textureRings_;
  std::array<uint32_t, static_cast<size_t>(Ring::Count)> allocationCounts_{};
  std::array<uint32_t, static_cast<size_t>(Ring::Count)> reallocationCounts_{};
  FrameTextureRequirementFlags allocatedRequirements_ =
      FrameTextureRequirementFlags::None;
  uint32_t textureRingCount_ = 0u;
  uint32_t framebufferWidth_ = 0u;
  uint32_t framebufferHeight_ = 0u;
  CoverageMode allocatedCoverage_ = CoverageMode::Sample1;
  HistoryRegistry historyRegistry_{};
  FrameTextureRequirementFlags pendingHistoryRequirements_ =
      FrameTextureRequirementFlags::None;
  FrameTextureRequirementFlags committedHistoryRequirements_ =
      FrameTextureRequirementFlags::None;
};

} // namespace nuri
