#pragma once

#include "nuri/defines.h"
#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/pipeline/frame_data_provider.h"

#include <array>
#include <cstdint>
#include <memory_resource>
#include <string>
#include <string_view>
#include <vector>

namespace nuri {

class NURI_API FrameCompositionProvider final : public FrameDataProvider {
public:
  explicit FrameCompositionProvider(
      GPUDevice &gpu,
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  ~FrameCompositionProvider() override;

  FrameCompositionProvider(const FrameCompositionProvider &) = delete;
  FrameCompositionProvider &
  operator=(const FrameCompositionProvider &) = delete;
  FrameCompositionProvider(FrameCompositionProvider &&) = delete;
  FrameCompositionProvider &operator=(FrameCompositionProvider &&) = delete;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "FrameCompositionProvider";
  }
  [[nodiscard]] Result<bool, std::string>
  prepare(FrameBuildContext &ctx) override;

private:
  using TextureRing = std::pmr::vector<TextureHandle>;

  Result<bool, std::string>
  ensureTextures(FrameTextureRequirementFlags requirements);
  Result<bool, std::string>
  recreateFullResTextureRing(TextureRing &textures, Format format,
                             TextureUsage usage,
                             std::string_view debugNameBase);
  Result<bool, std::string>
  recreateMipTextureRing(TextureRing &textures, uint32_t mipLevel,
                         std::string_view debugNameBase);
  Result<bool, std::string> recreateHistoryTextures();
  Result<bool, std::string> recreateMotionVectorTextures();
  Result<bool, std::string> recreateReactiveMaskTextures();
  void invalidateAllocationState() noexcept;
  void destroyTextures(TextureRing &textures);
  void destroyHistoryTextures();
  void destroyMotionVectorTextures();
  void destroyReactiveMaskTextures();
  [[nodiscard]] TextureHandle
  currentRingTexture(const TextureRing &textures,
                     uint64_t frameIndex) const noexcept;
  [[nodiscard]] TextureHandle
  previousRingTexture(const TextureRing &textures,
                      uint64_t frameIndex) const noexcept;

  GPUDevice &gpu_;
  std::pmr::memory_resource *memory_ = std::pmr::get_default_resource();
  std::array<TextureRing, kFrameCompositionSceneColorMipCount>
      sceneColorMipTextures_;
  TextureRing frameColorTextures_;
  TextureRing sceneDepthTextures_;
  TextureRing motionVectorTextures_;
  TextureRing reactiveMaskTextures_;
  TextureRing historyColorTextures_;
  FrameTextureRequirementFlags allocatedRequirements_ =
      FrameTextureRequirementFlags::None;
  // motionVectorAllocationCount_ and motionVectorReallocationCount_ track
  // motion-vector texture ring churn for profiling/debugging TAA reallocation
  // frequency; other frame textures are long-lived baseline resources here.
  uint32_t motionVectorAllocationCount_ = 0u;
  uint32_t motionVectorReallocationCount_ = 0u;
  uint32_t reactiveMaskAllocationCount_ = 0u;
  uint32_t reactiveMaskReallocationCount_ = 0u;
  uint32_t textureRingCount_ = 0u;
  uint32_t framebufferWidth_ = 0u;
  uint32_t framebufferHeight_ = 0u;
};

} // namespace nuri
