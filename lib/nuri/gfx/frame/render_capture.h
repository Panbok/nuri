#pragma once
#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/gfx/gpu_device.h"
namespace nuri {

inline void publishRequestedCapture(
    RenderFrameContext &frame, GPUDevice &gpu, std::string_view name,
    TextureHandle texture, RenderCaptureValueKind kind,
    RenderCaptureLifetimeClass lifetime, std::string_view colorSpace,
    std::string_view compareProfile, std::string_view producerPassLabel,
    std::string_view debugLabel = {}) {
  if (!isRenderCaptureRequested(frame, name) || !nuri::isValid(texture)) {
    return;
  }
  frame.captureRegistry.publish(RenderCapturePoint{
      .name = name,
      .version = 1u,
      .texture = texture,
      .format = gpu.getTextureFormat(texture),
      .dimensions = gpu.getTextureDimensions(texture),
      .frameIndex = frame.frameIndex,
      .mip = 0u,
      .layer = 0u,
      .kind = kind,
      .lifetime = lifetime,
      .colorSpace = colorSpace,
      .defaultCompareProfile = compareProfile,
      .producerPassLabel = producerPassLabel,
      .debugLabel = debugLabel.empty() ? name : debugLabel,
  });
}

} // namespace nuri
