#pragma once

#include "nuri/gfx/gpu_device.h"

#include <utility>

namespace nuri {

template <typename HandleType, void (GPUDevice::*DestroyResource)(HandleType)>
class OwnedGpuResource final {
public:
  OwnedGpuResource() = default;
  OwnedGpuResource(GPUDevice &gpu, HandleType handle) noexcept
      : gpu_(&gpu), handle_(handle) {}
  ~OwnedGpuResource() { reset(); }

  OwnedGpuResource(const OwnedGpuResource &) = delete;
  OwnedGpuResource &operator=(const OwnedGpuResource &) = delete;

  OwnedGpuResource(OwnedGpuResource &&other) noexcept
      : gpu_(std::exchange(other.gpu_, nullptr)),
        handle_(std::exchange(other.handle_, {})) {}

  OwnedGpuResource &operator=(OwnedGpuResource &&other) noexcept {
    if (this != &other) {
      reset();
      gpu_ = std::exchange(other.gpu_, nullptr);
      handle_ = std::exchange(other.handle_, {});
    }
    return *this;
  }

  [[nodiscard]] HandleType get() const noexcept { return handle_; }
  [[nodiscard]] bool valid() const noexcept { return nuri::isValid(handle_); }
  [[nodiscard]] explicit operator bool() const noexcept { return valid(); }

  void reset() noexcept {
    if (gpu_ != nullptr && nuri::isValid(handle_)) {
      (gpu_->*DestroyResource)(handle_);
    }
    gpu_ = nullptr;
    handle_ = {};
  }

  void reset(GPUDevice &gpu, HandleType handle) noexcept {
    reset();
    gpu_ = &gpu;
    handle_ = handle;
  }

  [[nodiscard]] HandleType release() noexcept {
    gpu_ = nullptr;
    return std::exchange(handle_, {});
  }

private:
  GPUDevice *gpu_ = nullptr;
  HandleType handle_{};
};

using OwnedBufferHandle =
    OwnedGpuResource<BufferHandle, &GPUDevice::destroyBuffer>;
using OwnedTextureHandle =
    OwnedGpuResource<TextureHandle, &GPUDevice::destroyTexture>;
using OwnedShaderHandle =
    OwnedGpuResource<ShaderHandle, &GPUDevice::destroyShaderModule>;
using OwnedRenderPipelineHandle =
    OwnedGpuResource<RenderPipelineHandle, &GPUDevice::destroyRenderPipeline>;
using OwnedComputePipelineHandle =
    OwnedGpuResource<ComputePipelineHandle, &GPUDevice::destroyComputePipeline>;
using OwnedMeshletPipelineHandle =
    OwnedGpuResource<MeshletPipelineHandle, &GPUDevice::destroyMeshletPipeline>;

} // namespace nuri
