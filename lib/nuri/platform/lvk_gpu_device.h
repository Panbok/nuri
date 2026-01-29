#pragma once

#include "nuri/gfx/gpu_device.h"

namespace nuri {

class Window;

class LvkGPUDevice final : public GPUDevice {
public:
  static std::unique_ptr<LvkGPUDevice> create(Window &window);
  ~LvkGPUDevice() override;

  LvkGPUDevice(const LvkGPUDevice &) = delete;
  LvkGPUDevice &operator=(const LvkGPUDevice &) = delete;
  LvkGPUDevice(LvkGPUDevice &&) = delete;
  LvkGPUDevice &operator=(LvkGPUDevice &&) = delete;

  // Window/Swapchain
  void pollEvents() override;
  bool shouldClose() const override;
  void getFramebufferSize(int32_t &outWidth,
                          int32_t &outHeight) const override;
  void resizeSwapchain(int32_t width, int32_t height) override;
  Format getSwapchainFormat() const override;
  double getTime() const override;

  // Resource creation
  Result<BufferHandle, std::string>
  createBuffer(const BufferDesc &desc,
               std::string_view debugName = {}) override;
  Result<TextureHandle, std::string>
  createTexture(const TextureDesc &desc,
                std::string_view debugName = {}) override;
  Result<ShaderHandle, std::string>
  createShaderModule(const ShaderDesc &desc) override;
  Result<RenderPipelineHandle, std::string>
  createRenderPipeline(const RenderPipelineDesc &desc,
                       std::string_view debugName = {}) override;
  Result<ComputePipelineHandle, std::string>
  createComputePipeline(const ComputePipelineDesc &desc,
                        std::string_view debugName = {}) override;

  // Resource queries
  bool isValid(BufferHandle h) const override;
  bool isValid(TextureHandle h) const override;
  bool isValid(ShaderHandle h) const override;
  bool isValid(RenderPipelineHandle h) const override;
  bool isValid(ComputePipelineHandle h) const override;
  Format getTextureFormat(TextureHandle h) const override;

  // Rendering
  Result<bool, std::string> submitFrame(const RenderFrame &frame) override;

  // Shutdown
  void waitIdle() override;

private:
  LvkGPUDevice();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace nuri
