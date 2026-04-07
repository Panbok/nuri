#pragma once

#include "nuri/defines.h"
#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/pipeline/frame_data_provider.h"
#include "nuri/resources/gpu/buffer.h"

#include <limits>
#include <memory>
#include <string>
#include <string_view>

namespace nuri {

class NURI_API MaterialTableGpuProvider final : public FrameDataProvider {
public:
  explicit MaterialTableGpuProvider(GPUDevice &gpu);
  ~MaterialTableGpuProvider() override;

  MaterialTableGpuProvider(const MaterialTableGpuProvider &) = delete;
  MaterialTableGpuProvider &
  operator=(const MaterialTableGpuProvider &) = delete;
  MaterialTableGpuProvider(MaterialTableGpuProvider &&) = delete;
  MaterialTableGpuProvider &operator=(MaterialTableGpuProvider &&) = delete;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "MaterialTableGpuProvider";
  }
  [[nodiscard]] Result<bool, std::string>
  prepare(FrameBuildContext &ctx) override;

private:
  Result<bool, std::string>
  ensureBufferCapacity(std::unique_ptr<Buffer> &buffer, size_t &capacityBytes,
                       size_t requiredBytes, std::string_view debugName);
  void destroyBuffers();

  GPUDevice &gpu_;
  std::unique_ptr<Buffer> headerBuffer_;
  std::unique_ptr<Buffer> clearcoatBuffer_;
  std::unique_ptr<Buffer> sheenBuffer_;
  std::unique_ptr<Buffer> transmissionBuffer_;
  std::unique_ptr<Buffer> specularBuffer_;
  size_t headerCapacityBytes_ = 0u;
  size_t clearcoatCapacityBytes_ = 0u;
  size_t sheenCapacityBytes_ = 0u;
  size_t transmissionCapacityBytes_ = 0u;
  size_t specularCapacityBytes_ = 0u;
  uint64_t uploadedVersion_ = std::numeric_limits<uint64_t>::max();
};

} // namespace nuri
