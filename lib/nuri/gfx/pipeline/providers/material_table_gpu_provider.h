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
  struct ManagedBuffer {
    std::unique_ptr<Buffer> buffer{};
    size_t capacityBytes = 0u;
  };

  static constexpr uint64_t kNoVersionUploaded =
      std::numeric_limits<uint64_t>::max();

  Result<bool, std::string> ensureBufferCapacity(ManagedBuffer &managedBuffer,
                                                 size_t requiredBytes,
                                                 std::string_view debugName);
  void destroyBuffers();

  GPUDevice &gpu_;
  ManagedBuffer headerBuffer_;
  ManagedBuffer clearcoatBuffer_;
  ManagedBuffer sheenBuffer_;
  ManagedBuffer transmissionBuffer_;
  ManagedBuffer specularBuffer_;
  uint64_t uploadedVersion_ = kNoVersionUploaded;
};

} // namespace nuri
