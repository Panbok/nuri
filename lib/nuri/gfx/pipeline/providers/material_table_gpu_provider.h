#pragma once
#include "nuri/defines.h"
#include "nuri/gfx/dynamic_buffer.h"
#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/pipeline/render_pipeline.h"
#include "nuri/resources/gpu/buffer.h"
#include <array>
#include <limits>
#include <memory>
#include <string>
namespace nuri {

class NURI_API MaterialTableGpuProvider final {
public:
  explicit MaterialTableGpuProvider(GPUDevice &gpu);
  ~MaterialTableGpuProvider();
  MaterialTableGpuProvider(const MaterialTableGpuProvider &) = delete;
  MaterialTableGpuProvider &
  operator=(const MaterialTableGpuProvider &) = delete;
  MaterialTableGpuProvider(MaterialTableGpuProvider &&) = delete;
  MaterialTableGpuProvider &operator=(MaterialTableGpuProvider &&) = delete;
  [[nodiscard]] Result<bool, std::string> prepare(FrameBuildContext &ctx);

private:
  static constexpr uint64_t kNoVersionUploaded =
      std::numeric_limits<uint64_t>::max();
  void destroyBuffers();
  GPUDevice &gpu_;
  std::array<DynamicBufferSlot, 5> buffers_{};
  uint64_t uploadedVersion_ = kNoVersionUploaded;
};

} // namespace nuri
