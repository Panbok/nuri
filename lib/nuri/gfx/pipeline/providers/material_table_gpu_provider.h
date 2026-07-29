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
#include <vector>
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
  void onFrameSubmitted(const RenderFrameContext &frame) noexcept;
  void onFrameAbandoned(const RenderFrameContext &) noexcept;

private:
  static constexpr uint64_t kNoVersionUploaded =
      std::numeric_limits<uint64_t>::max();
  struct TableState {
    std::unique_ptr<DynamicBufferRing> ring;
    std::vector<uint64_t> laneVersions;
  };
  void abandonPrepared() noexcept;
  GPUDevice &gpu_;
  std::array<TableState, 5> tables_{};
};

} // namespace nuri
