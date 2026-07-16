#include "nuri/gfx/gpu_device.h"

#include "nuri/platform/nvrhi_gpu_device.h"

namespace nuri {

std::unique_ptr<GPUDevice> GPUDevice::create(Window &window,
                                             const GPUDeviceCreateDesc &desc) {
  return NvrhiGPUDevice::create(window, desc);
}

} // namespace nuri
