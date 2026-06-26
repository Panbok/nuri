#include "nuri/gfx/gpu_device.h"

#include "nuri/core/log.h"
#include "nuri/platform/lvk_gpu_device.h"
#include "nuri/platform/nvrhi_gpu_device.h"
#include "nuri/utils/env_utils.h"

#include <algorithm>
#include <cctype>

namespace nuri {

std::unique_ptr<GPUDevice> GPUDevice::create(Window &window,
                                             const GPUDeviceCreateDesc &desc) {
  std::string backend;
  switch (desc.backend) {
  case GPUBackendPreference::Lvk:
    backend = "lvk";
    break;
  case GPUBackendPreference::Nvrhi:
    backend = "nvrhi";
    break;
  case GPUBackendPreference::Default:
    backend = readEnvVar("NURI_GPU_BACKEND").value_or("nvrhi");
    std::transform(
        backend.begin(), backend.end(), backend.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    break;
  }

  if (backend == "lvk") {
    NURI_LOG_INFO("GPUDevice::create: creating LVK backend");
    return LvkGPUDevice::create(window, desc);
  }

  NURI_LOG_INFO("GPUDevice::create: creating NVRHI backend");
  return NvrhiGPUDevice::create(window, desc);
}

} // namespace nuri
