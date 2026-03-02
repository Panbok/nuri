#pragma once

#include "nuri/core/result.h"
#include "nuri/core/runtime_config.h"
#include "nuri/gfx/gpu_types.h"

#include <filesystem>
#include <span>
#include <vector>

namespace nuri {
class GPUDevice;
}

namespace nuri::bakery::detail {

struct BrdfBakePlan {
  bool shouldBake = false;
  std::filesystem::path shaderPath;
  std::filesystem::path outputPath;
};

struct BrdfBakeGpuState {
  BufferHandle outputBuffer{};
  ComputePipelineHandle computePipeline{};
  ShaderHandle computeShader{};
};

[[nodiscard]] Result<BrdfBakePlan, std::string>
planBrdfLutBake(const RuntimeConfig &config, bool forceRebuild);

[[nodiscard]] Result<bool, std::string>
setupBrdfLutBake(GPUDevice &gpu, const std::filesystem::path &shaderPath,
                 BrdfBakeGpuState &state);

[[nodiscard]] Result<std::vector<std::byte>, std::string>
runBrdfLutBakeStep(GPUDevice &gpu, const BrdfBakeGpuState &state);

void cleanupBrdfLutBake(GPUDevice &gpu, BrdfBakeGpuState &state);

[[nodiscard]] Result<bool, std::string>
writeBrdfLutKtx2(std::span<const std::byte> bytes,
                 const std::filesystem::path &outputPath);

[[nodiscard]] uint32_t brdfTotalSteps();

} // namespace nuri::bakery::detail
