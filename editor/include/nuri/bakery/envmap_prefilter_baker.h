#pragma once

#include "nuri/core/result.h"
#include "nuri/core/runtime_config.h"
#include "nuri/gfx/gpu_types.h"
#include "nuri/resources/gpu/texture.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace nuri {
class GPUDevice;
}

namespace nuri::bakery::detail {

enum class EnvDistribution : uint32_t {
  Lambertian = 0u,
  GGX = 1u,
  Charlie = 2u,
};

constexpr std::size_t kEnvDistributionCount =
    static_cast<std::size_t>(EnvDistribution::Charlie) + 1u;

struct EnvOutputPaths {
  std::filesystem::path irradiance;
  std::filesystem::path ggx;
  std::filesystem::path charlie;
};

struct EnvBakePlan {
  bool shouldBake = false;
  std::filesystem::path shaderPath;
  std::filesystem::path hdrPath;
  EnvOutputPaths outputs{};
  uint32_t irradianceFaceSize = 0;
  uint32_t specularFaceSize = 0;
  std::array<uint32_t, kEnvDistributionCount> mipCounts{};
};

struct EnvDistributionRuntime {
  EnvDistribution distribution = EnvDistribution::Lambertian;
  uint32_t baseFaceSize = 0;
  uint32_t sampleCount = 0;
  uint32_t mipLevels = 0;
  std::filesystem::path outputPath;
  std::vector<std::vector<std::byte>> levels;
};

struct EnvPreparedCubemapCpuData {
  uint32_t faceSize = 0;
  std::vector<std::byte> floatBytes;
};

enum class EnvSetupPhase : uint8_t {
  StartCpuPrep,
  WaitCpuPrep,
  CreateTexture,
  CompileShader,
  CreatePipeline,
  Finalize,
  Done,
};

enum class EnvSetupStatus : uint8_t { InProgress, Ready, Skipped };

struct EnvSetupProgress {
  EnvSetupStatus status = EnvSetupStatus::InProgress;
  std::string summary;
};

struct EnvBakeGpuState {
  std::unique_ptr<Texture> sourceCubemap{};
  ShaderHandle computeShader{};
  ComputePipelineHandle computePipeline{};
  BufferHandle tileOutputBuffer{};
  uint64_t tileOutputBufferAddress = 0u;
  size_t tileOutputBufferBytes = 0u;
  std::vector<std::byte> tileScratchBytes{};
  uint32_t envMapTexId = 0;
  uint32_t envMapSamplerId = 0;
  uint32_t sourceCubemapWidth = 0u;
  uint32_t sourceCubemapHeight = 0u;
  std::optional<EnvPreparedCubemapCpuData> preparedSourceCpu{};
  std::shared_future<Result<EnvPreparedCubemapCpuData, std::string>>
      sourcePrepFuture{};
  bool sourcePrepInFlight = false;
  EnvSetupPhase setupPhase = EnvSetupPhase::StartCpuPrep;
  std::array<EnvDistributionRuntime, kEnvDistributionCount> runs{};
  uint32_t activeDistributionIndex = 0;
  uint32_t activeMipLevel = 0;
  uint32_t activeFace = 0;
  uint32_t activeTileIndex = 0;
  uint32_t bakeTileSize = 0;
  uint32_t completedSteps = 0;
  uint32_t totalSteps = 0;
  bool initialized = false;
};

struct EnvWriteDistributionPayload {
  EnvDistribution distribution = EnvDistribution::Lambertian;
  uint32_t baseFaceSize = 0;
  std::filesystem::path outputPath;
  std::vector<std::vector<std::byte>> levelData;
};

struct EnvWritePayload {
  std::array<EnvWriteDistributionPayload, kEnvDistributionCount> distributions{};
};

struct EnvGpuStepProgress {
  uint32_t completedSteps = 0;
  uint32_t totalSteps = 0;
  bool finished = false;
};

[[nodiscard]] Result<EnvBakePlan, std::string>
planEnvmapPrefilterBake(const RuntimeConfig &config,
                        const std::filesystem::path &environmentHdrPath,
                        bool forceRebuild);

[[nodiscard]] Result<EnvSetupProgress, std::string>
advanceEnvmapPrefilterSetup(GPUDevice &gpu, const EnvBakePlan &plan,
                            EnvBakeGpuState &state);

[[nodiscard]] Result<EnvGpuStepProgress, std::string>
runEnvmapPrefilterBakeOneStep(GPUDevice &gpu, EnvBakeGpuState &state);

[[nodiscard]] EnvWritePayload collectEnvWritePayload(EnvBakeGpuState &state);

void cleanupEnvmapPrefilterBake(GPUDevice &gpu, EnvBakeGpuState &state);

[[nodiscard]] Result<bool, std::string>
writeEnvmapPrefilterOutputs(const EnvWritePayload &payload);

[[nodiscard]] std::string_view envDistributionName(EnvDistribution distribution);

} // namespace nuri::bakery::detail
