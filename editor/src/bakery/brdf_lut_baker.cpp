#include "nuri/editor_pch.h"

#include "nuri/bakery/brdf_lut_baker.h"

#include "nuri/core/log.h"
#include "nuri/gfx/gpu_descriptors.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/gpu_render_types.h"
#include "nuri/gfx/shader.h"

#include <ktx.h>
#include <vulkan/vulkan_core.h>

namespace nuri::bakery::detail {
namespace {

constexpr uint32_t kBrdfLutWidth = 256u;
constexpr uint32_t kBrdfLutHeight = 256u;
constexpr uint32_t kBrdfLutChannels = 4u;
constexpr uint32_t kBrdfLutLocalSize = 16u;
constexpr uint32_t kBrdfLutNumSamples = 1024u;
constexpr size_t kBrdfLutValueCount = static_cast<size_t>(kBrdfLutWidth) *
                                      static_cast<size_t>(kBrdfLutHeight) *
                                      static_cast<size_t>(kBrdfLutChannels);
constexpr size_t kBrdfLutBytes = kBrdfLutValueCount * sizeof(uint16_t);

struct BrdfLutPushConstants {
  uint32_t width = kBrdfLutWidth;
  uint32_t height = kBrdfLutHeight;
  uint64_t outputBufferAddress = 0;
};
static_assert(sizeof(BrdfLutPushConstants) <= 128,
              "BrdfLutPushConstants exceeds Vulkan minimum guarantee");

} // namespace

Result<BrdfBakePlan, std::string> planBrdfLutBake(const RuntimeConfig &config,
                                                  bool forceRebuild) {
  BrdfBakePlan plan{};
  plan.shaderPath = config.roots.shaders / "brdf_lut.comp";
  plan.outputPath = config.roots.textures / "brdf_lut.ktx2";

  if (!std::filesystem::exists(plan.shaderPath)) {
    return Result<BrdfBakePlan, std::string>::makeError(
        "BRDF LUT baker: shader does not exist: '" + plan.shaderPath.string() +
        "'");
  }

  if (forceRebuild) {
    plan.shouldBake = true;
    return Result<BrdfBakePlan, std::string>::makeResult(std::move(plan));
  }

  if (std::filesystem::exists(plan.outputPath)) {
    std::error_code ecShader;
    const auto shaderWriteTime =
        std::filesystem::last_write_time(plan.shaderPath, ecShader);
    std::error_code ecKtx2;
    const auto ktx2WriteTime =
        std::filesystem::last_write_time(plan.outputPath, ecKtx2);
    if (!ecShader && !ecKtx2 && shaderWriteTime <= ktx2WriteTime) {
      plan.shouldBake = false;
      return Result<BrdfBakePlan, std::string>::makeResult(std::move(plan));
    }
  }

  plan.shouldBake = true;
  return Result<BrdfBakePlan, std::string>::makeResult(std::move(plan));
}

Result<bool, std::string>
setupBrdfLutBake(GPUDevice &gpu, const std::filesystem::path &shaderPath,
                 BrdfBakeGpuState &state) {
  cleanupBrdfLutBake(gpu, state);

  auto shader = Shader::create("brdf_lut_baker", gpu);
  if (!shader) {
    return Result<bool, std::string>::makeError(
        "BRDF LUT baker: failed to create shader helper");
  }

  auto compileResult =
      shader->compileFromFile(shaderPath.string(), ShaderStage::Compute);
  if (compileResult.hasError()) {
    cleanupBrdfLutBake(gpu, state);
    return Result<bool, std::string>::makeError(
        "BRDF LUT baker: shader compile failed: " + compileResult.error());
  }
  state.computeShader = compileResult.value();

  const BufferDesc bufferDesc{
      .usage = BufferUsage::Storage,
      .storage = Storage::HostVisible,
      .size = kBrdfLutBytes,
  };
  auto outputBufferResult =
      gpu.createBuffer(bufferDesc, "brdf_lut_output_buffer");
  if (outputBufferResult.hasError()) {
    cleanupBrdfLutBake(gpu, state);
    return Result<bool, std::string>::makeError(
        "BRDF LUT baker: output buffer creation failed: " +
        outputBufferResult.error());
  }
  state.outputBuffer = outputBufferResult.value();

  const std::array<SpecializationEntry, 1> specEntries = {
      SpecializationEntry{
          .constantId = 0u,
          .offset = 0u,
          .size = sizeof(kBrdfLutNumSamples),
      },
  };
  const ComputePipelineDesc pipelineDesc{
      .computeShader = state.computeShader,
      .specInfo =
          {
              .entries = std::span<const SpecializationEntry>(
                  specEntries.data(), specEntries.size()),
              .data = &kBrdfLutNumSamples,
              .dataSize = sizeof(kBrdfLutNumSamples),
          },
  };
  auto pipelineResult =
      gpu.createComputePipeline(pipelineDesc, "brdf_lut_compute");
  if (pipelineResult.hasError()) {
    cleanupBrdfLutBake(gpu, state);
    return Result<bool, std::string>::makeError(
        "BRDF LUT baker: compute pipeline creation failed: " +
        pipelineResult.error());
  }
  state.computePipeline = pipelineResult.value();

  return Result<bool, std::string>::makeResult(true);
}

Result<std::vector<std::byte>, std::string>
runBrdfLutBakeStep(GPUDevice &gpu, const BrdfBakeGpuState &state) {
  if (!nuri::isValid(state.computePipeline) ||
      !nuri::isValid(state.outputBuffer)) {
    return Result<std::vector<std::byte>, std::string>::makeError(
        "BRDF LUT baker: GPU state is not initialized");
  }

  const uint64_t outputAddress = gpu.getBufferDeviceAddress(state.outputBuffer);
  if (outputAddress == 0) {
    return Result<std::vector<std::byte>, std::string>::makeError(
        "BRDF LUT baker: output buffer address is invalid");
  }

  const BrdfLutPushConstants pushConstants{
      .outputBufferAddress = outputAddress,
  };
  const ComputeDispatchItem dispatch{
      .pipeline = state.computePipeline,
      .dispatch =
          {
              .x = (kBrdfLutWidth + (kBrdfLutLocalSize - 1u)) /
                   kBrdfLutLocalSize,
              .y = (kBrdfLutHeight + (kBrdfLutLocalSize - 1u)) /
                   kBrdfLutLocalSize,
              .z = 1u,
          },
      .pushConstants = std::span<const std::byte>(
          reinterpret_cast<const std::byte *>(&pushConstants),
          sizeof(pushConstants)),
      .debugLabel = "BRDF LUT Bake",
      .debugColor = 0xff2288ccu,
  };
  auto dispatchResult = gpu.submitComputeDispatches(
      std::span<const ComputeDispatchItem>(&dispatch, 1));
  if (dispatchResult.hasError()) {
    return Result<std::vector<std::byte>, std::string>::makeError(
        "BRDF LUT baker: compute dispatch failed: " + dispatchResult.error());
  }

  std::vector<std::byte> lutBytes(kBrdfLutBytes);
  auto readResult =
      gpu.readBuffer(state.outputBuffer, 0u,
                     std::span<std::byte>(lutBytes.data(), lutBytes.size()));
  if (readResult.hasError()) {
    return Result<std::vector<std::byte>, std::string>::makeError(
        "BRDF LUT baker: failed to read output buffer: " + readResult.error());
  }

  return Result<std::vector<std::byte>, std::string>::makeResult(
      std::move(lutBytes));
}

void cleanupBrdfLutBake(GPUDevice &gpu, BrdfBakeGpuState &state) {
  if (nuri::isValid(state.computePipeline)) {
    gpu.destroyComputePipeline(state.computePipeline);
    state.computePipeline = {};
  }
  if (nuri::isValid(state.computeShader)) {
    gpu.destroyShaderModule(state.computeShader);
    state.computeShader = {};
  }
  if (nuri::isValid(state.outputBuffer)) {
    gpu.destroyBuffer(state.outputBuffer);
    state.outputBuffer = {};
  }
}

Result<bool, std::string>
writeBrdfLutKtx2(std::span<const std::byte> bytes,
                 const std::filesystem::path &outputPath) {
  if (bytes.size() != kBrdfLutBytes) {
    return Result<bool, std::string>::makeError(
        "BRDF LUT write: invalid data size");
  }

  const std::filesystem::path parent = outputPath.parent_path();
  if (!parent.empty()) {
    std::error_code ec;
    std::filesystem::create_directories(parent, ec);
    if (ec) {
      return Result<bool, std::string>::makeError(
          "BRDF LUT write: failed to create output directory '" +
          parent.string() + "': " + ec.message());
    }
  }

  const ktxTextureCreateInfo createInfo{
      .glInternalformat = 0u,
      .vkFormat = VK_FORMAT_R16G16B16A16_SFLOAT,
      .baseWidth = kBrdfLutWidth,
      .baseHeight = kBrdfLutHeight,
      .baseDepth = 1u,
      .numDimensions = 2u,
      .numLevels = 1u,
      .numLayers = 1u,
      .numFaces = 1u,
      .isArray = KTX_FALSE,
      .generateMipmaps = KTX_FALSE,
  };

  ktxTexture2 *textureKtx2 = nullptr;
  const auto createError = ktxTexture2_Create(
      &createInfo, KTX_TEXTURE_CREATE_ALLOC_STORAGE, &textureKtx2);
  if (createError != KTX_SUCCESS || textureKtx2 == nullptr) {
    if (textureKtx2 != nullptr) {
      ktxTexture_Destroy(ktxTexture(textureKtx2));
      textureKtx2 = nullptr;
    }
    return Result<bool, std::string>::makeError(
        "BRDF LUT write: ktxTexture2_Create failed with code " +
        std::to_string(static_cast<int>(createError)));
  }

  ktxTexture *texture = ktxTexture(textureKtx2);
  std::memcpy(ktxTexture_GetData(texture), bytes.data(), bytes.size());

  const std::string outputPathStr = outputPath.string();
  const auto writeError =
      ktxTexture_WriteToNamedFile(texture, outputPathStr.c_str());
  ktxTexture_Destroy(texture);
  if (writeError != KTX_SUCCESS) {
    return Result<bool, std::string>::makeError(
        "BRDF LUT write: ktxTexture_WriteToNamedFile failed with code " +
        std::to_string(static_cast<int>(writeError)));
  }

  return Result<bool, std::string>::makeResult(true);
}

uint32_t brdfTotalSteps() { return 1u; }

} // namespace nuri::bakery::detail
