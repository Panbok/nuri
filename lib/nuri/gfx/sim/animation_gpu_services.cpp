#include "nuri/pch.h"

#include "nuri/gfx/sim/animation_gpu_services.h"

namespace nuri {
namespace {

constexpr std::string_view kSampleShaderName = "animation_pose_sample.comp";
constexpr std::string_view kBlendShaderName = "animation_pose_blend.comp";
constexpr std::string_view kWorldShaderName = "animation_pose_world.comp";
constexpr std::string_view kScatterShaderName =
    "animation_pose_instance_scatter.comp";
constexpr std::string_view kMorphShaderName = "animation_pose_morph.comp";
constexpr std::string_view kSkinPaletteShaderName =
    "animation_pose_skin_palette.comp";
constexpr std::string_view kSkinShaderName = "animation_pose_skin.comp";

Result<bool, std::string>
createComputePipeline(GPUDevice &gpu, ShaderHandle shader,
                      std::string_view debugName,
                      OwnedComputePipelineHandle &outPipeline) {
  auto result = gpu.createComputePipeline(
      ComputePipelineDesc{.computeShader = shader}, debugName);
  if (result.hasError()) {
    return Result<bool, std::string>::makeError(result.error());
  }
  outPipeline.reset(gpu, result.value());
  return Result<bool, std::string>::makeResult(true);
}

} // namespace

AnimationGpuServices::AnimationGpuServices(GPUDevice &gpu,
                                           std::filesystem::path shaderRoot,
                                           std::pmr::memory_resource *memory)
    : gpu_(gpu), shaderRoot_(std::move(shaderRoot)),
      memory_(memory != nullptr ? memory : std::pmr::get_default_resource()) {}

AnimationGpuServices::~AnimationGpuServices() {
  destroyPipelines();
  destroyShaders();
}

Result<void, std::string> AnimationGpuServices::ensureInitialized() {
  if (initialized_) {
    return Result<void, std::string>::makeResult();
  }
  auto shaderResult = createShaders();
  if (shaderResult.hasError()) {
    return Result<void, std::string>::makeError(shaderResult.error());
  }
  auto pipelineResult = createPipelines();
  if (pipelineResult.hasError()) {
    destroyShaders();
    return Result<void, std::string>::makeError(pipelineResult.error());
  }
  initialized_ = true;
  return Result<void, std::string>::makeResult();
}

Result<bool, std::string> AnimationGpuServices::createShaders() {
  destroyShaders();
  std::unique_ptr<Shader> shader = Shader::create("animation_pose", gpu_);
  if (!shader) {
    return Result<bool, std::string>::makeError(
        "AnimationGpuServices::createShaders: failed to create shader wrapper");
  }

  const std::array shaderNames = {
      kSampleShaderName,  kBlendShaderName, kWorldShaderName,
      kScatterShaderName, kMorphShaderName, kSkinPaletteShaderName,
      kSkinShaderName,
  };
  std::array<OwnedShaderHandle, 7u> compiledShaders{};
  for (size_t shaderIndex = 0u; shaderIndex < shaderNames.size();
       ++shaderIndex) {
    const std::string_view fileName = shaderNames[shaderIndex];
    const std::filesystem::path path = shaderRoot_ / std::string(fileName);
    auto result = shader->compileFromFile(path.string(), ShaderStage::Compute);
    if (result.hasError()) {
      return Result<bool, std::string>::makeError(result.error());
    }
    compiledShaders[shaderIndex].reset(gpu_, result.value());
  }

  shader_ = std::move(shader);
  sampleShaderHandle_ = std::move(compiledShaders[0u]);
  blendShaderHandle_ = std::move(compiledShaders[1u]);
  worldShaderHandle_ = std::move(compiledShaders[2u]);
  scatterShaderHandle_ = std::move(compiledShaders[3u]);
  morphShaderHandle_ = std::move(compiledShaders[4u]);
  skinPaletteShaderHandle_ = std::move(compiledShaders[5u]);
  skinShaderHandle_ = std::move(compiledShaders[6u]);
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> AnimationGpuServices::createPipelines() {
  destroyPipelines();
  std::array<OwnedComputePipelineHandle, 7u> pipelines{};

  const std::array pipelineSpecs = {
      std::pair{sampleShaderHandle_.get(),
                std::string_view{"animation_pose_sample"}},
      std::pair{blendShaderHandle_.get(),
                std::string_view{"animation_pose_blend"}},
      std::pair{worldShaderHandle_.get(),
                std::string_view{"animation_pose_world"}},
      std::pair{scatterShaderHandle_.get(),
                std::string_view{"animation_pose_scatter"}},
      std::pair{morphShaderHandle_.get(),
                std::string_view{"animation_pose_morph"}},
      std::pair{skinPaletteShaderHandle_.get(),
                std::string_view{"animation_pose_skin_palette"}},
      std::pair{skinShaderHandle_.get(),
                std::string_view{"animation_pose_skin"}},
  };
  for (size_t pipelineIndex = 0u; pipelineIndex < pipelineSpecs.size();
       ++pipelineIndex) {
    const auto &[shader, debugName] = pipelineSpecs[pipelineIndex];
    auto result = createComputePipeline(gpu_, shader, debugName,
                                        pipelines[pipelineIndex]);
    if (result.hasError()) {
      return result;
    }
  }

  samplePipelineHandle_ = std::move(pipelines[0u]);
  blendPipelineHandle_ = std::move(pipelines[1u]);
  worldPipelineHandle_ = std::move(pipelines[2u]);
  scatterPipelineHandle_ = std::move(pipelines[3u]);
  morphPipelineHandle_ = std::move(pipelines[4u]);
  skinPalettePipelineHandle_ = std::move(pipelines[5u]);
  skinPipelineHandle_ = std::move(pipelines[6u]);
  return Result<bool, std::string>::makeResult(true);
}

void AnimationGpuServices::destroyPipelines() noexcept {
  skinPipelineHandle_.reset();
  skinPalettePipelineHandle_.reset();
  morphPipelineHandle_.reset();
  scatterPipelineHandle_.reset();
  worldPipelineHandle_.reset();
  blendPipelineHandle_.reset();
  samplePipelineHandle_.reset();
  initialized_ = false;
}

void AnimationGpuServices::destroyShaders() noexcept {
  sampleShaderHandle_.reset();
  blendShaderHandle_.reset();
  worldShaderHandle_.reset();
  scatterShaderHandle_.reset();
  morphShaderHandle_.reset();
  skinPaletteShaderHandle_.reset();
  skinShaderHandle_.reset();
  shader_.reset();
}

Result<std::unique_ptr<Buffer>, std::string>
AnimationGpuServices::createStorageBuffer(size_t sizeBytes,
                                          std::string_view debugName) {
  return Buffer::create(gpu_,
                        BufferDesc{.usage = BufferUsage::Storage,
                                   .storage = Storage::Device,
                                   .size = std::max(sizeBytes, size_t{1u})},
                        debugName);
}

Result<std::unique_ptr<Buffer>, std::string>
AnimationGpuServices::createStorageVertexBuffer(size_t sizeBytes,
                                                std::string_view debugName) {
  return Buffer::create(
      gpu_,
      BufferDesc{.usage = BufferUsage::Storage | BufferUsage::Vertex,
                 .storage = Storage::Device,
                 .size = std::max(sizeBytes, size_t{1u})},
      debugName);
}

} // namespace nuri
