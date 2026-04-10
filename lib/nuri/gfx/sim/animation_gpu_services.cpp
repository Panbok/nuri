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
createComputePipeline(Pipeline &pipeline, ShaderHandle shader,
                      std::string_view debugName,
                      ComputePipelineHandle &outHandle) {
  auto result = pipeline.createComputePipeline(
      ComputePipelineDesc{.computeShader = shader}, debugName);
  if (result.hasError()) {
    return Result<bool, std::string>::makeError(result.error());
  }
  outHandle = result.value();
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
  shader_ = Shader::create("animation_pose", gpu_);
  if (!shader_) {
    return Result<bool, std::string>::makeError(
        "AnimationGpuServices::createShaders: failed to create shader wrapper");
  }

  const std::array specs = {
      std::pair{kSampleShaderName, &sampleShaderHandle_},
      std::pair{kBlendShaderName, &blendShaderHandle_},
      std::pair{kWorldShaderName, &worldShaderHandle_},
      std::pair{kScatterShaderName, &scatterShaderHandle_},
      std::pair{kMorphShaderName, &morphShaderHandle_},
      std::pair{kSkinPaletteShaderName, &skinPaletteShaderHandle_},
      std::pair{kSkinShaderName, &skinShaderHandle_},
  };
  for (const auto &[fileName, outHandle] : specs) {
    const std::filesystem::path path = shaderRoot_ / std::string(fileName);
    auto result = shader_->compileFromFile(path.string(), ShaderStage::Compute);
    if (result.hasError()) {
      return Result<bool, std::string>::makeError(result.error());
    }
    *outHandle = result.value();
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> AnimationGpuServices::createPipelines() {
  destroyPipelines();
  samplePipeline_ = Pipeline::create(gpu_);
  blendPipeline_ = Pipeline::create(gpu_);
  worldPipeline_ = Pipeline::create(gpu_);
  scatterPipeline_ = Pipeline::create(gpu_);
  morphPipeline_ = Pipeline::create(gpu_);
  skinPalettePipeline_ = Pipeline::create(gpu_);
  skinPipeline_ = Pipeline::create(gpu_);
  if (!samplePipeline_ || !blendPipeline_ || !worldPipeline_ ||
      !scatterPipeline_ || !morphPipeline_ || !skinPalettePipeline_ ||
      !skinPipeline_) {
    return Result<bool, std::string>::makeError(
        "AnimationGpuServices::createPipelines: failed to allocate pipeline "
        "wrappers");
  }

  auto sampleResult =
      createComputePipeline(*samplePipeline_, sampleShaderHandle_,
                            "animation_pose_sample", samplePipelineHandle_);
  if (sampleResult.hasError()) {
    return sampleResult;
  }
  auto blendResult =
      createComputePipeline(*blendPipeline_, blendShaderHandle_,
                            "animation_pose_blend", blendPipelineHandle_);
  if (blendResult.hasError()) {
    return blendResult;
  }
  auto worldResult =
      createComputePipeline(*worldPipeline_, worldShaderHandle_,
                            "animation_pose_world", worldPipelineHandle_);
  if (worldResult.hasError()) {
    return worldResult;
  }
  auto scatterResult =
      createComputePipeline(*scatterPipeline_, scatterShaderHandle_,
                            "animation_pose_scatter", scatterPipelineHandle_);
  if (scatterResult.hasError()) {
    return scatterResult;
  }
  auto morphResult =
      createComputePipeline(*morphPipeline_, morphShaderHandle_,
                            "animation_pose_morph", morphPipelineHandle_);
  if (morphResult.hasError()) {
    return morphResult;
  }
  auto skinPaletteResult = createComputePipeline(
      *skinPalettePipeline_, skinPaletteShaderHandle_,
      "animation_pose_skin_palette", skinPalettePipelineHandle_);
  if (skinPaletteResult.hasError()) {
    return skinPaletteResult;
  }
  return createComputePipeline(*skinPipeline_, skinShaderHandle_,
                               "animation_pose_skin", skinPipelineHandle_);
}

void AnimationGpuServices::destroyPipelines() noexcept {
  skinPipeline_.reset();
  skinPalettePipeline_.reset();
  morphPipeline_.reset();
  scatterPipeline_.reset();
  worldPipeline_.reset();
  blendPipeline_.reset();
  samplePipeline_.reset();
  samplePipelineHandle_ = {};
  blendPipelineHandle_ = {};
  worldPipelineHandle_ = {};
  scatterPipelineHandle_ = {};
  morphPipelineHandle_ = {};
  skinPalettePipelineHandle_ = {};
  skinPipelineHandle_ = {};
  initialized_ = false;
}

void AnimationGpuServices::destroyShaders() noexcept {
  if (nuri::isValid(sampleShaderHandle_)) {
    gpu_.destroyShaderModule(sampleShaderHandle_);
  }
  if (nuri::isValid(blendShaderHandle_)) {
    gpu_.destroyShaderModule(blendShaderHandle_);
  }
  if (nuri::isValid(worldShaderHandle_)) {
    gpu_.destroyShaderModule(worldShaderHandle_);
  }
  if (nuri::isValid(scatterShaderHandle_)) {
    gpu_.destroyShaderModule(scatterShaderHandle_);
  }
  if (nuri::isValid(morphShaderHandle_)) {
    gpu_.destroyShaderModule(morphShaderHandle_);
  }
  if (nuri::isValid(skinPaletteShaderHandle_)) {
    gpu_.destroyShaderModule(skinPaletteShaderHandle_);
  }
  if (nuri::isValid(skinShaderHandle_)) {
    gpu_.destroyShaderModule(skinShaderHandle_);
  }
  sampleShaderHandle_ = {};
  blendShaderHandle_ = {};
  worldShaderHandle_ = {};
  scatterShaderHandle_ = {};
  morphShaderHandle_ = {};
  skinPaletteShaderHandle_ = {};
  skinShaderHandle_ = {};
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
