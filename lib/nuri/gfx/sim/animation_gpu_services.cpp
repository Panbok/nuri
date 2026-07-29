#include "nuri/gfx/sim/animation_gpu_services.h"
#include "nuri/gfx/shader.h"
namespace nuri {
namespace {
struct AnimationProgramDesc {
  std::string_view shaderFile;
  std::string_view debugName;
};
constexpr std::array kAnimationPrograms{
    AnimationProgramDesc{"animation_pose_sample.comp", "animation_pose_sample"},
    AnimationProgramDesc{"animation_pose_blend.comp", "animation_pose_blend"},
    AnimationProgramDesc{"animation_pose_world.comp", "animation_pose_world"},
    AnimationProgramDesc{"animation_pose_instance_scatter.comp",
                         "animation_pose_scatter"},
    AnimationProgramDesc{"animation_pose_morph.comp", "animation_pose_morph"},
    AnimationProgramDesc{"animation_pose_skin_palette.comp",
                         "animation_pose_skin_palette"},
    AnimationProgramDesc{"animation_pose_skin.comp", "animation_pose_skin"},
};
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

ComputePipelineHandle
AnimationGpuServices::pipeline(Program program) const noexcept {
  return pipelines_[static_cast<size_t>(program)].get();
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
  std::array<OwnedShaderHandle, kProgramCount> compiledShaders{};
  for (size_t index = 0u; index < kAnimationPrograms.size(); ++index) {
    const std::filesystem::path path =
        shaderRoot_ / std::string(kAnimationPrograms[index].shaderFile);
    auto result = compileShaderFile(gpu_, "animation_pose", path.string(),
                                    ShaderStage::Compute);
    if (result.hasError()) {
      return Result<bool, std::string>::makeError(result.error());
    }
    compiledShaders[index].reset(gpu_, result.value());
  }
  shaders_ = std::move(compiledShaders);
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> AnimationGpuServices::createPipelines() {
  destroyPipelines();
  std::array<OwnedComputePipelineHandle, kProgramCount> pipelines{};
  for (size_t index = 0u; index < kAnimationPrograms.size(); ++index) {
    auto result = createComputePipeline(gpu_, shaders_[index].get(),
                                        kAnimationPrograms[index].debugName,
                                        pipelines[index]);
    if (result.hasError()) {
      return result;
    }
  }
  pipelines_ = std::move(pipelines);
  return Result<bool, std::string>::makeResult(true);
}

void AnimationGpuServices::destroyPipelines() noexcept {
  for (auto handle = pipelines_.rbegin(); handle != pipelines_.rend();
       ++handle) {
    handle->reset();
  }
  initialized_ = false;
}

void AnimationGpuServices::destroyShaders() noexcept {
  for (OwnedShaderHandle &shader : shaders_) {
    shader.reset();
  }
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
