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
} // namespace

AnimationGpuServices::AnimationGpuServices(GPUDevice &gpu,
                                           std::filesystem::path shaderRoot,
                                           std::pmr::memory_resource *memory)
    : gpu_(gpu), shaderRoot_(std::move(shaderRoot)),
      memory_(memory != nullptr ? memory : std::pmr::get_default_resource()) {}

AnimationGpuServices::~AnimationGpuServices() = default;

ComputePipelineHandle
AnimationGpuServices::pipeline(Program program) const noexcept {
  return programs_.computePipeline(static_cast<size_t>(program));
}

Result<void, std::string> AnimationGpuServices::ensureInitialized() {
  if (initialized_) {
    return Result<void, std::string>::makeResult();
  }
  std::array<ShaderSpec, kProgramCount> shaderSpecs{};
  for (size_t index = 0u; index < kProgramCount; ++index) {
    shaderSpecs[index] = ShaderSpec{
        .debugName = "animation_pose",
        .path = shaderRoot_ / std::string(kAnimationPrograms[index].shaderFile),
        .stage = ShaderStage::Compute,
    };
  }
  auto shaderResult = programs_.compileShaders(gpu_, shaderSpecs);
  if (shaderResult.hasError()) {
    return Result<void, std::string>::makeError(shaderResult.error());
  }
  std::array<ComputePipelineSpec, kProgramCount> pipelineSpecs{};
  for (size_t index = 0u; index < kProgramCount; ++index) {
    pipelineSpecs[index] = ComputePipelineSpec{
        .debugName = kAnimationPrograms[index].debugName,
        .desc = {.computeShader = programs_.shader(index)},
    };
  }
  auto pipelineResult = programs_.replaceComputePipelines(gpu_, pipelineSpecs);
  if (pipelineResult.hasError()) {
    programs_.reset();
    return Result<void, std::string>::makeError(pipelineResult.error());
  }
  initialized_ = true;
  return Result<void, std::string>::makeResult();
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
