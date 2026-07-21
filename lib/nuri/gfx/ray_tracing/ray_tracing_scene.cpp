#include "nuri/gfx/ray_tracing/ray_tracing_scene.h"

#include "nuri/core/containers/hash_map.h"
#include "nuri/core/profiling.h"
#include "nuri/gfx/renderers/scene_draw_database.h"
#include "nuri/gfx/sim/animation_scene_frame_data.h"
#include "nuri/math/utils.h"
#include "nuri/pch.h"
#include "nuri/resources/gpu/resource_manager.h"
#include "nuri/scene/render_scene.h"

namespace nuri {
namespace {

constexpr uint32_t kDecodeWorkgroupSize = 64u;
constexpr uint32_t kRtGeometryDoubleSided = 1u << 0u;
constexpr uint32_t kRtGeometryAlphaMasked = 1u << 1u;
constexpr uint32_t kRtGeometryAlphaBlended = 1u << 2u;
constexpr uint32_t kRtGeometryTransmission = 1u << 3u;

template <typename T>
[[nodiscard]] std::span<const std::byte> bytesOf(std::span<T> values) {
  return std::as_bytes(values);
}

template <typename T>
[[nodiscard]] std::span<const std::byte> bytesOf(const T &value) {
  return {reinterpret_cast<const std::byte *>(&value), sizeof(T)};
}

[[nodiscard]] uint32_t divRoundUp(uint32_t value, uint32_t divisor) noexcept {
  return value / divisor + (value % divisor != 0u ? 1u : 0u);
}

[[nodiscard]] uint32_t lod0IndexOffset(const Submesh &submesh) noexcept {
  return submesh.lodCount != 0u && submesh.lods[0].indexCount != 0u
             ? submesh.lods[0].indexOffset
             : submesh.indexOffset;
}

[[nodiscard]] uint32_t lod0IndexCount(const Submesh &submesh) noexcept {
  return submesh.lodCount != 0u && submesh.lods[0].indexCount != 0u
             ? submesh.lods[0].indexCount
             : submesh.indexCount;
}

void appendUnique(std::pmr::vector<BufferHandle> &output, BufferHandle handle) {
  if (!nuri::isValid(handle) ||
      std::ranges::find(output, handle) != output.end()) {
    return;
  }
  output.push_back(handle);
}

void appendUnique(std::pmr::vector<BufferHandle> &output,
                  std::pmr::vector<RenderGraphAccessMode> &modes,
                  BufferHandle handle, RenderGraphAccessMode mode) {
  if (!nuri::isValid(handle)) {
    return;
  }
  const auto found = std::ranges::find(output, handle);
  if (found == output.end()) {
    output.push_back(handle);
    modes.push_back(mode);
    return;
  }
  const size_t index = static_cast<size_t>(found - output.begin());
  modes[index] = modes[index] | mode;
}

} // namespace

RayTracingScene::RayTracingScene(GPUDevice &gpu, RuntimeDDGIShaderConfig config,
                                 std::pmr::memory_resource *memory)
    : gpu_(gpu), config_(std::move(config)),
      memory_(memory != nullptr ? memory : std::pmr::get_default_resource()),
      staticGeometries_(memory_), dynamicGeometries_(memory_),
      instanceRecords_(memory_), geometryRecords_(memory_),
      tlasInstances_(memory_), decodeWork_(memory_), decodeDispatches_(memory_),
      dynamicPushConstants_(memory_), dynamicDispatches_(memory_),
      dependencyBuffers_(memory_), dependencyBufferModes_(memory_),
      blasBuildItems_(memory_), blasBufferUses_(memory_), blasUses_(memory_),
      tlasUses_(memory_), indirectReferences_(memory_) {
  auto result = initialize();
  if (result.hasError()) {
    initializationError_ = result.error();
  }
}

RayTracingScene::~RayTracingScene() { clearSceneResources(); }

Result<bool, std::string> RayTracingScene::initialize() {
  decodeShader_ = Shader::create("rt_decode_positions", gpu_);
  auto shader = decodeShader_->compileFromFile(config_.decodePositions.string(),
                                               ShaderStage::Compute);
  if (shader.hasError()) {
    return Result<bool, std::string>::makeError(shader.error());
  }
  auto pipeline = gpu_.createComputePipeline(
      ComputePipelineDesc{.computeShader = shader.value()},
      "rt_decode_positions");
  if (pipeline.hasError()) {
    return Result<bool, std::string>::makeError(pipeline.error());
  }
  decodePipeline_.reset(gpu_, pipeline.value());
  dynamicVertexShader_ = Shader::create("rt_prepare_dynamic_vertices", gpu_);
  auto dynamicShader = dynamicVertexShader_->compileFromFile(
      config_.prepareDynamicVertices.string(), ShaderStage::Compute);
  if (dynamicShader.hasError()) {
    return Result<bool, std::string>::makeError(dynamicShader.error());
  }
  auto dynamicPipeline = gpu_.createComputePipeline(
      ComputePipelineDesc{.computeShader = dynamicShader.value()},
      "rt_prepare_dynamic_vertices");
  if (dynamicPipeline.hasError()) {
    return Result<bool, std::string>::makeError(dynamicPipeline.error());
  }
  dynamicVertexPipeline_.reset(gpu_, dynamicPipeline.value());
  return Result<bool, std::string>::makeResult(true);
}

void RayTracingScene::clearSceneResources() noexcept {
  staticGeometries_.clear();
  dynamicGeometries_.clear();
  instanceTable_.reset();
  geometryTable_.reset();
  tlas_.reset();
  instanceRecords_.clear();
  geometryRecords_.clear();
  tlasInstances_.clear();
  decodeWork_.clear();
  decodeDispatches_.clear();
  dynamicPushConstants_.clear();
  dynamicDispatches_.clear();
  dependencyBuffers_.clear();
  dependencyBufferModes_.clear();
  blasBuildItems_.clear();
  blasBufferUses_.clear();
  blasUses_.clear();
  tlasUses_.clear();
  indirectReferences_.clear();
  indirectReferenceInputs_ = {};
  indirectReferencesDirty_ = true;
  buildCompletion_ = {};
  ready_ = false;
  failed_ = false;
  topologyBuildScheduled_ = false;
  dynamicUpdateScheduled_ = false;
  transformUpdateScheduled_ = false;
  scheduledFrameIndex_ = UINT64_MAX;
  animationVersion_ = 0u;
  pendingAnimationVersion_ = 0u;
  pendingTransformVersion_ = 0u;
  sceneId_ = 0u;
  topologyVersion_ = 0u;
  transformVersion_ = 0u;
  deformationVersion_ = 0u;
  geometryMutationVersion_ = 0u;
  staticInstanceCount_ = 0u;
  excludedDynamicInstances_ = 0u;
}

void RayTracingScene::pollCompletion() noexcept {
  if (!ready_ && nuri::isValid(buildCompletion_) &&
      gpu_.isSubmissionComplete(buildCompletion_)) {
    buildCompletion_ = {};
    ready_ = true;
  }
}

Result<bool, std::string>
RayTracingScene::rebuildStaticScene(FrameBuildContext &ctx,
                                    const SceneDrawDatabase &database) {
  clearSceneResources();
  const RenderScene &scene = *ctx.frame.scene;
  const std::span<const SceneDrawRecord> draws = database.draws();
  const std::span<const SceneInstanceRecord> instances = database.instances();
  HashMap<uint64_t, uint32_t> geometryIndices;
  geometryIndices.reserve(instances.size());
  excludedDynamicInstances_ = 0u;
  staticInstanceCount_ = 0u;
  const AnimationSceneFrameData *animationData =
      ctx.shared.animationSceneGpuData.has_value()
          ? &*ctx.shared.animationSceneGpuData
          : nullptr;
  const bool animationDataMatchesScene =
      animationData != nullptr && animationData->scene == &scene &&
      animationData->sceneTopologyVersion == scene.topologyVersion() &&
      animationData->renderableCount == instances.size() &&
      animationData->geometryOverridesByRenderable.size() == instances.size();

  for (uint32_t instanceIndex = 0u;
       instanceIndex < static_cast<uint32_t>(instances.size());
       ++instanceIndex) {
    const SceneInstanceRecord &instance = instances[instanceIndex];
    if (instance.renderable == nullptr || instance.model == nullptr) {
      continue;
    }
    if (instance.firstDraw > draws.size() ||
        instance.drawCount > draws.size() - instance.firstDraw) {
      return Result<bool, std::string>::makeError(
          "RayTracingScene: instance draw range is invalid");
    }
    const std::span<const SceneDrawRecord> instanceDraws =
        draws.subspan(instance.firstDraw, instance.drawCount);
    const bool deforming = instance.dynamicCaster;
    const bool animatedTransform =
        animationDataMatchesScene &&
        std::ranges::find(animationData->animatedRenderableIndices,
                          instanceIndex) !=
            animationData->animatedRenderableIndices.end();
    bool dynamic = deforming || animatedTransform;
    if (animationDataMatchesScene &&
        instanceIndex < animationData->geometryOverridesByRenderable.size()) {
      const AnimatedRenderableGeometryOverride &overrideGeometry =
          animationData->geometryOverridesByRenderable[instanceIndex];
      dynamic = dynamic || (overrideGeometry.enabled &&
                            nuri::isValid(overrideGeometry.vertexBuffer));
    }
    if (dynamic) {
      if (!animationDataMatchesScene) {
        ++excludedDynamicInstances_;
        continue;
      }
      const AnimatedRenderableGeometryOverride &overrideGeometry =
          animationData->geometryOverridesByRenderable[instanceIndex];
      const bool hasOverride =
          overrideGeometry.enabled &&
          nuri::isValid(overrideGeometry.vertexBuffer) &&
          overrideGeometry.vertexCount >= instance.model->vertexCount();
      if (deforming && !hasOverride) {
        ++excludedDynamicInstances_;
        continue;
      }
      GeometryAllocationView allocation{};
      if (!gpu_.resolveGeometry(instance.model->geometryHandle(), allocation)) {
        return Result<bool, std::string>::makeError(
            "RayTracingScene: failed to resolve dynamic geometry");
      }
      const uint64_t worldVertexBytes =
          static_cast<uint64_t>(instance.model->vertexCount()) * 32u;
      if (worldVertexBytes == 0u || worldVertexBytes > SIZE_MAX) {
        return Result<bool, std::string>::makeError(
            "RayTracingScene: invalid dynamic vertex buffer size");
      }
      auto worldVertices = gpu_.createBuffer(
          BufferDesc{.usage = BufferUsage::Storage |
                              BufferUsage::AccelerationStructureBuildInput,
                     .storage = Storage::Device,
                     .size = static_cast<size_t>(worldVertexBytes)},
          "rt_dynamic_world_vertices");
      if (worldVertices.hasError()) {
        return Result<bool, std::string>::makeError(worldVertices.error());
      }
      dynamicGeometries_.emplace_back(memory_);
      DynamicGeometryEntry &entry = dynamicGeometries_.back();
      entry.model = instance.model;
      entry.renderableIndex = instanceIndex;
      entry.sourceVertexBuffer =
          hasOverride ? overrideGeometry.vertexBuffer : allocation.vertexBuffer;
      entry.sourceDecodeBuffer =
          hasOverride ? BufferHandle{} : instance.model->vertexDecodeBuffer();
      entry.sourceVertexByteOffset = hasOverride
                                         ? overrideGeometry.vertexByteOffset
                                         : allocation.vertexByteOffset;
      entry.sourceIndexBuffer = allocation.indexBuffer;
      entry.sourceIndexBaseOffset = allocation.indexByteOffset;
      entry.vertexCount = instance.model->vertexCount();
      entry.sourcePackedVertexFormat = static_cast<uint32_t>(
          hasOverride ? PackedVertexFormat::AnimatedFloat32
                      : instance.model->drawVertexFormat());
      entry.indexFormat =
          allocation.indexCount != 0u &&
                  allocation.indexByteSize / allocation.indexCount ==
                      sizeof(uint16_t)
              ? IndexFormat::U16
              : IndexFormat::U32;
      entry.worldVertices.reset(gpu_, worldVertices.value());
      const uint32_t indexStride =
          entry.indexFormat == IndexFormat::U16 ? 2u : 4u;
      const std::span<const Submesh> submeshes = instance.model->submeshes();
      entry.geometries.reserve(submeshes.size());
      for (const Submesh &submesh : submeshes) {
        const uint32_t indexCount = lod0IndexCount(submesh);
        if (indexCount == 0u) {
          continue;
        }
        entry.geometries.push_back(AccelerationStructureTriangleGeometryDesc{
            .vertexBuffer = entry.worldVertices.get(),
            .indexBuffer = entry.sourceIndexBuffer,
            .vertexFormat = Format::RGB32_FLOAT,
            .indexFormat = entry.indexFormat,
            .vertexByteOffset = 0u,
            .indexByteOffset =
                entry.sourceIndexBaseOffset +
                static_cast<uint64_t>(lod0IndexOffset(submesh)) * indexStride,
            .vertexStrideBytes = 32u,
            .vertexCount = entry.vertexCount,
            .indexCount = indexCount,
            .flags = AccelerationStructureGeometryFlags::None,
        });
      }
      if (entry.geometries.empty()) {
        return Result<bool, std::string>::makeError(
            "RayTracingScene: dynamic geometry has no LOD0 triangles");
      }
      auto blas = gpu_.createBottomLevelAccelerationStructure(
          BlasCreateDesc{
              .geometries = entry.geometries,
              .buildFlags = AccelerationStructureBuildFlags::PreferFastBuild |
                            AccelerationStructureBuildFlags::AllowUpdate,
          },
          "rt_dynamic_blas");
      if (blas.hasError()) {
        return Result<bool, std::string>::makeError(blas.error());
      }
      entry.blas.reset(gpu_, blas.value());
      if (instanceRecords_.size() > kRayTracingInstanceCustomIndexLimit) {
        return Result<bool, std::string>::makeError(
            "RayTracingScene: TLAS custom-index limit exceeded");
      }
      const uint32_t firstGeometry =
          static_cast<uint32_t>(geometryRecords_.size());
      uint32_t geometryCount = 0u;
      const uint64_t indexBaseAddress = gpu_.getBufferDeviceAddress(
          entry.sourceIndexBuffer, entry.sourceIndexBaseOffset);
      const uint64_t worldVertexAddress =
          gpu_.getBufferDeviceAddress(entry.worldVertices.get());
      for (const SceneDrawRecord &draw : instanceDraws) {
        if (draw.submesh == nullptr) {
          continue;
        }
        uint32_t flags = 0u;
        flags |= draw.doubleSided ? kRtGeometryDoubleSided : 0u;
        flags |= draw.alphaMasked ? kRtGeometryAlphaMasked : 0u;
        flags |= draw.alphaBlended ? kRtGeometryAlphaBlended : 0u;
        flags |= draw.transmission ? kRtGeometryTransmission : 0u;
        geometryRecords_.push_back(RtGeometryGpuData{
            .indexBufferAddress = indexBaseAddress,
            .vertexBufferAddress = worldVertexAddress,
            .vertexDecodeAddress = 0u,
            .materialIndex = draw.materialIndex,
            .indexOffset = lod0IndexOffset(*draw.submesh),
            .vertexOffset = 0u,
            .indexFormatAndVertexFormat = packRtGeometryFormats(
                entry.indexFormat,
                static_cast<uint32_t>(PackedVertexFormat::AnimatedFloat32)),
            .flags = flags,
        });
        ++geometryCount;
      }
      if (geometryCount != entry.geometries.size()) {
        return Result<bool, std::string>::makeError(
            "RayTracingScene: dynamic instance and BLAS layouts differ");
      }
      const uint32_t rtInstanceIndex =
          static_cast<uint32_t>(instanceRecords_.size());
      instanceRecords_.push_back(RtInstanceGpuData{
          .firstGeometryRecord = firstGeometry,
          .geometryCount = geometryCount,
          .renderableIndex = instanceIndex,
          .flags = 1u,
          .worldFromObject = glm::mat4(1.0f),
          .objectFromWorld = glm::mat4(1.0f),
      });
      tlasInstances_.push_back(AccelerationStructureInstanceDesc{
          .transform = makeAccelerationStructureTransform(glm::mat4(1.0f)),
          .customIndex = rtInstanceIndex,
          .mask = 0xffu,
          .flags = AccelerationStructureInstanceFlags::TriangleCullDisable |
                   AccelerationStructureInstanceFlags::ForceNonOpaque,
          .bottomLevel = entry.blas.get(),
      });
      continue;
    }
    const uint64_t key = handleKey(instance.model->geometryHandle());
    uint32_t geometryIndex = UINT32_MAX;
    if (const auto found = geometryIndices.find(key);
        found != geometryIndices.end()) {
      geometryIndex = found->second;
    } else {
      GeometryAllocationView allocation{};
      if (!gpu_.resolveGeometry(instance.model->geometryHandle(), allocation)) {
        return Result<bool, std::string>::makeError(
            "RayTracingScene: failed to resolve static geometry");
      }
      if (instance.model->drawVertexFormat() !=
          PackedVertexFormat::StaticQuantized20) {
        return Result<bool, std::string>::makeError(
            "RayTracingScene: unsupported static packed vertex format");
      }
      const uint64_t decodedSize =
          static_cast<uint64_t>(instance.model->vertexCount()) * 16u;
      if (decodedSize == 0u || decodedSize > SIZE_MAX) {
        return Result<bool, std::string>::makeError(
            "RayTracingScene: invalid decoded position buffer size");
      }
      auto decoded = gpu_.createBuffer(
          BufferDesc{.usage = BufferUsage::Storage |
                              BufferUsage::AccelerationStructureBuildInput,
                     .storage = Storage::Device,
                     .size = static_cast<size_t>(decodedSize)},
          "rt_decoded_positions");
      if (decoded.hasError()) {
        return Result<bool, std::string>::makeError(decoded.error());
      }
      staticGeometries_.emplace_back(memory_);
      StaticGeometryEntry &entry = staticGeometries_.back();
      entry.geometry = instance.model->geometryHandle();
      entry.model = instance.model;
      entry.sourceVertexBuffer = allocation.vertexBuffer;
      entry.sourceDecodeBuffer = instance.model->vertexDecodeBuffer();
      entry.sourceIndexBuffer = allocation.indexBuffer;
      entry.sourceIndexBaseOffset = allocation.indexByteOffset;
      entry.vertexCount = instance.model->vertexCount();
      entry.indexFormat =
          allocation.indexCount != 0u &&
                  allocation.indexByteSize / allocation.indexCount ==
                      sizeof(uint16_t)
              ? IndexFormat::U16
              : IndexFormat::U32;
      entry.decodedPositions.reset(gpu_, decoded.value());
      entry.decodedAddress = gpu_.getBufferDeviceAddress(decoded.value());
      if (entry.decodedAddress == 0u ||
          !nuri::isValid(entry.sourceDecodeBuffer)) {
        return Result<bool, std::string>::makeError(
            "RayTracingScene: static decode buffers have no device address");
      }
      const uint32_t indexStride =
          entry.indexFormat == IndexFormat::U16 ? 2u : 4u;
      const std::span<const Submesh> submeshes = instance.model->submeshes();
      entry.geometries.reserve(submeshes.size());
      for (const Submesh &submesh : submeshes) {
        const uint32_t indexCount = lod0IndexCount(submesh);
        if (indexCount == 0u) {
          continue;
        }
        entry.geometries.push_back(AccelerationStructureTriangleGeometryDesc{
            .vertexBuffer = entry.decodedPositions.get(),
            .indexBuffer = entry.sourceIndexBuffer,
            .vertexFormat = Format::RGB32_FLOAT,
            .indexFormat = entry.indexFormat,
            .vertexByteOffset = 0u,
            .indexByteOffset =
                entry.sourceIndexBaseOffset +
                static_cast<uint64_t>(lod0IndexOffset(submesh)) * indexStride,
            .vertexStrideBytes = 16u,
            .vertexCount = entry.vertexCount,
            .indexCount = indexCount,
            .flags = AccelerationStructureGeometryFlags::None,
        });
      }
      if (entry.geometries.empty()) {
        return Result<bool, std::string>::makeError(
            "RayTracingScene: static geometry has no LOD0 triangles");
      }
      auto blas = gpu_.createBottomLevelAccelerationStructure(
          BlasCreateDesc{.geometries = entry.geometries,
                         .buildFlags =
                             AccelerationStructureBuildFlags::PreferFastTrace},
          "rt_static_blas");
      if (blas.hasError()) {
        return Result<bool, std::string>::makeError(blas.error());
      }
      entry.blas.reset(gpu_, blas.value());
      geometryIndex = static_cast<uint32_t>(staticGeometries_.size() - 1u);
      geometryIndices.emplace(key, geometryIndex);
    }

    const StaticGeometryEntry &geometry = staticGeometries_[geometryIndex];
    if (instanceRecords_.size() > kRayTracingInstanceCustomIndexLimit) {
      return Result<bool, std::string>::makeError(
          "RayTracingScene: TLAS custom-index limit exceeded");
    }
    const uint32_t firstGeometry =
        static_cast<uint32_t>(geometryRecords_.size());
    uint32_t geometryCount = 0u;
    const uint64_t indexBaseAddress = gpu_.getBufferDeviceAddress(
        geometry.sourceIndexBuffer, geometry.sourceIndexBaseOffset);
    for (const SceneDrawRecord &draw : instanceDraws) {
      if (draw.submesh == nullptr) {
        continue;
      }
      uint32_t flags = 0u;
      flags |= draw.doubleSided ? kRtGeometryDoubleSided : 0u;
      flags |= draw.alphaMasked ? kRtGeometryAlphaMasked : 0u;
      flags |= draw.alphaBlended ? kRtGeometryAlphaBlended : 0u;
      flags |= draw.transmission ? kRtGeometryTransmission : 0u;
      geometryRecords_.push_back(RtGeometryGpuData{
          .indexBufferAddress = indexBaseAddress,
          .vertexBufferAddress = draw.baseVertexBufferAddress,
          .vertexDecodeAddress = draw.baseVertexDecodeBufferAddress,
          .materialIndex = draw.materialIndex,
          .indexOffset = lod0IndexOffset(*draw.submesh),
          .vertexOffset = 0u,
          .indexFormatAndVertexFormat = packRtGeometryFormats(
              draw.indexFormat, draw.basePackedVertexFormat),
          .flags = flags,
      });
      ++geometryCount;
    }
    if (geometryCount != geometry.geometries.size()) {
      return Result<bool, std::string>::makeError(
          "RayTracingScene: instance and BLAS geometry layouts differ");
    }
    const glm::mat4 &world = instance.renderable->modelMatrix;
    const uint32_t rtInstanceIndex =
        static_cast<uint32_t>(instanceRecords_.size());
    instanceRecords_.push_back(RtInstanceGpuData{
        .firstGeometryRecord = firstGeometry,
        .geometryCount = geometryCount,
        .renderableIndex = instanceIndex,
        .worldFromObject = world,
        .objectFromWorld = safeInverseOrIdentity(world),
    });
    tlasInstances_.push_back(AccelerationStructureInstanceDesc{
        .transform = makeAccelerationStructureTransform(world),
        .customIndex = rtInstanceIndex,
        .mask = 0xffu,
        .flags = AccelerationStructureInstanceFlags::TriangleCullDisable |
                 AccelerationStructureInstanceFlags::ForceNonOpaque,
        .bottomLevel = geometry.blas.get(),
    });
    ++staticInstanceCount_;
  }

  if (tlasInstances_.empty()) {
    sceneId_ = scene.id();
    topologyVersion_ = scene.topologyVersion();
    transformVersion_ = scene.transformVersion();
    deformationVersion_ = animationDataMatchesScene
                              ? animationData->version
                              : scene.deformationVersion();
    geometryMutationVersion_ = gpu_.geometryMutationVersion();
    animationVersion_ = animationDataMatchesScene ? animationData->version : 0u;
    return Result<bool, std::string>::makeResult(true);
  }
  auto tlas = gpu_.createTopLevelAccelerationStructure(
      TlasCreateDesc{
          .maxInstanceCount = static_cast<uint32_t>(tlasInstances_.size()),
          .buildFlags = AccelerationStructureBuildFlags::PreferFastTrace |
                        AccelerationStructureBuildFlags::AllowUpdate,
      },
      "rt_scene_tlas");
  if (tlas.hasError()) {
    return Result<bool, std::string>::makeError(tlas.error());
  }
  tlas_.reset(gpu_, tlas.value());
  auto tableResult = uploadTables();
  if (tableResult.hasError()) {
    return tableResult;
  }
  sceneId_ = scene.id();
  topologyVersion_ = scene.topologyVersion();
  transformVersion_ = scene.transformVersion();
  deformationVersion_ = animationDataMatchesScene ? animationData->version
                                                  : scene.deformationVersion();
  animationVersion_ = animationDataMatchesScene ? animationData->version : 0u;
  geometryMutationVersion_ = gpu_.geometryMutationVersion();
  return appendTopologyBuildPasses(ctx);
}

Result<bool, std::string> RayTracingScene::uploadTables() {
  auto createTable =
      [this](OwnedBufferHandle &destination, std::span<const std::byte> bytes,
             std::string_view name) -> Result<bool, std::string> {
    destination.reset();
    auto created = gpu_.createBuffer(
        BufferDesc{.usage = BufferUsage::Storage,
                   .storage = Storage::Device,
                   .size = std::max<size_t>(bytes.size(), 16u),
                   .data = bytes},
        name);
    if (created.hasError()) {
      return Result<bool, std::string>::makeError(created.error());
    }
    destination.reset(gpu_, created.value());
    return Result<bool, std::string>::makeResult(true);
  };
  auto instanceResult =
      createTable(instanceTable_, bytesOf(std::span(instanceRecords_)),
                  "rt_instance_table");
  if (instanceResult.hasError()) {
    return instanceResult;
  }
  return createTable(geometryTable_, bytesOf(std::span(geometryRecords_)),
                     "rt_geometry_table");
}

Result<bool, std::string>
RayTracingScene::appendTopologyBuildPasses(FrameBuildContext &ctx) {
  decodeWork_.clear();
  size_t decodeCount = 0u;
  for (const StaticGeometryEntry &entry : staticGeometries_) {
    decodeCount += entry.model->submeshes().size();
  }
  decodeWork_.reserve(decodeCount);
  for (const StaticGeometryEntry &entry : staticGeometries_) {
    const uint64_t sourceAddress =
        gpu_.getBufferDeviceAddress(entry.sourceVertexBuffer);
    const uint64_t decodeAddress =
        gpu_.getBufferDeviceAddress(entry.sourceDecodeBuffer);
    const std::span<const Submesh> submeshes = entry.model->submeshes();
    for (uint32_t submeshIndex = 0u;
         submeshIndex < static_cast<uint32_t>(submeshes.size());
         ++submeshIndex) {
      const Submesh &submesh = submeshes[submeshIndex];
      if (submesh.vertexCount == 0u) {
        continue;
      }
      decodeWork_.push_back(DecodeWork{
          .constants =
              DecodePushConstants{
                  .sourceVertices = sourceAddress,
                  .sourceDecode = decodeAddress,
                  .destinationPositions = entry.decodedAddress,
                  .vertexOffset = submesh.vertexOffset,
                  .vertexCount = submesh.vertexCount,
                  .vertexDecodeIndex = submeshIndex,
                  .packedVertexFormat = static_cast<uint32_t>(
                      PackedVertexFormat::StaticQuantized20),
              },
          .dependencies = {entry.sourceVertexBuffer, entry.sourceDecodeBuffer,
                           entry.decodedPositions.get()},
      });
    }
  }
  decodeDispatches_.clear();
  decodeDispatches_.reserve(decodeWork_.size());
  for (const DecodeWork &work : decodeWork_) {
    decodeDispatches_.push_back(ComputeDispatchItem{
        .pipeline = decodePipeline_.get(),
        .dispatch =
            DispatchSize{
                .x = divRoundUp(work.constants.vertexCount,
                                kDecodeWorkgroupSize),
                .y = 1u,
                .z = 1u,
            },
        .pushConstants = bytesOf(work.constants),
        .dependencyBuffers = work.dependencies,
        .dependencyBufferAccessModes = work.accessModes,
        .debugLabel = "RT Decode Static Positions",
        .debugColor = 0xffbb66ffu,
    });
  }
  if (!decodeDispatches_.empty()) {
    auto decodePass = ctx.graph.addGraphicsPass(RenderGraphGraphicsPassDesc{
        .executionMode = RenderPassExecutionMode::ComputeOnly,
        .hasColorAttachment = false,
        .preDispatches = decodeDispatches_,
        .gpuTimingScope = GpuTimingScope::RayTracingBLAS,
        .debugLabel = "RT Decode Static Positions",
        .debugColor = 0xffbb66ffu,
    });
    if (decodePass.hasError()) {
      return Result<bool, std::string>::makeError(decodePass.error());
    }
  }
  if (!dynamicGeometries_.empty()) {
    if (!ctx.shared.animationSceneGpuData.has_value()) {
      return Result<bool, std::string>::makeError(
          "RayTracingScene: dynamic geometry has no animation frame data");
    }
    auto dynamic =
        appendDynamicVertexPass(ctx, *ctx.shared.animationSceneGpuData);
    if (dynamic.hasError()) {
      return dynamic;
    }
  }

  blasBuildItems_.clear();
  blasBufferUses_.clear();
  blasUses_.clear();
  for (const StaticGeometryEntry &entry : staticGeometries_) {
    blasBuildItems_.push_back(AccelerationStructureBuildItem{
        .command = BuildBlasItem{.destination = entry.blas.get(),
                                 .geometries = entry.geometries},
    });
    const auto decoded = ctx.graph.importBuffer(entry.decodedPositions.get(),
                                                "rt_decoded_positions");
    const auto indices =
        ctx.graph.importBuffer(entry.sourceIndexBuffer, "rt_index_input");
    const auto blas =
        ctx.graph.importAccelerationStructure(entry.blas.get(), "rt_blas");
    if (decoded.hasError() || indices.hasError() || blas.hasError()) {
      return Result<bool, std::string>::makeError(
          "RayTracingScene: failed to import BLAS build resources");
    }
    blasBufferUses_.push_back(
        {.buffer = decoded.value(), .access = RenderGraphAccessMode::Read});
    blasBufferUses_.push_back(
        {.buffer = indices.value(), .access = RenderGraphAccessMode::Read});
    blasUses_.push_back({
        .accelerationStructure = blas.value(),
        .access = RenderGraphAccelerationStructureAccess::BuildWrite,
    });
  }
  for (const DynamicGeometryEntry &entry : dynamicGeometries_) {
    blasBuildItems_.push_back(AccelerationStructureBuildItem{
        .command = BuildBlasItem{.destination = entry.blas.get(),
                                 .geometries = entry.geometries},
    });
    const auto vertices = ctx.graph.importBuffer(entry.worldVertices.get(),
                                                 "rt_dynamic_world_vertices");
    const auto indices =
        ctx.graph.importBuffer(entry.sourceIndexBuffer, "rt_index_input");
    const auto blas = ctx.graph.importAccelerationStructure(entry.blas.get(),
                                                            "rt_dynamic_blas");
    if (vertices.hasError() || indices.hasError() || blas.hasError()) {
      return Result<bool, std::string>::makeError(
          "RayTracingScene: failed to import dynamic BLAS build resources");
    }
    blasBufferUses_.push_back(
        {.buffer = vertices.value(), .access = RenderGraphAccessMode::Read});
    blasBufferUses_.push_back(
        {.buffer = indices.value(), .access = RenderGraphAccessMode::Read});
    blasUses_.push_back({
        .accelerationStructure = blas.value(),
        .access = RenderGraphAccelerationStructureAccess::BuildWrite,
    });
  }
  auto blasPass = ctx.graph.addAccelerationStructurePass(
      RenderGraphAccelerationStructurePassDesc{
          .builds = blasBuildItems_,
          .buffers = blasBufferUses_,
          .accelerationStructures = blasUses_,
          .gpuTimingScope = GpuTimingScope::RayTracingBLAS,
          .debugLabel = "RT Scene BLAS Build",
          .debugColor = 0xff9966ffu,
      });
  if (blasPass.hasError()) {
    return Result<bool, std::string>::makeError(blasPass.error());
  }

  tlasUses_.clear();
  for (const StaticGeometryEntry &entry : staticGeometries_) {
    auto blas =
        ctx.graph.importAccelerationStructure(entry.blas.get(), "rt_blas");
    if (blas.hasError()) {
      return Result<bool, std::string>::makeError(blas.error());
    }
    tlasUses_.push_back({
        .accelerationStructure = blas.value(),
        .access = RenderGraphAccelerationStructureAccess::BuildRead,
    });
  }
  for (const DynamicGeometryEntry &entry : dynamicGeometries_) {
    auto blas = ctx.graph.importAccelerationStructure(entry.blas.get(),
                                                      "rt_dynamic_blas");
    if (blas.hasError()) {
      return Result<bool, std::string>::makeError(blas.error());
    }
    tlasUses_.push_back({
        .accelerationStructure = blas.value(),
        .access = RenderGraphAccelerationStructureAccess::BuildRead,
    });
  }
  auto graphTlas =
      ctx.graph.importAccelerationStructure(tlas_.get(), "rt_scene_tlas");
  if (graphTlas.hasError()) {
    return Result<bool, std::string>::makeError(graphTlas.error());
  }
  tlasUses_.push_back({
      .accelerationStructure = graphTlas.value(),
      .access = RenderGraphAccelerationStructureAccess::BuildWrite,
  });
  const std::array tlasBuilds{AccelerationStructureBuildItem{
      .command = BuildTlasItem{.destination = tlas_.get(),
                               .instances = tlasInstances_},
  }};
  auto tlasPass = ctx.graph.addAccelerationStructurePass(
      RenderGraphAccelerationStructurePassDesc{
          .builds = tlasBuilds,
          .accelerationStructures = tlasUses_,
          .gpuTimingScope = GpuTimingScope::RayTracingTLAS,
          .debugLabel = "RT Scene TLAS Build",
          .debugColor = 0xff7755ffu,
      });
  if (tlasPass.hasError()) {
    return Result<bool, std::string>::makeError(tlasPass.error());
  }
  topologyBuildScheduled_ = true;
  scheduledFrameIndex_ = ctx.frame.frameIndex;
  publish(ctx, graphTlas.value());
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> RayTracingScene::appendDynamicVertexPass(
    FrameBuildContext &ctx, const AnimationSceneFrameData &animationData) {
  if (!nuri::isValid(animationData.instanceMatricesBuffer) ||
      animationData.instanceMatricesAddress == 0u ||
      animationData.geometryOverridesByRenderable.size() <
          animationData.renderableCount) {
    return Result<bool, std::string>::makeError(
        "RayTracingScene: incomplete animation frame data");
  }
  dynamicPushConstants_.clear();
  dynamicDispatches_.clear();
  dependencyBuffers_.clear();
  dependencyBufferModes_.clear();
  size_t dispatchCount = 0u;
  for (const DynamicGeometryEntry &entry : dynamicGeometries_) {
    dispatchCount += entry.model->submeshes().size();
  }
  dynamicPushConstants_.reserve(dispatchCount);
  dynamicDispatches_.reserve(dispatchCount);
  appendUnique(dependencyBuffers_, dependencyBufferModes_,
               animationData.instanceMatricesBuffer,
               RenderGraphAccessMode::Read);
  for (DynamicGeometryEntry &entry : dynamicGeometries_) {
    if (entry.renderableIndex >=
        animationData.geometryOverridesByRenderable.size()) {
      return Result<bool, std::string>::makeError(
          "RayTracingScene: dynamic renderable index is out of range");
    }
    const AnimatedRenderableGeometryOverride &overrideGeometry =
        animationData.geometryOverridesByRenderable[entry.renderableIndex];
    const bool hasOverride = overrideGeometry.enabled &&
                             nuri::isValid(overrideGeometry.vertexBuffer) &&
                             overrideGeometry.vertexCount >= entry.vertexCount;
    if (hasOverride) {
      entry.sourceVertexBuffer = overrideGeometry.vertexBuffer;
      entry.sourceVertexByteOffset = overrideGeometry.vertexByteOffset;
      entry.sourcePackedVertexFormat =
          static_cast<uint32_t>(PackedVertexFormat::AnimatedFloat32);
    } else if (!nuri::isValid(entry.sourceDecodeBuffer)) {
      return Result<bool, std::string>::makeError(
          "RayTracingScene: dynamic geometry override is unavailable");
    }
    const uint64_t sourceAddress = gpu_.getBufferDeviceAddress(
        entry.sourceVertexBuffer, entry.sourceVertexByteOffset);
    const uint64_t decodeAddress =
        nuri::isValid(entry.sourceDecodeBuffer)
            ? gpu_.getBufferDeviceAddress(entry.sourceDecodeBuffer)
            : 0u;
    const uint64_t destinationAddress =
        gpu_.getBufferDeviceAddress(entry.worldVertices.get());
    if (sourceAddress == 0u || destinationAddress == 0u) {
      return Result<bool, std::string>::makeError(
          "RayTracingScene: dynamic vertex buffer has no device address");
    }
    const std::span<const Submesh> submeshes = entry.model->submeshes();
    for (uint32_t submeshIndex = 0u;
         submeshIndex < static_cast<uint32_t>(submeshes.size());
         ++submeshIndex) {
      const Submesh &submesh = submeshes[submeshIndex];
      if (submesh.vertexCount == 0u) {
        continue;
      }
      dynamicPushConstants_.push_back(DynamicVertexPushConstants{
          .sourceVertices = sourceAddress,
          .sourceDecode = decodeAddress,
          .instanceData = animationData.instanceMatricesAddress,
          .destinationVertices = destinationAddress,
          .renderableIndex = entry.renderableIndex,
          .vertexOffset = submesh.vertexOffset,
          .vertexCount = submesh.vertexCount,
          .vertexDecodeIndex = submeshIndex,
          .packedVertexFormat = entry.sourcePackedVertexFormat,
      });
      dynamicDispatches_.push_back(ComputeDispatchItem{
          .pipeline = dynamicVertexPipeline_.get(),
          .dispatch = {.x = divRoundUp(submesh.vertexCount,
                                       kDecodeWorkgroupSize),
                       .y = 1u,
                       .z = 1u},
          .pushConstants = bytesOf(dynamicPushConstants_.back()),
          .debugLabel = "RT Prepare Dynamic Vertices",
          .debugColor = 0xffcc7744u,
      });
    }
    appendUnique(dependencyBuffers_, dependencyBufferModes_,
                 entry.sourceVertexBuffer, RenderGraphAccessMode::Read);
    appendUnique(dependencyBuffers_, dependencyBufferModes_,
                 entry.sourceDecodeBuffer, RenderGraphAccessMode::Read);
    appendUnique(dependencyBuffers_, dependencyBufferModes_,
                 entry.worldVertices.get(), RenderGraphAccessMode::Write);
  }
  if (dynamicDispatches_.empty()) {
    return Result<bool, std::string>::makeResult(true);
  }
  auto pass = ctx.graph.addGraphicsPass(RenderGraphGraphicsPassDesc{
      .executionMode = RenderPassExecutionMode::ComputeOnly,
      .hasColorAttachment = false,
      .preDispatches = dynamicDispatches_,
      .dependencyBuffers = dependencyBuffers_,
      .dependencyBufferAccessModes = dependencyBufferModes_,
      .gpuTimingScope = GpuTimingScope::RayTracingBLAS,
      .debugLabel = "RT Prepare Dynamic Vertices",
      .debugColor = 0xffcc7744u,
  });
  if (pass.hasError()) {
    return Result<bool, std::string>::makeError(pass.error());
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> RayTracingScene::appendDynamicUpdatePasses(
    FrameBuildContext &ctx, const AnimationSceneFrameData &animationData) {
  auto vertices = appendDynamicVertexPass(ctx, animationData);
  if (vertices.hasError()) {
    return vertices;
  }
  blasBuildItems_.clear();
  blasBufferUses_.clear();
  blasUses_.clear();
  blasBuildItems_.reserve(dynamicGeometries_.size());
  for (const DynamicGeometryEntry &entry : dynamicGeometries_) {
    blasBuildItems_.push_back(AccelerationStructureBuildItem{
        .command = UpdateBlasItem{.destination = entry.blas.get(),
                                  .geometries = entry.geometries},
    });
    const auto worldVertices = ctx.graph.importBuffer(
        entry.worldVertices.get(), "rt_dynamic_world_vertices");
    const auto indices =
        ctx.graph.importBuffer(entry.sourceIndexBuffer, "rt_index_input");
    const auto blas = ctx.graph.importAccelerationStructure(entry.blas.get(),
                                                            "rt_dynamic_blas");
    if (worldVertices.hasError() || indices.hasError() || blas.hasError()) {
      return Result<bool, std::string>::makeError(
          "RayTracingScene: failed to import dynamic BLAS update resources");
    }
    blasBufferUses_.push_back({.buffer = worldVertices.value(),
                               .access = RenderGraphAccessMode::Read});
    blasBufferUses_.push_back(
        {.buffer = indices.value(), .access = RenderGraphAccessMode::Read});
    blasUses_.push_back({
        .accelerationStructure = blas.value(),
        .access = RenderGraphAccelerationStructureAccess::BuildWrite,
    });
  }
  auto blasPass = ctx.graph.addAccelerationStructurePass(
      RenderGraphAccelerationStructurePassDesc{
          .builds = blasBuildItems_,
          .buffers = blasBufferUses_,
          .accelerationStructures = blasUses_,
          .gpuTimingScope = GpuTimingScope::RayTracingBLAS,
          .debugLabel = "RT Dynamic BLAS Update",
          .debugColor = 0xffcc6655u,
      });
  if (blasPass.hasError()) {
    return Result<bool, std::string>::makeError(blasPass.error());
  }
  pendingAnimationVersion_ = animationData.version;
  dynamicUpdateScheduled_ = true;
  scheduledFrameIndex_ = ctx.frame.frameIndex;
  ctx.frame.metrics.rayTracingScene.dynamicBlasUpdates =
      static_cast<uint32_t>(dynamicGeometries_.size());
  return appendTlasUpdatePass(ctx);
}

Result<bool, std::string>
RayTracingScene::updateTransforms(FrameBuildContext &ctx,
                                  const SceneDrawDatabase &database) {
  const std::span<const SceneInstanceRecord> instances = database.instances();
  for (size_t rtIndex = 0u; rtIndex < instanceRecords_.size(); ++rtIndex) {
    RtInstanceGpuData &record = instanceRecords_[rtIndex];
    if (record.renderableIndex >= instances.size() ||
        instances[record.renderableIndex].renderable == nullptr) {
      return Result<bool, std::string>::makeError(
          "RayTracingScene: renderable topology changed during TLAS update");
    }
    if (record.flags == 0u) {
      const glm::mat4 &world =
          instances[record.renderableIndex].renderable->modelMatrix;
      record.worldFromObject = world;
      record.objectFromWorld = safeInverseOrIdentity(world);
      tlasInstances_[rtIndex].transform =
          makeAccelerationStructureTransform(world);
    }
  }
  auto upload = gpu_.updateBuffer(instanceTable_.get(),
                                  bytesOf(std::span(instanceRecords_)));
  if (upload.hasError()) {
    return upload;
  }
  pendingTransformVersion_ = ctx.frame.scene->transformVersion();
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
RayTracingScene::appendTlasUpdatePass(FrameBuildContext &ctx) {
  tlasUses_.clear();
  for (const StaticGeometryEntry &entry : staticGeometries_) {
    auto blas =
        ctx.graph.importAccelerationStructure(entry.blas.get(), "rt_blas");
    if (blas.hasError()) {
      return Result<bool, std::string>::makeError(blas.error());
    }
    tlasUses_.push_back({
        .accelerationStructure = blas.value(),
        .access = RenderGraphAccelerationStructureAccess::BuildRead,
    });
  }
  for (const DynamicGeometryEntry &entry : dynamicGeometries_) {
    auto blas = ctx.graph.importAccelerationStructure(entry.blas.get(),
                                                      "rt_dynamic_blas");
    if (blas.hasError()) {
      return Result<bool, std::string>::makeError(blas.error());
    }
    tlasUses_.push_back({
        .accelerationStructure = blas.value(),
        .access = RenderGraphAccelerationStructureAccess::BuildRead,
    });
  }
  auto graphTlas =
      ctx.graph.importAccelerationStructure(tlas_.get(), "rt_scene_tlas");
  if (graphTlas.hasError()) {
    return Result<bool, std::string>::makeError(graphTlas.error());
  }
  tlasUses_.push_back({
      .accelerationStructure = graphTlas.value(),
      .access = RenderGraphAccelerationStructureAccess::BuildWrite,
  });
  const std::array tlasUpdates{AccelerationStructureBuildItem{
      .command = UpdateTlasItem{.destination = tlas_.get(),
                                .instances = tlasInstances_},
  }};
  auto pass = ctx.graph.addAccelerationStructurePass(
      RenderGraphAccelerationStructurePassDesc{
          .builds = tlasUpdates,
          .accelerationStructures = tlasUses_,
          .gpuTimingScope = GpuTimingScope::RayTracingTLAS,
          .debugLabel = "RT Scene TLAS Update",
          .debugColor = 0xff7755ffu,
      });
  if (pass.hasError()) {
    return Result<bool, std::string>::makeError(pass.error());
  }
  publish(ctx, graphTlas.value());
  ctx.frame.metrics.rayTracingScene.tlasUpdates = 1u;
  return Result<bool, std::string>::makeResult(true);
}

void RayTracingScene::rebuildIndirectReferences(
    const FrameSharedResources &shared) {
  const MaterialTableGpuData *material = shared.materialTableGpuData.has_value()
                                             ? &*shared.materialTableGpuData
                                             : nullptr;
  const std::array<BufferHandle, 7u> inputs{
      instanceTable_.get(),
      geometryTable_.get(),
      material != nullptr ? material->headerBuffer : BufferHandle{},
      material != nullptr ? material->clearcoatBuffer : BufferHandle{},
      material != nullptr ? material->sheenBuffer : BufferHandle{},
      material != nullptr ? material->transmissionBuffer : BufferHandle{},
      material != nullptr ? material->specularBuffer : BufferHandle{},
  };
  if (!indirectReferencesDirty_ && inputs == indirectReferenceInputs_) {
    return;
  }
  indirectReferences_.clear();
  appendUnique(indirectReferences_, instanceTable_.get());
  appendUnique(indirectReferences_, geometryTable_.get());
  for (const StaticGeometryEntry &entry : staticGeometries_) {
    appendUnique(indirectReferences_, entry.sourceVertexBuffer);
    appendUnique(indirectReferences_, entry.sourceDecodeBuffer);
    appendUnique(indirectReferences_, entry.sourceIndexBuffer);
    appendUnique(indirectReferences_, entry.decodedPositions.get());
  }
  for (const DynamicGeometryEntry &entry : dynamicGeometries_) {
    appendUnique(indirectReferences_, entry.sourceVertexBuffer);
    appendUnique(indirectReferences_, entry.sourceIndexBuffer);
    appendUnique(indirectReferences_, entry.worldVertices.get());
  }
  if (material != nullptr) {
    appendUnique(indirectReferences_, material->headerBuffer);
    appendUnique(indirectReferences_, material->clearcoatBuffer);
    appendUnique(indirectReferences_, material->sheenBuffer);
    appendUnique(indirectReferences_, material->transmissionBuffer);
    appendUnique(indirectReferences_, material->specularBuffer);
  }
  indirectReferenceInputs_ = inputs;
  indirectReferencesDirty_ = false;
}

void RayTracingScene::publish(FrameBuildContext &ctx,
                              RenderGraphAccelerationStructureId graphTlas) {
  rebuildIndirectReferences(ctx.shared);
  RayTracingSceneReadiness readiness = RayTracingSceneReadiness::Building;
  if (failed_) {
    readiness = RayTracingSceneReadiness::Failed;
  } else if (ready_) {
    readiness = RayTracingSceneReadiness::Ready;
  }
  const MaterialTableGpuData *materials =
      ctx.shared.materialTableGpuData.has_value()
          ? &*ctx.shared.materialTableGpuData
          : nullptr;
  ctx.shared.rayTracingScene = RayTracingSceneFrameView{
      .topLevelAccelerationStructure = tlas_.get(),
      .graphTopLevelAccelerationStructure = graphTlas,
      .instanceTable = instanceTable_.get(),
      .geometryTable = geometryTable_.get(),
      .materialTable =
          materials != nullptr ? materials->headerBuffer : BufferHandle{},
      .instanceTableAddress = gpu_.getBufferDeviceAddress(instanceTable_.get()),
      .geometryTableAddress = gpu_.getBufferDeviceAddress(geometryTable_.get()),
      .materialTableAddress =
          materials != nullptr ? materials->headerBufferAddress : 0u,
      .indirectSubmissionReferences = indirectReferences_,
      .indirectSubmissionTextureReferences =
          ctx.shared.sceneDrawDatabase->rayTracingMaterialTextures(),
      .sceneId = sceneId_,
      .topologyVersion = topologyVersion_,
      .transformVersion = transformUpdateScheduled_ ? pendingTransformVersion_
                                                    : transformVersion_,
      .deformationVersion = dynamicUpdateScheduled_ ? pendingAnimationVersion_
                                                    : deformationVersion_,
      .geometryMutationVersion = geometryMutationVersion_,
      .instanceCount = static_cast<uint32_t>(instanceRecords_.size()),
      .geometryCount = static_cast<uint32_t>(geometryRecords_.size()),
      .completedBlasBuilds =
          ready_ ? static_cast<uint32_t>(staticGeometries_.size() +
                                         dynamicGeometries_.size())
                 : 0u,
      .totalBlasBuilds = static_cast<uint32_t>(staticGeometries_.size() +
                                               dynamicGeometries_.size()),
      .readiness = readiness,
      .ready = ready_,
  };
  RayTracingSceneFrameMetrics &metrics = ctx.frame.metrics.rayTracingScene;
  metrics.staticInstances = staticInstanceCount_;
  metrics.dynamicInstances = static_cast<uint32_t>(dynamicGeometries_.size());
  metrics.excludedDynamicInstances = excludedDynamicInstances_;
  metrics.staticBlasCount = static_cast<uint32_t>(staticGeometries_.size());
  metrics.dynamicBlasCount = static_cast<uint32_t>(dynamicGeometries_.size());
  metrics.tlasCount = nuri::isValid(tlas_.get()) ? 1u : 0u;
  metrics.uniqueStaticGeometry =
      static_cast<uint32_t>(staticGeometries_.size());
  metrics.geometryRecords = static_cast<uint32_t>(geometryRecords_.size());
  metrics.decodeDispatches = static_cast<uint32_t>(decodeWork_.size());
  metrics.blasBuilds = topologyBuildScheduled_
                           ? static_cast<uint32_t>(staticGeometries_.size() +
                                                   dynamicGeometries_.size())
                           : 0u;
  metrics.dynamicVertexDispatches =
      topologyBuildScheduled_ || dynamicUpdateScheduled_
          ? static_cast<uint32_t>(dynamicDispatches_.size())
          : 0u;
  metrics.tlasBuilds = topologyBuildScheduled_ ? 1u : 0u;
  metrics.queuedBlasBuilds =
      ready_ ? 0u : metrics.staticBlasCount + metrics.dynamicBlasCount;
  metrics.consumedRebuildEpoch = consumedRebuildEpoch_;
  metrics.readiness = readiness;
  for (const StaticGeometryEntry &entry : staticGeometries_) {
    metrics.decodedVertices += entry.vertexCount;
    metrics.decodedPositionBytes +=
        static_cast<uint64_t>(entry.vertexCount) * 16u;
    for (const auto &geometry : entry.geometries) {
      metrics.triangles += geometry.indexCount / 3u;
    }
  }
  for (const DynamicGeometryEntry &entry : dynamicGeometries_) {
    for (const auto &geometry : entry.geometries) {
      metrics.triangles += geometry.indexCount / 3u;
    }
  }
  metrics.tableBytes = instanceRecords_.size() * sizeof(RtInstanceGpuData) +
                       geometryRecords_.size() * sizeof(RtGeometryGpuData);
  if (hasGpuTimingScope(ctx.frame.gpuTiming, GpuTimingScope::RayTracingScene)) {
    metrics.gpuTimeMs = ctx.frame.gpuTiming.rayTracingSceneTimeMs;
    metrics.gpuTimingSourceFrameIndex =
        ctx.frame.gpuTiming.rayTracingSceneSourceFrameIndex;
    metrics.gpuTimingAvailable = 1u;
  }
  if (hasGpuTimingScope(ctx.frame.gpuTiming, GpuTimingScope::RayTracingBLAS)) {
    metrics.blasGpuTimeMs = ctx.frame.gpuTiming.rayTracingBlasTimeMs;
    metrics.blasGpuTimingAvailable = 1u;
  }
  if (hasGpuTimingScope(ctx.frame.gpuTiming, GpuTimingScope::RayTracingTLAS)) {
    metrics.tlasGpuTimeMs = ctx.frame.gpuTiming.rayTracingTlasTimeMs;
    metrics.tlasGpuTimingAvailable = 1u;
  }
}

Result<bool, std::string> RayTracingScene::prepare(FrameBuildContext &ctx) {
  NURI_PROFILER_FUNCTION();
  ctx.shared.rayTracingScene.reset();
  ctx.frame.metrics.rayTracingScene = {};
  ctx.frame.metrics.rayTracingScene.consumedRebuildEpoch =
      consumedRebuildEpoch_;
  topologyBuildScheduled_ = false;
  const RenderSettings::DDGISettings &settings =
      renderSettingsOrDefault(ctx.frame).ddgi;
  if (!settings.enabled || ctx.frame.scene == nullptr) {
    clearSceneResources();
    ctx.frame.metrics.rayTracingScene.readiness =
        RayTracingSceneReadiness::Disabled;
    return Result<bool, std::string>::makeResult(true);
  }
  const RayTracingCapabilities &caps = gpu_.getDeviceCaps().rayTracing;
  if (!caps.accelerationStructure || !caps.rayQuery ||
      !caps.bufferDeviceAddress) {
    clearSceneResources();
    ctx.frame.metrics.rayTracingScene.readiness =
        RayTracingSceneReadiness::Unsupported;
    return Result<bool, std::string>::makeResult(true);
  }
  if (!initializationError_.empty()) {
    return Result<bool, std::string>::makeError(initializationError_);
  }
  if (ctx.shared.sceneDrawDatabase == nullptr) {
    return Result<bool, std::string>::makeError(
        "RayTracingScene: SceneDrawDatabase was not published");
  }
  pollCompletion();
  const RenderScene &scene = *ctx.frame.scene;
  const bool explicitRebuild = ddgiEpochIsPending(
      settings.requestedEpochs.rebuildRayTracingScene, consumedRebuildEpoch_);
  const uint64_t geometryVersion = gpu_.geometryMutationVersion();
  const AnimationSceneFrameData *animationData =
      ctx.shared.animationSceneGpuData.has_value()
          ? &*ctx.shared.animationSceneGpuData
          : nullptr;
  const bool matchingAnimationData =
      animationData != nullptr && animationData->scene == &scene &&
      animationData->sceneTopologyVersion == scene.topologyVersion() &&
      animationData->renderableCount ==
          ctx.shared.sceneDrawDatabase->instances().size();
  const bool topologyChanged =
      explicitRebuild || sceneId_ != scene.id() ||
      topologyVersion_ != scene.topologyVersion() ||
      geometryMutationVersion_ != geometryVersion || failed_ ||
      (matchingAnimationData && animationVersion_ == 0u &&
       animationData->version != 0u);
  if (topologyChanged) {
    auto rebuild = rebuildStaticScene(ctx, *ctx.shared.sceneDrawDatabase);
    if (rebuild.hasError()) {
      failed_ = true;
      return rebuild;
    }
    consumedRebuildEpoch_ = std::max(
        consumedRebuildEpoch_, settings.requestedEpochs.rebuildRayTracingScene);
    if (!nuri::isValid(tlas_.get())) {
      ctx.frame.metrics.rayTracingScene.readiness =
          ready_ ? RayTracingSceneReadiness::Ready
                 : RayTracingSceneReadiness::Building;
      return Result<bool, std::string>::makeResult(true);
    }
    return Result<bool, std::string>::makeResult(true);
  }
  RenderGraphAccelerationStructureId graphTlas{};
  if (nuri::isValid(tlas_.get())) {
    auto imported =
        ctx.graph.importAccelerationStructure(tlas_.get(), "rt_scene_tlas");
    if (imported.hasError()) {
      return Result<bool, std::string>::makeError(imported.error());
    }
    graphTlas = imported.value();
  }
  const bool transformsChanged =
      ready_ && transformVersion_ != scene.transformVersion();
  if (transformsChanged) {
    auto transforms = updateTransforms(ctx, *ctx.shared.sceneDrawDatabase);
    if (transforms.hasError()) {
      return transforms;
    }
    transformUpdateScheduled_ = true;
    scheduledFrameIndex_ = ctx.frame.frameIndex;
  }
  const bool dynamicChanged = ready_ && !dynamicGeometries_.empty() &&
                              matchingAnimationData &&
                              animationVersion_ != animationData->version;
  if (dynamicChanged) {
    return appendDynamicUpdatePasses(ctx, *animationData);
  }
  if (transformsChanged) {
    return appendTlasUpdatePass(ctx);
  }
  publish(ctx, graphTlas);
  return Result<bool, std::string>::makeResult(true);
}

void RayTracingScene::onFrameSubmitted(
    const RenderFrameContext &frame) noexcept {
  if (scheduledFrameIndex_ != frame.frameIndex) {
    return;
  }
  if (topologyBuildScheduled_) {
    auto completion = gpu_.captureWorkCompletion();
    if (completion.hasError()) {
      failed_ = true;
      return;
    }
    buildCompletion_ = completion.value();
    // The next frame is submitted to the same graphics queue, so queue order is
    // sufficient for consumers once this build submission has succeeded. Keep
    // the completion token for retirement/lifetime tracking, not readiness.
    ready_ = true;
    topologyBuildScheduled_ = false;
  }
  if (dynamicUpdateScheduled_) {
    animationVersion_ = pendingAnimationVersion_;
    deformationVersion_ = pendingAnimationVersion_;
    dynamicUpdateScheduled_ = false;
  }
  if (transformUpdateScheduled_) {
    transformVersion_ = pendingTransformVersion_;
    transformUpdateScheduled_ = false;
  }
  scheduledFrameIndex_ = UINT64_MAX;
}

void RayTracingScene::onFrameAbandoned(
    const RenderFrameContext &frame) noexcept {
  if (topologyBuildScheduled_ && scheduledFrameIndex_ == frame.frameIndex) {
    failed_ = true;
    topologyBuildScheduled_ = false;
  }
  if (dynamicUpdateScheduled_ && scheduledFrameIndex_ == frame.frameIndex) {
    dynamicUpdateScheduled_ = false;
    pendingAnimationVersion_ = animationVersion_;
  }
  if (transformUpdateScheduled_ && scheduledFrameIndex_ == frame.frameIndex) {
    transformUpdateScheduled_ = false;
    pendingTransformVersion_ = transformVersion_;
  }
  if (scheduledFrameIndex_ == frame.frameIndex) {
    scheduledFrameIndex_ = UINT64_MAX;
  }
}

} // namespace nuri
