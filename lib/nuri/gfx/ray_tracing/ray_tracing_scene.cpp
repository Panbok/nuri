#include "nuri/gfx/ray_tracing/ray_tracing_scene.h"

#include "nuri/core/containers/hash_map.h"
#include "nuri/core/profiling.h"
#include "nuri/gfx/renderers/scene_draw_database.h"
#include "nuri/gfx/shader.h"
#include "nuri/gfx/sim/animation_scene_frame_data.h"
#include "nuri/math/utils.h"
#include "nuri/resources/gpu/resource_manager.h"
#include "nuri/scene/render_scene.h"

namespace nuri {
namespace {

constexpr uint32_t kDecodeWorkgroupSize = 64u;
constexpr uint32_t kRtGeometryDoubleSided = 1u << 0u;
constexpr uint32_t kRtGeometryAlphaMasked = 1u << 1u;
constexpr uint32_t kRtGeometryAlphaBlended = 1u << 2u;
constexpr uint32_t kRtGeometryTransmission = 1u << 3u;

[[nodiscard]] AccelerationStructureTransform
makeAccelerationStructureTransform(const glm::mat4 &matrix) noexcept {
  return AccelerationStructureTransform{.rowMajor3x4 = {
                                            matrix[0][0],
                                            matrix[1][0],
                                            matrix[2][0],
                                            matrix[3][0],
                                            matrix[0][1],
                                            matrix[1][1],
                                            matrix[2][1],
                                            matrix[3][1],
                                            matrix[0][2],
                                            matrix[1][2],
                                            matrix[2][2],
                                            matrix[3][2],
                                        }};
}

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

void appendUnique(std::pmr::vector<RenderGraphBufferUse> &output,
                  RenderGraphBufferId buffer, RenderGraphAccessMode access) {
  const auto found =
      std::ranges::find_if(output, [buffer](const RenderGraphBufferUse use) {
        return use.buffer.value == buffer.value;
      });
  if (found == output.end()) {
    output.push_back({.buffer = buffer, .access = access});
    return;
  }
  found->access = found->access | access;
}

} // namespace

RayTracingScene::RayTracingScene(GPUDevice &gpu, RuntimeDDGIShaderConfig config,
                                 std::pmr::memory_resource *memory)
    : gpu_(gpu), config_(std::move(config)),
      memory_(memory != nullptr ? memory : std::pmr::get_default_resource()),
      staticGeometries_(memory_), dynamicGeometries_(memory_),
      instanceRecords_(memory_), geometryRecords_(memory_),
      surfaceBoundsRecords_(memory_), currentGeometryBounds_(memory_),
      committedGeometryBounds_(memory_), tlasInstances_(memory_),
      decodeWork_(memory_), decodeDispatches_(memory_),
      dynamicPushConstants_(memory_), dynamicDispatches_(memory_),
      dispatchBufferUses_(memory_), blasBuildItems_(memory_),
      blasBufferUses_(memory_), blasUses_(memory_), tlasUses_(memory_),
      indirectReferences_(memory_) {
  auto result = initialize();
  if (result.hasError()) {
    initializationError_ = result.error();
  }
}

RayTracingScene::~RayTracingScene() { clearSceneResources(); }

Result<bool, std::string> RayTracingScene::initialize() {
  auto shader =
      compileShaderFile(gpu_, "rt_decode_positions",
                        config_.decodePositions.string(), ShaderStage::Compute);
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
  auto dynamicShader = compileShaderFile(
      gpu_, "rt_prepare_dynamic_vertices",
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

void RayTracingScene::clearSceneResources(bool clearChangeTracking) noexcept {
  staticGeometries_.clear();
  dynamicGeometries_.clear();
  instanceTable_.reset();
  geometryTable_.reset();
  surfaceBounds_.reset();
  tlas_.reset();
  instanceRecords_.clear();
  geometryRecords_.clear();
  surfaceBoundsRecords_.clear();
  tlasInstances_.clear();
  decodeWork_.clear();
  decodeDispatches_.clear();
  dynamicPushConstants_.clear();
  dynamicDispatches_.clear();
  dispatchBufferUses_.clear();
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
  pendingFrame_ = PendingFrame{.rebuildEpoch = consumedRebuildEpoch_};
  animationVersion_ = 0u;
  sceneId_ = 0u;
  topologyVersion_ = 0u;
  transformVersion_ = 0u;
  deformationVersion_ = 0u;
  geometryMutationVersion_ = 0u;
  staticInstanceCount_ = 0u;
  staticSurfaceBoundsCount_ = 0u;
  dynamicSurfaceBoundsCount_ = 0u;
  staticSurfaceBoundsAvailable_ = false;
  dynamicSurfaceBoundsAvailable_ = false;
  excludedDynamicInstances_ = 0u;
  currentGeometryBounds_.clear();
  currentStaticCoverageBounds_ = {};
  if (clearChangeTracking) {
    committedGeometryBounds_.clear();
    pendingGeometryChanges_ = {};
    committedGeometryChanges_ = {};
    pendingGeometryChangeCount_ = 0u;
    committedGeometryChangeCount_ = 0u;
    geometryChangeFrameIndex_ = UINT64_MAX;
    geometryChangeOverflow_ = false;
    committedGeometryChangeOverflow_ = false;
  }
}

void RayTracingScene::pollCompletion() noexcept {
  if (!ready_ && nuri::isValid(buildCompletion_) &&
      gpu_.isSubmissionComplete(buildCompletion_)) {
    buildCompletion_ = {};
    ready_ = true;
  }
}

Result<bool, std::string> RayTracingScene::appendInstance(
    std::span<const SceneDrawRecord> draws, uint32_t renderableIndex,
    const InstanceGeometrySource &source, uint32_t expectedGeometryCount,
    const glm::mat4 &worldFromObject, AccelerationStructureHandle blas,
    bool dynamic) {
  if (instanceRecords_.size() > kRayTracingInstanceCustomIndexLimit) {
    return Result<bool, std::string>::makeError(
        "RayTracingScene: TLAS custom-index limit exceeded");
  }
  const uint32_t firstGeometry = static_cast<uint32_t>(geometryRecords_.size());
  const uint64_t indexBaseAddress =
      gpu_.getBufferDeviceAddress(source.indexBuffer, source.indexBaseOffset);
  for (const SceneDrawRecord &draw : draws) {
    if (draw.submesh == nullptr) {
      continue;
    }
    const uint32_t flags = (draw.doubleSided ? kRtGeometryDoubleSided : 0u) |
                           (draw.alphaMasked ? kRtGeometryAlphaMasked : 0u) |
                           (draw.alphaBlended ? kRtGeometryAlphaBlended : 0u) |
                           (draw.transmission ? kRtGeometryTransmission : 0u);
    geometryRecords_.push_back(RtGeometryGpuData{
        .indexBufferAddress = indexBaseAddress,
        .vertexBufferAddress = source.useDrawVertexLayout
                                   ? draw.baseVertexBufferAddress
                                   : source.vertexAddress,
        .vertexDecodeAddress = source.useDrawVertexLayout
                                   ? draw.baseVertexDecodeBufferAddress
                                   : source.vertexDecodeAddress,
        .materialIndex = draw.materialIndex,
        .indexOffset = lod0IndexOffset(*draw.submesh),
        .vertexOffset = 0u,
        .indexFormatAndVertexFormat = packRtGeometryFormats(
            source.useDrawVertexLayout ? draw.indexFormat : source.indexFormat,
            source.useDrawVertexLayout ? draw.basePackedVertexFormat
                                       : source.packedVertexFormat),
        .flags = flags,
    });
  }
  const uint32_t geometryCount =
      static_cast<uint32_t>(geometryRecords_.size()) - firstGeometry;
  if (geometryCount != expectedGeometryCount) {
    return Result<bool, std::string>::makeError(
        dynamic ? "RayTracingScene: dynamic instance and BLAS layouts differ"
                : "RayTracingScene: instance and BLAS geometry layouts differ");
  }
  const uint32_t rtInstanceIndex =
      static_cast<uint32_t>(instanceRecords_.size());
  instanceRecords_.push_back(RtInstanceGpuData{
      .firstGeometryRecord = firstGeometry,
      .geometryCount = geometryCount,
      .renderableIndex = renderableIndex,
      .flags = dynamic ? 1u : 0u,
      .worldFromObject = worldFromObject,
      .objectFromWorld = safeInverseOrIdentity(worldFromObject),
  });
  tlasInstances_.push_back(AccelerationStructureInstanceDesc{
      .transform = makeAccelerationStructureTransform(worldFromObject),
      .customIndex = rtInstanceIndex,
      .mask = 0xffu,
      .flags = AccelerationStructureInstanceFlags::TriangleCullDisable |
               AccelerationStructureInstanceFlags::ForceNonOpaque,
      .bottomLevel = blas,
  });
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
RayTracingScene::rebuildStaticScene(FrameBuildContext &ctx,
                                    const SceneDrawDatabase &database) {
  const RenderScene &scene = *ctx.frame.scene;
  const bool sceneChanged = sceneId_ != 0u && sceneId_ != scene.id();
  clearSceneResources(false);
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
      const uint64_t worldVertexAddress =
          gpu_.getBufferDeviceAddress(entry.worldVertices.get());
      auto appended = appendInstance(
          instanceDraws, instanceIndex,
          InstanceGeometrySource{
              .indexBuffer = entry.sourceIndexBuffer,
              .indexBaseOffset = entry.sourceIndexBaseOffset,
              .vertexAddress = worldVertexAddress,
              .indexFormat = entry.indexFormat,
              .packedVertexFormat =
                  static_cast<uint32_t>(PackedVertexFormat::AnimatedFloat32),
          },
          static_cast<uint32_t>(entry.geometries.size()), glm::mat4(1.0f),
          entry.blas.get(), true);
      if (appended.hasError()) {
        return appended;
      }
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
    const glm::mat4 &world = instance.renderable->modelMatrix;
    auto appended =
        appendInstance(instanceDraws, instanceIndex,
                       InstanceGeometrySource{
                           .indexBuffer = geometry.sourceIndexBuffer,
                           .indexBaseOffset = geometry.sourceIndexBaseOffset,
                           .useDrawVertexLayout = true,
                       },
                       static_cast<uint32_t>(geometry.geometries.size()), world,
                       geometry.blas.get(), false);
    if (appended.hasError()) {
      return appended;
    }
    ++staticInstanceCount_;
  }

  rebuildSurfaceBounds(database);
  beginGeometryChanges(ctx.frame.frameIndex, false);
  prepareTopologyChangeRegions(scene.topologyVersion(), sceneChanged);
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

void RayTracingScene::rebuildSurfaceBounds(const SceneDrawDatabase &database) {
  surfaceBoundsRecords_.clear();
  currentGeometryBounds_.clear();
  staticSurfaceBoundsCount_ = 0u;
  dynamicSurfaceBoundsCount_ = 0u;
  staticSurfaceBoundsAvailable_ = true;
  dynamicSurfaceBoundsAvailable_ = true;
  const std::span<const SceneInstanceRecord> instances = database.instances();
  const auto appendCategory = [&](bool dynamic) {
    for (const RtInstanceGpuData &record : instanceRecords_) {
      const bool recordDynamic = record.flags != 0u;
      if (recordDynamic != dynamic ||
          record.renderableIndex >= instances.size()) {
        continue;
      }
      const SceneInstanceRecord &instance = instances[record.renderableIndex];
      if (instance.model == nullptr || instance.renderable == nullptr) {
        (dynamic ? dynamicSurfaceBoundsAvailable_
                 : staticSurfaceBoundsAvailable_) = false;
        continue;
      }
      const BoundingBox worldBounds = instance.model->bounds().getTransformed(
          instance.renderable->modelMatrix);
      const bool valid =
          std::isfinite(worldBounds.min_.x) &&
          std::isfinite(worldBounds.min_.y) &&
          std::isfinite(worldBounds.min_.z) &&
          std::isfinite(worldBounds.max_.x) &&
          std::isfinite(worldBounds.max_.y) &&
          std::isfinite(worldBounds.max_.z) &&
          glm::all(glm::lessThanEqual(worldBounds.min_, worldBounds.max_));
      currentGeometryBounds_.push_back(GeometryBoundsSnapshot{
          .bounds = worldBounds,
          .sourceId = instance.renderable->id.value,
          .dynamic = dynamic,
          .deforming = dynamic && instance.dynamicCaster,
          .boundsKnown = valid && !(dynamic && instance.dynamicCaster),
      });
      if (dynamic && instance.dynamicCaster) {
        dynamicSurfaceBoundsAvailable_ = false;
        continue;
      }
      if (!valid) {
        (dynamic ? dynamicSurfaceBoundsAvailable_
                 : staticSurfaceBoundsAvailable_) = false;
        continue;
      }
      if (surfaceBoundsRecords_.size() >= kMaxDDGISurfaceBounds) {
        staticSurfaceBoundsAvailable_ = false;
        dynamicSurfaceBoundsAvailable_ = false;
        return;
      }
      surfaceBoundsRecords_.push_back(RtSurfaceBoundsGpuData{
          .minimum = glm::vec4(worldBounds.min_, 0.0f),
          .maximum = glm::vec4(worldBounds.max_, 0.0f),
          .metadata =
              glm::uvec4(dynamic ? 1u : 0u, record.renderableIndex, 0u, 0u),
      });
      if (dynamic) {
        ++dynamicSurfaceBoundsCount_;
      } else {
        ++staticSurfaceBoundsCount_;
      }
    }
  };
  appendCategory(false);
  appendCategory(true);
  std::ranges::sort(currentGeometryBounds_,
                    [](const GeometryBoundsSnapshot &left,
                       const GeometryBoundsSnapshot &right) {
                      return left.sourceId < right.sourceId;
                    });
  currentStaticCoverageBounds_.complete = staticSurfaceBoundsAvailable_;
  for (const GeometryBoundsSnapshot &snapshot : currentGeometryBounds_) {
    if (snapshot.dynamic) {
      continue;
    }
    if (!snapshot.boundsKnown) {
      currentStaticCoverageBounds_.complete = false;
      continue;
    }
    if (!currentStaticCoverageBounds_.valid) {
      currentStaticCoverageBounds_.bounds = snapshot.bounds;
      currentStaticCoverageBounds_.valid = true;
    } else {
      currentStaticCoverageBounds_.bounds.combinePoint(snapshot.bounds.min_);
      currentStaticCoverageBounds_.bounds.combinePoint(snapshot.bounds.max_);
    }
  }
}

void RayTracingScene::beginGeometryChanges(uint64_t frameIndex,
                                           bool append) noexcept {
  if (!append || geometryChangeFrameIndex_ != frameIndex) {
    pendingGeometryChanges_ = {};
    pendingGeometryChangeCount_ = 0u;
    geometryChangeOverflow_ = false;
  }
  geometryChangeFrameIndex_ = frameIndex;
}

void RayTracingScene::publishGeometryChange(
    DDGISceneChangeRegion change) noexcept {
  if (!change.boundsKnown) {
    if (pendingGeometryChangeCount_ != 0u) {
      change.kind = DDGISceneChangeKind::StaticTopology;
    }
    pendingGeometryChanges_ = {};
    pendingGeometryChanges_[0] = change;
    pendingGeometryChangeCount_ = 1u;
    geometryChangeOverflow_ = true;
    return;
  }
  if (geometryChangeOverflow_) {
    return;
  }
  if (pendingGeometryChangeCount_ == kMaxDDGIGeometryChangeRegions) {
    pendingGeometryChanges_ = {};
    pendingGeometryChanges_[0] = DDGISceneChangeRegion{
        .kind = DDGISceneChangeKind::StaticTopology,
        .sourceVersion = change.sourceVersion,
        .boundsKnown = false,
    };
    pendingGeometryChangeCount_ = 1u;
    geometryChangeOverflow_ = true;
    return;
  }
  pendingGeometryChanges_[pendingGeometryChangeCount_++] = change;
}

void RayTracingScene::prepareTopologyChangeRegions(uint64_t sourceVersion,
                                                   bool sceneChanged) noexcept {
  if (committedGeometryBounds_.empty()) {
    return;
  }
  if (sceneChanged) {
    publishGeometryChange(DDGISceneChangeRegion{
        .kind = DDGISceneChangeKind::StaticTopology,
        .sourceVersion = sourceVersion,
        .boundsKnown = false,
    });
    return;
  }

  size_t previousIndex = 0u;
  size_t currentIndex = 0u;
  while (previousIndex < committedGeometryBounds_.size() ||
         currentIndex < currentGeometryBounds_.size()) {
    const GeometryBoundsSnapshot *previous =
        previousIndex < committedGeometryBounds_.size()
            ? &committedGeometryBounds_[previousIndex]
            : nullptr;
    const GeometryBoundsSnapshot *current =
        currentIndex < currentGeometryBounds_.size()
            ? &currentGeometryBounds_[currentIndex]
            : nullptr;
    const uint64_t previousId =
        previous != nullptr ? previous->sourceId : UINT64_MAX;
    const uint64_t currentId =
        current != nullptr ? current->sourceId : UINT64_MAX;
    const GeometryBoundsSnapshot *first = nullptr;
    const GeometryBoundsSnapshot *second = nullptr;
    if (previousId <= currentId) {
      first = previous;
      ++previousIndex;
    }
    if (currentId <= previousId) {
      second = current;
      ++currentIndex;
    }
    const GeometryBoundsSnapshot *identity = second != nullptr ? second : first;
    DDGISceneChangeRegion change{
        .kind = DDGISceneChangeKind::StaticTopology,
        .sourceId = identity != nullptr ? identity->sourceId : 0u,
        .sourceVersion = sourceVersion,
    };
    if (first != nullptr && second != nullptr && first->boundsKnown &&
        second->boundsKnown) {
      change.worldBounds =
          BoundingBox{glm::min(first->bounds.min_, second->bounds.min_),
                      glm::max(first->bounds.max_, second->bounds.max_)};
      change.boundsKnown = true;
    } else if (identity != nullptr && identity->boundsKnown) {
      change.worldBounds = identity->bounds;
      change.boundsKnown = true;
    }
    publishGeometryChange(change);
  }
}

void RayTracingScene::prepareTransformChangeRegions(
    uint64_t sourceVersion) noexcept {
  size_t previousIndex = 0u;
  size_t currentIndex = 0u;
  while (previousIndex < committedGeometryBounds_.size() &&
         currentIndex < currentGeometryBounds_.size()) {
    const GeometryBoundsSnapshot &previous =
        committedGeometryBounds_[previousIndex];
    const GeometryBoundsSnapshot &current =
        currentGeometryBounds_[currentIndex];
    if (previous.sourceId < current.sourceId) {
      ++previousIndex;
      continue;
    }
    if (current.sourceId < previous.sourceId) {
      ++currentIndex;
      continue;
    }
    ++previousIndex;
    ++currentIndex;
    const bool unchanged =
        previous.boundsKnown && current.boundsKnown &&
        glm::all(glm::equal(previous.bounds.min_, current.bounds.min_)) &&
        glm::all(glm::equal(previous.bounds.max_, current.bounds.max_));
    if (unchanged) {
      continue;
    }
    DDGISceneChangeRegion change{
        .kind = current.dynamic ? DDGISceneChangeKind::DynamicTransform
                                : DDGISceneChangeKind::StaticTransform,
        .sourceId = current.sourceId,
        .sourceVersion = sourceVersion,
    };
    if (previous.boundsKnown && current.boundsKnown) {
      change.worldBounds =
          BoundingBox{glm::min(previous.bounds.min_, current.bounds.min_),
                      glm::max(previous.bounds.max_, current.bounds.max_)};
      change.boundsKnown = true;
    }
    publishGeometryChange(change);
  }
}

void RayTracingScene::prepareDeformationChangeRegions(
    uint64_t sourceVersion) noexcept {
  for (const GeometryBoundsSnapshot &current : currentGeometryBounds_) {
    if (!current.dynamic) {
      continue;
    }
    const auto previous =
        std::ranges::lower_bound(committedGeometryBounds_, current.sourceId, {},
                                 &GeometryBoundsSnapshot::sourceId);
    DDGISceneChangeRegion change{
        .kind = DDGISceneChangeKind::Deformation,
        .sourceId = current.sourceId,
        .sourceVersion = sourceVersion,
    };
    if (previous != committedGeometryBounds_.end() &&
        previous->sourceId == current.sourceId && previous->boundsKnown &&
        current.boundsKnown) {
      change.worldBounds =
          BoundingBox{glm::min(previous->bounds.min_, current.bounds.min_),
                      glm::max(previous->bounds.max_, current.bounds.max_)};
      change.boundsKnown = true;
    }
    publishGeometryChange(change);
  }
}

void RayTracingScene::commitGeometryChanges(uint64_t frameIndex) noexcept {
  if (geometryChangeFrameIndex_ != frameIndex) {
    return;
  }
  committedGeometryBounds_ = currentGeometryBounds_;
  committedGeometryChanges_ = pendingGeometryChanges_;
  committedGeometryChangeCount_ = pendingGeometryChangeCount_;
  committedGeometryChangeOverflow_ = geometryChangeOverflow_;
  for (uint32_t index = 0u; index < committedGeometryChangeCount_; ++index) {
    committedGeometryChanges_[index].submissionSequence = frameIndex;
  }
  pendingGeometryChanges_ = {};
  pendingGeometryChangeCount_ = 0u;
  geometryChangeFrameIndex_ = UINT64_MAX;
  geometryChangeOverflow_ = false;
}

void RayTracingScene::abandonGeometryChanges(uint64_t frameIndex) noexcept {
  if (geometryChangeFrameIndex_ != frameIndex) {
    return;
  }
  pendingGeometryChanges_ = {};
  pendingGeometryChangeCount_ = 0u;
  geometryChangeFrameIndex_ = UINT64_MAX;
  geometryChangeOverflow_ = false;
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
  auto geometryResult =
      createTable(geometryTable_, bytesOf(std::span(geometryRecords_)),
                  "rt_geometry_table");
  if (geometryResult.hasError()) {
    return geometryResult;
  }
  return createTable(surfaceBounds_, bytesOf(std::span(surfaceBoundsRecords_)),
                     "rt_ddgi_surface_bounds");
}

Result<bool, std::string>
RayTracingScene::appendTopologyBuildPasses(FrameBuildContext &ctx) {
  decodeWork_.clear();
  dispatchBufferUses_.clear();
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
    const auto sourceVertices = ctx.graph.importBuffer(
        entry.sourceVertexBuffer, "rt_decode_source_vertices");
    const auto sourceDecode = ctx.graph.importBuffer(entry.sourceDecodeBuffer,
                                                     "rt_decode_source_table");
    const auto destinationPositions = ctx.graph.importBuffer(
        entry.decodedPositions.get(), "rt_decode_destination_positions");
    if (sourceVertices.hasError() || sourceDecode.hasError() ||
        destinationPositions.hasError()) {
      return Result<bool, std::string>::makeError(
          "RayTracingScene: failed to import static decode buffers");
    }
    appendUnique(dispatchBufferUses_, sourceVertices.value(),
                 RenderGraphAccessMode::Read);
    appendUnique(dispatchBufferUses_, sourceDecode.value(),
                 RenderGraphAccessMode::Read);
    appendUnique(dispatchBufferUses_, destinationPositions.value(),
                 RenderGraphAccessMode::Write);
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
        .debugLabel = "RT Decode Static Positions",
        .debugColor = 0xffbb66ffu,
    });
  }
  if (!decodeDispatches_.empty()) {
    auto decodePass = ctx.graph.addGraphicsPass(RenderGraphGraphicsPassDesc{
        .executionMode = RenderPassExecutionMode::ComputeOnly,
        .hasColorAttachment = false,
        .preDispatches = decodeDispatches_,
        .bufferUses = dispatchBufferUses_,
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
  pendingFrame_.topology = true;
  pendingFrame_.frameIndex = ctx.frame.frameIndex;
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
  dispatchBufferUses_.clear();
  size_t dispatchCount = 0u;
  for (const DynamicGeometryEntry &entry : dynamicGeometries_) {
    dispatchCount += entry.model->submeshes().size();
  }
  dynamicPushConstants_.reserve(dispatchCount);
  dynamicDispatches_.reserve(dispatchCount);
  const auto instanceMatrices = ctx.graph.importBuffer(
      animationData.instanceMatricesBuffer, "rt_dynamic_instance_matrices");
  if (instanceMatrices.hasError()) {
    return Result<bool, std::string>::makeError(instanceMatrices.error());
  }
  appendUnique(dispatchBufferUses_, instanceMatrices.value(),
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
    const auto sourceVertices = ctx.graph.importBuffer(
        entry.sourceVertexBuffer, "rt_dynamic_source_vertices");
    const auto destinationVertices = ctx.graph.importBuffer(
        entry.worldVertices.get(), "rt_dynamic_destination_vertices");
    if (sourceVertices.hasError() || destinationVertices.hasError()) {
      return Result<bool, std::string>::makeError(
          "RayTracingScene: failed to import dynamic vertex buffers");
    }
    appendUnique(dispatchBufferUses_, sourceVertices.value(),
                 RenderGraphAccessMode::Read);
    if (nuri::isValid(entry.sourceDecodeBuffer)) {
      const auto sourceDecode = ctx.graph.importBuffer(
          entry.sourceDecodeBuffer, "rt_dynamic_source_decode");
      if (sourceDecode.hasError()) {
        return Result<bool, std::string>::makeError(sourceDecode.error());
      }
      appendUnique(dispatchBufferUses_, sourceDecode.value(),
                   RenderGraphAccessMode::Read);
    }
    appendUnique(dispatchBufferUses_, destinationVertices.value(),
                 RenderGraphAccessMode::Write);
  }
  if (dynamicDispatches_.empty()) {
    return Result<bool, std::string>::makeResult(true);
  }
  auto pass = ctx.graph.addGraphicsPass(RenderGraphGraphicsPassDesc{
      .executionMode = RenderPassExecutionMode::ComputeOnly,
      .hasColorAttachment = false,
      .preDispatches = dynamicDispatches_,
      .bufferUses = dispatchBufferUses_,
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
  pendingFrame_.animationVersion = animationData.version;
  pendingFrame_.dynamic = true;
  pendingFrame_.frameIndex = ctx.frame.frameIndex;
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
  rebuildSurfaceBounds(database);
  beginGeometryChanges(ctx.frame.frameIndex, false);
  prepareTransformChangeRegions(ctx.frame.scene->transformVersion());
  upload = gpu_.updateBuffer(surfaceBounds_.get(),
                             bytesOf(std::span(surfaceBoundsRecords_)));
  if (upload.hasError()) {
    return upload;
  }
  pendingFrame_.transformVersion = ctx.frame.scene->transformVersion();
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
  const std::array<BufferHandle, 8u> inputs{
      instanceTable_.get(),
      geometryTable_.get(),
      surfaceBounds_.get(),
      material != nullptr ? (*material)[MaterialTableRegion::Header].buffer
                          : BufferHandle{},
      material != nullptr ? (*material)[MaterialTableRegion::Clearcoat].buffer
                          : BufferHandle{},
      material != nullptr ? (*material)[MaterialTableRegion::Sheen].buffer
                          : BufferHandle{},
      material != nullptr
          ? (*material)[MaterialTableRegion::Transmission].buffer
          : BufferHandle{},
      material != nullptr ? (*material)[MaterialTableRegion::Specular].buffer
                          : BufferHandle{},
  };
  if (!indirectReferencesDirty_ && inputs == indirectReferenceInputs_) {
    return;
  }
  indirectReferences_.clear();
  appendUnique(indirectReferences_, instanceTable_.get());
  appendUnique(indirectReferences_, geometryTable_.get());
  appendUnique(indirectReferences_, surfaceBounds_.get());
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
    for (const MaterialTableGpuRegion &region : material->regions) {
      appendUnique(indirectReferences_, region.buffer);
    }
  }
  indirectReferenceInputs_ = inputs;
  indirectReferencesDirty_ = false;
}

void RayTracingScene::publish(FrameBuildContext &ctx,
                              RenderGraphAccelerationStructureId graphTlas) {
  rebuildIndirectReferences(ctx.shared);
  const bool publishesPendingChanges =
      geometryChangeFrameIndex_ == ctx.frame.frameIndex;
  const std::span<const DDGISceneChangeRegion> geometryChanges =
      publishesPendingChanges
          ? std::span<const DDGISceneChangeRegion>(
                pendingGeometryChanges_.data(), pendingGeometryChangeCount_)
          : std::span<const DDGISceneChangeRegion>(
                committedGeometryChanges_.data(),
                committedGeometryChangeCount_);
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
      .materialTable = materials != nullptr
                           ? (*materials)[MaterialTableRegion::Header].buffer
                           : BufferHandle{},
      .surfaceBounds = surfaceBounds_.get(),
      .instanceTableAddress = gpu_.getBufferDeviceAddress(instanceTable_.get()),
      .geometryTableAddress = gpu_.getBufferDeviceAddress(geometryTable_.get()),
      .materialTableAddress =
          materials != nullptr
              ? (*materials)[MaterialTableRegion::Header].address
              : 0u,
      .surfaceBoundsAddress = gpu_.getBufferDeviceAddress(surfaceBounds_.get()),
      .indirectSubmissionReferences = indirectReferences_,
      .indirectSubmissionTextureReferences =
          ctx.shared.sceneDrawDatabase->rayTracingMaterialTextures(),
      .sceneId = sceneId_,
      .topologyVersion = topologyVersion_,
      .transformVersion = pendingFrame_.transform
                              ? pendingFrame_.transformVersion
                              : transformVersion_,
      .deformationVersion = pendingFrame_.dynamic
                                ? pendingFrame_.animationVersion
                                : deformationVersion_,
      .geometryMutationVersion = geometryMutationVersion_,
      .instanceCount = static_cast<uint32_t>(instanceRecords_.size()),
      .geometryCount = static_cast<uint32_t>(geometryRecords_.size()),
      .staticSurfaceBoundsCount = staticSurfaceBoundsCount_,
      .dynamicSurfaceBoundsCount = dynamicSurfaceBoundsCount_,
      .staticCoverageBounds = currentStaticCoverageBounds_,
      .geometryChangeRegions = geometryChanges,
      .completedBlasBuilds =
          ready_ ? static_cast<uint32_t>(staticGeometries_.size() +
                                         dynamicGeometries_.size())
                 : 0u,
      .totalBlasBuilds = static_cast<uint32_t>(staticGeometries_.size() +
                                               dynamicGeometries_.size()),
      .readiness = readiness,
      .staticSurfaceBoundsAvailable = staticSurfaceBoundsAvailable_,
      .dynamicSurfaceBoundsAvailable = dynamicSurfaceBoundsAvailable_,
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
  metrics.staticSurfaceBounds = staticSurfaceBoundsCount_;
  metrics.dynamicSurfaceBounds = dynamicSurfaceBoundsCount_;
  metrics.surfaceBoundsFallbacks = (!staticSurfaceBoundsAvailable_ ? 1u : 0u) +
                                   (!dynamicSurfaceBoundsAvailable_ ? 1u : 0u);
  metrics.geometryChangeRegions = static_cast<uint32_t>(geometryChanges.size());
  metrics.geometryChangeOverflows =
      (publishesPendingChanges ? geometryChangeOverflow_
                               : committedGeometryChangeOverflow_)
          ? 1u
          : 0u;
  metrics.decodeDispatches = static_cast<uint32_t>(decodeWork_.size());
  metrics.blasBuilds = pendingFrame_.topology
                           ? static_cast<uint32_t>(staticGeometries_.size() +
                                                   dynamicGeometries_.size())
                           : 0u;
  metrics.dynamicVertexDispatches =
      pendingFrame_.topology || pendingFrame_.dynamic
          ? static_cast<uint32_t>(dynamicDispatches_.size())
          : 0u;
  metrics.tlasBuilds = pendingFrame_.topology ? 1u : 0u;
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
  uint64_t scheduledScratchBytes = 0u;
  const auto accumulateFacts =
      [this, &scheduledScratchBytes](AccelerationStructureHandle handle,
                                     bool bottomLevel, bool buildScheduled,
                                     bool updateScheduled,
                                     RayTracingSceneFrameMetrics &out) {
        auto facts = gpu_.getAccelerationStructureFacts(handle);
        if (facts.hasError()) {
          return;
        }
        if (bottomLevel) {
          out.blasAllocationBytes += facts.value().allocationBytes;
        } else {
          out.tlasAllocationBytes += facts.value().allocationBytes;
        }
        if (buildScheduled) {
          scheduledScratchBytes += facts.value().buildScratchBytes;
        } else if (updateScheduled) {
          scheduledScratchBytes += facts.value().updateScratchBytes;
        }
      };
  for (const StaticGeometryEntry &entry : staticGeometries_) {
    accumulateFacts(entry.blas.get(), true, pendingFrame_.topology, false,
                    metrics);
  }
  for (const DynamicGeometryEntry &entry : dynamicGeometries_) {
    accumulateFacts(entry.blas.get(), true, pendingFrame_.topology,
                    pendingFrame_.dynamic, metrics);
  }
  if (tlas_.valid()) {
    accumulateFacts(tlas_.get(), false, pendingFrame_.topology,
                    pendingFrame_.dynamic || pendingFrame_.transform, metrics);
  }
  asScratchHighWaterBytes_ =
      std::max(asScratchHighWaterBytes_, scheduledScratchBytes);
  metrics.asScratchHighWaterBytes = asScratchHighWaterBytes_;
  metrics.asScratchCurrentBytes = scheduledScratchBytes;
  metrics.directBindingPoolHighWater =
      gpu_.getRayTracingBackendTelemetry().directBindingPoolHighWater;
  metrics.indirectSubmissionReferences =
      static_cast<uint32_t>(indirectReferences_.size());
  metrics.indirectTextureReferences = static_cast<uint32_t>(
      ctx.shared.sceneDrawDatabase->rayTracingMaterialTextures().size());
  metrics.graphAccelerationStructureDependencies =
      nuri::isValid(graphTlas) ? 1u : 0u;
  metrics.noAsWorkFrame =
      metrics.decodeDispatches == 0u && metrics.blasBuilds == 0u &&
              metrics.tlasBuilds == 0u && metrics.tlasUpdates == 0u &&
              metrics.dynamicBlasUpdates == 0u &&
              metrics.dynamicVertexDispatches == 0u
          ? 1u
          : 0u;
  metrics.tableBytes =
      instanceRecords_.size() * sizeof(RtInstanceGpuData) +
      geometryRecords_.size() * sizeof(RtGeometryGpuData) +
      surfaceBoundsRecords_.size() * sizeof(RtSurfaceBoundsGpuData);
  if (hasGpuTimingScope(ctx.frame.gpuTiming, GpuTimingScope::RayTracingScene)) {
    metrics.gpuTimeMs =
        ctx.frame.gpuTiming[GpuTimingScope::RayTracingScene].timeMs;
    metrics.gpuTimingSourceFrameIndex =
        ctx.frame.gpuTiming[GpuTimingScope::RayTracingScene].sourceFrameIndex;
    metrics.gpuTimingAvailable = 1u;
  }
  if (hasGpuTimingScope(ctx.frame.gpuTiming, GpuTimingScope::RayTracingBLAS)) {
    metrics.blasGpuTimeMs =
        ctx.frame.gpuTiming[GpuTimingScope::RayTracingBLAS].timeMs;
    metrics.blasGpuTimingAvailable = 1u;
  }
  if (hasGpuTimingScope(ctx.frame.gpuTiming, GpuTimingScope::RayTracingTLAS)) {
    metrics.tlasGpuTimeMs =
        ctx.frame.gpuTiming[GpuTimingScope::RayTracingTLAS].timeMs;
    metrics.tlasGpuTimingAvailable = 1u;
  }
}

Result<bool, std::string> RayTracingScene::prepare(FrameBuildContext &ctx) {
  NURI_PROFILER_FUNCTION();
  ctx.shared.rayTracingScene.reset();
  ctx.frame.metrics.rayTracingScene = {};
  struct PrepareCpuTimer {
    RayTracingSceneFrameMetrics &metrics;
    std::chrono::steady_clock::time_point start =
        std::chrono::steady_clock::now();
    ~PrepareCpuTimer() {
      metrics.prepareCpuTimeMs = std::chrono::duration<float, std::milli>(
                                     std::chrono::steady_clock::now() - start)
                                     .count();
    }
  } prepareCpuTimer{ctx.frame.metrics.rayTracingScene};
  ctx.frame.metrics.rayTracingScene.consumedRebuildEpoch =
      consumedRebuildEpoch_;
  pendingFrame_.topology = false;
  const RenderSettings::DDGISettings &settings = ctx.frame.settings.ddgi;
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
    const auto start = std::chrono::steady_clock::now();
    auto rebuild = rebuildStaticScene(ctx, *ctx.shared.sceneDrawDatabase);
    ctx.frame.metrics.rayTracingScene.topologyPrepareCpuTimeMs =
        std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - start)
            .count();
    if (rebuild.hasError()) {
      failed_ = true;
      return rebuild;
    }
    pendingFrame_.rebuildEpoch = std::max(
        consumedRebuildEpoch_, settings.requestedEpochs.rebuildRayTracingScene);
    pendingFrame_.frameIndex = ctx.frame.frameIndex;
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
    const auto start = std::chrono::steady_clock::now();
    auto transforms = updateTransforms(ctx, *ctx.shared.sceneDrawDatabase);
    ctx.frame.metrics.rayTracingScene.transformPrepareCpuTimeMs =
        std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - start)
            .count();
    if (transforms.hasError()) {
      return transforms;
    }
    pendingFrame_.transform = true;
    pendingFrame_.frameIndex = ctx.frame.frameIndex;
  }
  const bool dynamicChanged = ready_ && !dynamicGeometries_.empty() &&
                              matchingAnimationData &&
                              animationVersion_ != animationData->version;
  if (dynamicChanged) {
    const auto start = std::chrono::steady_clock::now();
    rebuildSurfaceBounds(*ctx.shared.sceneDrawDatabase);
    beginGeometryChanges(ctx.frame.frameIndex, transformsChanged);
    prepareDeformationChangeRegions(animationData->version);
    auto uploadBounds = gpu_.updateBuffer(
        surfaceBounds_.get(), bytesOf(std::span(surfaceBoundsRecords_)));
    if (uploadBounds.hasError()) {
      return uploadBounds;
    }
    auto result = appendDynamicUpdatePasses(ctx, *animationData);
    ctx.frame.metrics.rayTracingScene.deformationPrepareCpuTimeMs =
        std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - start)
            .count();
    return result;
  }
  if (transformsChanged) {
    const auto start = std::chrono::steady_clock::now();
    auto result = appendTlasUpdatePass(ctx);
    ctx.frame.metrics.rayTracingScene.tlasPrepareCpuTimeMs =
        std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - start)
            .count();
    return result;
  }
  publish(ctx, graphTlas);
  return Result<bool, std::string>::makeResult(true);
}

void RayTracingScene::onFrameSubmitted(
    const RenderFrameContext &frame) noexcept {
  commitGeometryChanges(frame.frameIndex);
  if (pendingFrame_.frameIndex != frame.frameIndex) {
    return;
  }
  consumedRebuildEpoch_ =
      std::max(consumedRebuildEpoch_, pendingFrame_.rebuildEpoch);
  pendingFrame_.rebuildEpoch = consumedRebuildEpoch_;
  if (frame.scene != nullptr) {
    (void)frame.scene->stageDDGIStaticCoverageBounds(
        currentStaticCoverageBounds_);
  }
  if (pendingFrame_.topology) {
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
    pendingFrame_.topology = false;
  }
  if (pendingFrame_.dynamic) {
    animationVersion_ = pendingFrame_.animationVersion;
    deformationVersion_ = pendingFrame_.animationVersion;
    pendingFrame_.dynamic = false;
  }
  if (pendingFrame_.transform) {
    transformVersion_ = pendingFrame_.transformVersion;
    pendingFrame_.transform = false;
  }
  pendingFrame_.frameIndex = UINT64_MAX;
}

void RayTracingScene::onFrameAbandoned(
    const RenderFrameContext &frame) noexcept {
  abandonGeometryChanges(frame.frameIndex);
  if (pendingFrame_.topology && pendingFrame_.frameIndex == frame.frameIndex) {
    failed_ = true;
    pendingFrame_.topology = false;
  }
  if (pendingFrame_.dynamic && pendingFrame_.frameIndex == frame.frameIndex) {
    pendingFrame_.dynamic = false;
    pendingFrame_.animationVersion = animationVersion_;
  }
  if (pendingFrame_.transform && pendingFrame_.frameIndex == frame.frameIndex) {
    pendingFrame_.transform = false;
    pendingFrame_.transformVersion = transformVersion_;
  }
  if (pendingFrame_.frameIndex == frame.frameIndex) {
    pendingFrame_.rebuildEpoch = consumedRebuildEpoch_;
    pendingFrame_.frameIndex = UINT64_MAX;
  }
}

} // namespace nuri
