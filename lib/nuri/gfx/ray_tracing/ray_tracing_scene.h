#pragma once

#include "nuri/core/runtime_config.h"
#include "nuri/gfx/owned_gpu_resource.h"
#include "nuri/gfx/pipeline/render_pipeline.h"
#include "nuri/gfx/ray_tracing/ray_tracing_types.h"
#include "nuri/gfx/shader.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <memory_resource>
#include <span>
#include <string>
#include <vector>

namespace nuri {

class GPUDevice;
class Model;
class SceneDrawDatabase;
struct AnimationSceneFrameData;

class NURI_API RayTracingScene final {
public:
  RayTracingScene(
      GPUDevice &gpu, RuntimeDDGIShaderConfig config,
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  ~RayTracingScene();
  RayTracingScene(const RayTracingScene &) = delete;
  RayTracingScene &operator=(const RayTracingScene &) = delete;
  RayTracingScene(RayTracingScene &&) = delete;
  RayTracingScene &operator=(RayTracingScene &&) = delete;

  [[nodiscard]] Result<bool, std::string> prepare(FrameBuildContext &ctx);
  void onFrameSubmitted(const RenderFrameContext &frame) noexcept;
  void onFrameAbandoned(const RenderFrameContext &frame) noexcept;

private:
  struct DecodePushConstants {
    uint64_t sourceVertices = 0u;
    uint64_t sourceDecode = 0u;
    uint64_t destinationPositions = 0u;
    uint32_t vertexOffset = 0u;
    uint32_t vertexCount = 0u;
    uint32_t vertexDecodeIndex = 0u;
    uint32_t packedVertexFormat = 0u;
  };
  static_assert(sizeof(DecodePushConstants) == 40u);

  struct DecodeWork {
    DecodePushConstants constants{};
    std::array<BufferHandle, 3u> dependencies{};
    std::array<RenderGraphAccessMode, 3u> accessModes{
        RenderGraphAccessMode::Read, RenderGraphAccessMode::Read,
        RenderGraphAccessMode::Write};
  };

  struct DynamicVertexPushConstants {
    uint64_t sourceVertices = 0u;
    uint64_t sourceDecode = 0u;
    uint64_t instanceData = 0u;
    uint64_t destinationVertices = 0u;
    uint32_t renderableIndex = 0u;
    uint32_t vertexOffset = 0u;
    uint32_t vertexCount = 0u;
    uint32_t vertexDecodeIndex = 0u;
    uint32_t packedVertexFormat = 0u;
  };
  static_assert(sizeof(DynamicVertexPushConstants) == 56u);

  struct StaticGeometryEntry {
    explicit StaticGeometryEntry(
        std::pmr::memory_resource *memory = std::pmr::get_default_resource())
        : geometries(memory) {}
    GeometryAllocationHandle geometry{};
    const Model *model = nullptr;
    BufferHandle sourceVertexBuffer{};
    BufferHandle sourceDecodeBuffer{};
    BufferHandle sourceIndexBuffer{};
    uint64_t sourceIndexBaseOffset = 0u;
    uint64_t decodedAddress = 0u;
    uint32_t vertexCount = 0u;
    IndexFormat indexFormat = IndexFormat::U32;
    OwnedBufferHandle decodedPositions{};
    OwnedAccelerationStructure blas{};
    std::pmr::vector<AccelerationStructureTriangleGeometryDesc> geometries;
  };

  struct DynamicGeometryEntry {
    explicit DynamicGeometryEntry(
        std::pmr::memory_resource *memory = std::pmr::get_default_resource())
        : geometries(memory) {}
    const Model *model = nullptr;
    uint32_t renderableIndex = 0u;
    BufferHandle sourceVertexBuffer{};
    BufferHandle sourceDecodeBuffer{};
    BufferHandle sourceIndexBuffer{};
    uint64_t sourceVertexByteOffset = 0u;
    uint64_t sourceIndexBaseOffset = 0u;
    uint32_t vertexCount = 0u;
    uint32_t sourcePackedVertexFormat = 0u;
    IndexFormat indexFormat = IndexFormat::U32;
    OwnedBufferHandle worldVertices{};
    OwnedAccelerationStructure blas{};
    std::pmr::vector<AccelerationStructureTriangleGeometryDesc> geometries;
  };

  struct GeometryBoundsSnapshot {
    BoundingBox bounds{};
    uint64_t sourceId = 0u;
    bool dynamic = false;
    bool deforming = false;
    bool boundsKnown = false;
  };

  [[nodiscard]] Result<bool, std::string> initialize();
  [[nodiscard]] Result<bool, std::string>
  rebuildStaticScene(FrameBuildContext &ctx, const SceneDrawDatabase &database);
  [[nodiscard]] Result<bool, std::string>
  updateTransforms(FrameBuildContext &ctx, const SceneDrawDatabase &database);
  [[nodiscard]] Result<bool, std::string>
  appendTopologyBuildPasses(FrameBuildContext &ctx);
  [[nodiscard]] Result<bool, std::string>
  appendDynamicVertexPass(FrameBuildContext &ctx,
                          const AnimationSceneFrameData &animationData);
  [[nodiscard]] Result<bool, std::string>
  appendDynamicUpdatePasses(FrameBuildContext &ctx,
                            const AnimationSceneFrameData &animationData);
  [[nodiscard]] Result<bool, std::string>
  appendTlasUpdatePass(FrameBuildContext &ctx);
  [[nodiscard]] Result<bool, std::string> uploadTables();
  void rebuildSurfaceBounds(const SceneDrawDatabase &database);
  void prepareTopologyChangeRegions(uint64_t sourceVersion,
                                    bool sceneChanged) noexcept;
  void prepareTransformChangeRegions(uint64_t sourceVersion) noexcept;
  void prepareDeformationChangeRegions(uint64_t sourceVersion) noexcept;
  void publishGeometryChange(DDGISceneChangeRegion change) noexcept;
  void beginGeometryChanges(uint64_t frameIndex, bool append) noexcept;
  void commitGeometryChanges(uint64_t frameIndex) noexcept;
  void abandonGeometryChanges(uint64_t frameIndex) noexcept;
  void rebuildIndirectReferences(const FrameSharedResources &shared);
  void clearSceneResources(bool clearChangeTracking = true) noexcept;
  void pollCompletion() noexcept;
  void publish(FrameBuildContext &ctx,
               RenderGraphAccelerationStructureId graphTlas = {});

  GPUDevice &gpu_;
  RuntimeDDGIShaderConfig config_{};
  std::pmr::memory_resource *memory_ = std::pmr::get_default_resource();
  std::unique_ptr<Shader> decodeShader_{};
  std::unique_ptr<Shader> dynamicVertexShader_{};
  OwnedComputePipelineHandle decodePipeline_{};
  OwnedComputePipelineHandle dynamicVertexPipeline_{};
  std::string initializationError_{};
  std::pmr::vector<StaticGeometryEntry> staticGeometries_;
  std::pmr::vector<DynamicGeometryEntry> dynamicGeometries_;
  std::pmr::vector<RtInstanceGpuData> instanceRecords_;
  std::pmr::vector<RtGeometryGpuData> geometryRecords_;
  std::pmr::vector<RtSurfaceBoundsGpuData> surfaceBoundsRecords_;
  std::pmr::vector<GeometryBoundsSnapshot> currentGeometryBounds_;
  std::pmr::vector<GeometryBoundsSnapshot> committedGeometryBounds_;
  DDGISceneCoverageBounds currentStaticCoverageBounds_{};
  std::array<DDGISceneChangeRegion, kMaxDDGIGeometryChangeRegions>
      pendingGeometryChanges_{};
  std::array<DDGISceneChangeRegion, kMaxDDGIGeometryChangeRegions>
      committedGeometryChanges_{};
  std::pmr::vector<AccelerationStructureInstanceDesc> tlasInstances_;
  std::pmr::vector<DecodeWork> decodeWork_;
  std::pmr::vector<ComputeDispatchItem> decodeDispatches_;
  std::pmr::vector<DynamicVertexPushConstants> dynamicPushConstants_;
  std::pmr::vector<ComputeDispatchItem> dynamicDispatches_;
  std::pmr::vector<BufferHandle> dependencyBuffers_;
  std::pmr::vector<RenderGraphAccessMode> dependencyBufferModes_;
  std::pmr::vector<AccelerationStructureBuildItem> blasBuildItems_;
  std::pmr::vector<RenderGraphBufferUse> blasBufferUses_;
  std::pmr::vector<RenderGraphAccelerationStructureUse> blasUses_;
  std::pmr::vector<RenderGraphAccelerationStructureUse> tlasUses_;
  std::pmr::vector<BufferHandle> indirectReferences_;
  std::array<BufferHandle, 8u> indirectReferenceInputs_{};
  OwnedBufferHandle instanceTable_{};
  OwnedBufferHandle geometryTable_{};
  OwnedBufferHandle surfaceBounds_{};
  OwnedAccelerationStructure tlas_{};
  SubmissionHandle buildCompletion_{};
  uint64_t sceneId_ = 0u;
  uint64_t topologyVersion_ = 0u;
  uint64_t transformVersion_ = 0u;
  uint64_t pendingTransformVersion_ = 0u;
  uint64_t deformationVersion_ = 0u;
  uint64_t geometryMutationVersion_ = 0u;
  uint64_t asScratchHighWaterBytes_ = 0u;
  uint64_t animationVersion_ = 0u;
  uint64_t pendingAnimationVersion_ = 0u;
  uint64_t consumedRebuildEpoch_ = 0u;
  uint64_t pendingRebuildEpoch_ = 0u;
  uint64_t scheduledFrameIndex_ = UINT64_MAX;
  uint32_t excludedDynamicInstances_ = 0u;
  uint32_t staticInstanceCount_ = 0u;
  uint32_t staticSurfaceBoundsCount_ = 0u;
  uint32_t dynamicSurfaceBoundsCount_ = 0u;
  uint32_t pendingGeometryChangeCount_ = 0u;
  uint32_t committedGeometryChangeCount_ = 0u;
  uint64_t geometryChangeFrameIndex_ = UINT64_MAX;
  bool geometryChangeOverflow_ = false;
  bool committedGeometryChangeOverflow_ = false;
  bool topologyBuildScheduled_ = false;
  bool dynamicUpdateScheduled_ = false;
  bool transformUpdateScheduled_ = false;
  bool indirectReferencesDirty_ = true;
  bool ready_ = false;
  bool failed_ = false;
  bool staticSurfaceBoundsAvailable_ = false;
  bool dynamicSurfaceBoundsAvailable_ = false;
};

} // namespace nuri
