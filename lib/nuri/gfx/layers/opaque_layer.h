#pragma once

#include "nuri/core/layer.h"
#include "nuri/core/runtime_config.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/pipeline.h"
#include "nuri/gfx/shader.h"
#include "nuri/resources/cpu/mesh_data.h"
#include "nuri/resources/gpu/buffer.h"
#include "nuri/scene/render_scene.h"

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <memory_resource>
#include <utility>
#include <vector>

#include <glm/glm.hpp>

namespace nuri {

using OpaqueLayerConfig = RuntimeOpaqueShaderConfig;

class NURI_API OpaqueLayer final : public Layer {
public:
  explicit OpaqueLayer(
      GPUDevice &gpu, OpaqueLayerConfig config,
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  ~OpaqueLayer() override;

  OpaqueLayer(const OpaqueLayer &) = delete;
  OpaqueLayer &operator=(const OpaqueLayer &) = delete;
  OpaqueLayer(OpaqueLayer &&) = delete;
  OpaqueLayer &operator=(OpaqueLayer &&) = delete;

  static std::unique_ptr<OpaqueLayer>
  create(GPUDevice &gpu, OpaqueLayerConfig config,
         std::pmr::memory_resource *memory = std::pmr::get_default_resource()) {
    return std::make_unique<OpaqueLayer>(gpu, std::move(config), memory);
  }

  void onAttach() override;
  void onDetach() override;
  void onResize(int32_t width, int32_t height) override;
  Result<bool, std::string> buildRenderPasses(RenderFrameContext &frame,
                                              RenderPassList &out) override;

private:
  struct FrameData {
    glm::mat4 view{1.0f};
    glm::mat4 proj{1.0f};
    glm::vec4 cameraPos{0.0f, 0.0f, 0.0f, 1.0f};
    uint32_t cubemapTexId = 0;
    uint32_t hasCubemap = 0;
    uint32_t _padding0 = 0;
    uint32_t _padding1 = 0;
  };

  struct PushConstants {
    uint64_t frameDataAddress = 0;
    uint64_t vertexBufferAddress = 0;
    uint64_t instanceMatricesAddress = 0;
    uint64_t instanceRemapAddress = 0;
    uint64_t instanceMetaAddress = 0;
    uint64_t instanceCentersPhaseAddress = 0;
    uint64_t instanceBaseMatricesAddress = 0;
    uint32_t instanceCount = 0;
    float timeSeconds = 0.0f;
    float tessNearDistance = 1.0f;
    float tessFarDistance = 8.0f;
    float tessMinFactor = 1.0f;
    float tessMaxFactor = 6.0f;
    uint32_t debugVisualizationMode = 0;
  };
  static_assert(sizeof(PushConstants) <= 128,
                "OpaqueLayer::PushConstants exceeds Vulkan minimum guarantee");

  struct RenderableTemplate {
    const OpaqueRenderable *renderable = nullptr;
  };

  struct MeshDrawTemplate {
    const OpaqueRenderable *renderable = nullptr;
    const Submesh *submesh = nullptr;
    uint32_t instanceIndex = 0;
    GeometryAllocationHandle geometryHandle{};
    BufferHandle indexBuffer{};
    uint64_t indexBufferOffset = 0;
    uint64_t vertexBufferAddress = 0;
  };

  struct TessCandidate {
    float distanceSq = 0.0f;
    uint32_t instanceId = 0;
  };

  struct DynamicBufferSlot {
    std::unique_ptr<Buffer> buffer;
    size_t capacityBytes = 0;
  };

  Result<bool, std::string> ensureInitialized();
  Result<bool, std::string> recreateDepthTexture();
  Result<bool, std::string> ensureFrameDataBufferCapacity(size_t requiredBytes);
  Result<bool, std::string>
  ensureCentersPhaseBufferCapacity(size_t requiredBytes);
  Result<bool, std::string>
  ensureInstanceBaseMatricesBufferCapacity(size_t requiredBytes);
  Result<bool, std::string>
  ensureInstanceMetaBufferCapacity(size_t requiredBytes);
  Result<bool, std::string> ensureRingBufferCount(uint32_t requiredCount);
  Result<bool, std::string>
  ensureInstanceMatricesRingCapacity(size_t requiredBytes);
  Result<bool, std::string>
  ensureInstanceRemapRingCapacity(size_t requiredBytes);
  Result<bool, std::string> rebuildSceneCache(const RenderScene &scene);
  Result<bool, std::string> createShaders();
  Result<bool, std::string> createPipelines();
  Result<bool, std::string> ensureWireframePipeline();
  Result<bool, std::string> ensureTessWireframePipeline();
  Result<bool, std::string> ensureGsOverlayPipeline();
  Result<bool, std::string> ensureGsTessOverlayPipeline();
  void resetOverlayPipelineState();
  void invalidateAutoLodCache();
  void updateFastAutoLodCache(
      const Submesh *submesh, const glm::vec3 &cameraPosition,
      const std::array<float, 3> &sortedLodThresholds,
      const std::array<size_t, Submesh::kMaxLodCount> &bucketCounts,
      size_t remapCount, size_t instanceCount);
  void destroyDepthTexture();
  void destroyBuffers();

  GPUDevice &gpu_;
  OpaqueLayerConfig config_{};
  std::unique_ptr<Shader> meshShader_;
  std::unique_ptr<Shader> meshTessShader_;
  std::unique_ptr<Shader> meshDebugOverlayShader_;
  std::unique_ptr<Shader> computeShader_;
  std::unique_ptr<Pipeline> meshPipeline_;
  std::unique_ptr<Pipeline> computePipeline_;
  std::unique_ptr<Buffer> frameDataBuffer_;
  std::unique_ptr<Buffer> instanceCentersPhaseBuffer_;
  std::unique_ptr<Buffer> instanceBaseMatricesBuffer_;
  std::unique_ptr<Buffer> instanceMetaBuffer_;
  std::vector<DynamicBufferSlot> instanceMatricesRing_;
  std::vector<DynamicBufferSlot> instanceRemapRing_;
  TextureHandle depthTexture_{};

  ShaderHandle meshVertexShader_{};
  ShaderHandle meshTessVertexShader_{};
  ShaderHandle meshTessControlShader_{};
  ShaderHandle meshTessEvalShader_{};
  ShaderHandle meshFragmentShader_{};
  ShaderHandle meshDebugOverlayGeometryShader_{};
  ShaderHandle meshDebugOverlayFragmentShader_{};
  ShaderHandle computeShaderHandle_{};
  RenderPipelineHandle meshFillPipelineHandle_{};
  RenderPipelineHandle meshTessPipelineHandle_{};
  RenderPipelineHandle meshGsOverlayPipelineHandle_{};
  RenderPipelineHandle meshGsTessOverlayPipelineHandle_{};
  RenderPipelineHandle meshWireframePipelineHandle_{};
  RenderPipelineHandle meshTessWireframePipelineHandle_{};
  ComputePipelineHandle computePipelineHandle_{};

  size_t frameDataBufferCapacityBytes_ = 0;
  size_t instanceCentersPhaseBufferCapacityBytes_ = 0;
  size_t instanceBaseMatricesBufferCapacityBytes_ = 0;
  size_t instanceMetaBufferCapacityBytes_ = 0;
  bool initialized_ = false;
  bool tessellationUnsupported_ = false;
  bool wireframePipelineInitialized_ = false;
  bool wireframePipelineUnsupported_ = false;
  bool tessWireframePipelineInitialized_ = false;
  bool tessWireframePipelineUnsupported_ = false;
  bool gsOverlayPipelineInitialized_ = false;
  bool gsOverlayPipelineUnsupported_ = false;
  bool gsTessOverlayPipelineInitialized_ = false;
  bool gsTessOverlayPipelineUnsupported_ = false;
  bool loggedWireframeFallbackUnsupported_ = false;
  bool loggedTessWireframeFallbackUnsupported_ = false;
  bool loggedGsOverlayUnsupported_ = false;
  bool loggedGsTessOverlayUnsupported_ = false;

  const RenderScene *cachedScene_ = nullptr;
  uint64_t cachedTopologyVersion_ = std::numeric_limits<uint64_t>::max();
  uint64_t cachedTransformVersion_ = std::numeric_limits<uint64_t>::max();
  bool instanceStaticBuffersDirty_ = true;
  bool uniformSingleSubmeshPath_ = false;

  struct AutoLodCache {
    bool valid = false;
    glm::vec3 cameraPos{0.0f};
    std::array<float, 3> thresholds = {0.0f, 0.0f, 0.0f};
    std::array<size_t, Submesh::kMaxLodCount> bucketCounts{};
    size_t remapCount = 0;
    size_t instanceCount = 0;
    const Submesh *submesh = nullptr;
  };
  AutoLodCache autoLodCache_{};

  std::pmr::vector<RenderableTemplate> renderableTemplates_;
  std::pmr::vector<MeshDrawTemplate> meshDrawTemplates_;
  std::pmr::vector<uint32_t> templateBatchIndices_;
  std::pmr::vector<size_t> batchWriteOffsets_;
  std::pmr::vector<glm::vec4> instanceCentersPhase_;
  std::pmr::vector<glm::mat4> instanceBaseMatrices_;
  std::pmr::vector<glm::vec4> instanceLodCentersInvRadiusSq_;
  std::pmr::vector<uint32_t> instanceAlbedoTexIds_;
  std::pmr::vector<uint32_t> instanceAutoLodLevels_;
  std::pmr::vector<uint8_t> instanceTessSelection_;
  std::pmr::vector<TessCandidate> tessCandidates_;
  std::pmr::vector<uint32_t> instanceRemap_;
  std::pmr::vector<PushConstants> drawPushConstants_;
  std::pmr::vector<DrawItem> drawItems_;
  std::pmr::vector<DrawItem> overlayDrawItems_;
  std::pmr::vector<DrawItem> passDrawItems_;
  std::pmr::vector<ComputeDispatchItem> preDispatches_;
  std::pmr::vector<BufferHandle> passDependencyBuffers_;
  std::pmr::vector<BufferHandle> dispatchDependencyBuffers_;
  FrameData frameData_{};
  PushConstants computePushConstants_{};
  DrawItem baseMeshFillDraw_{};
  DrawItem baseMeshWireframeDraw_{};
  uint64_t statsLogFrameCounter_ = 0;
};

} // namespace nuri
