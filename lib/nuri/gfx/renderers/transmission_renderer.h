#pragma once

#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <memory_resource>
#include <utility>
#include <vector>

#include <glm/glm.hpp>

#include "nuri/core/runtime_config.h"
#include "nuri/defines.h"
#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/render_graph/render_graph.h"
#include "nuri/gfx/renderers/detail/instance_data.h"
#include "nuri/resources/cpu/mesh_data.h"
#include "nuri/resources/gpu/buffer.h"
#include "nuri/resources/gpu/material.h"
#include "nuri/resources/gpu/model.h"
#include "nuri/resources/gpu/texture.h"
#include "nuri/scene/render_scene.h"

namespace nuri {

using TransmissionRendererConfig = RuntimeOpaqueShaderConfig;

class ResourceManager;
class Shader;

class NURI_API TransmissionRenderer {
public:
  explicit TransmissionRenderer(
      GPUDevice &gpu, const TransmissionRendererConfig &config,
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  ~TransmissionRenderer();

  TransmissionRenderer(const TransmissionRenderer &) = delete;
  TransmissionRenderer &operator=(const TransmissionRenderer &) = delete;
  TransmissionRenderer(TransmissionRenderer &&) = delete;
  TransmissionRenderer &operator=(TransmissionRenderer &&) = delete;

  void onAttach();
  void onDetach();
  void publishFrameData(RenderFrameContext &frame);
  [[nodiscard]] bool hasPreparedTransmissionMainPass() const noexcept;
  Result<bool, std::string>
  prepareTransmissionPasses(RenderFrameContext &frame);
  Result<bool, std::string>
  appendTransmissionMainPass(RenderFrameContext &frame,
                             RenderGraphBuilder &graph);

private:
  using FrameData = ForwardSceneFrameData;
  static_assert(sizeof(FrameData) == 336,
                "TransmissionRenderer::FrameData must match shader layout");

  struct MeshPushConstants {
    uint64_t frameDataAddress = 0;
    uint64_t vertexBufferAddress = 0;
    uint64_t vertexDecodeBufferAddress = 0;
    uint64_t instanceMatricesAddress = 0;
    uint64_t instanceRemapAddress = 0;
    uint64_t instanceCentersPhaseAddress = 0;
    uint64_t instanceBaseMatricesAddress = 0;
    uint32_t instanceCount = 0;
    uint32_t materialIndex = 0;
    uint32_t vertexDecodeIndex = 0;
    uint32_t packedVertexFormat = 0;
    float timeSeconds = 0.0f;
    // Transmission fragment shading aliases these tessellation slots as
    // per-draw model scale to keep the push-constant layout shader-compatible.
    float tessNearDistance = 1.0f;
    float tessFarDistance = 8.0f;
    float tessMinFactor = 1.0f;
    float tessMaxFactor = 1.0f;
    uint32_t debugVisualizationMode = 0;

    void setTransmissionScale(const glm::vec3 &scale) noexcept {
      tessNearDistance = scale.x;
      tessFarDistance = scale.y;
      tessMinFactor = scale.z;
    }
  };
  static_assert(
      sizeof(MeshPushConstants) <= 128,
      "TransmissionRenderer::MeshPushConstants exceeds Vulkan guarantee");

  struct MeshDrawTemplate {
    const Renderable *renderable = nullptr;
    const Submesh *submesh = nullptr;
    uint32_t submeshIndex = 0;
    BufferHandle indexBuffer{};
    uint64_t indexBufferOffset = 0;
    // base* handles are unresolved source buffers captured from geometry;
    // vertexBuffer/vertexBufferByteOffset describe the intended binding/view.
    // GPU virtual addresses are resolved later during pass preparation after
    // resource binding/animation overrides are known.
    BufferHandle baseVertexBuffer{};
    uint64_t vertexBufferByteOffset = 0;
    BufferHandle vertexBuffer{};
    BufferHandle baseVertexDecodeBuffer{};
    uint64_t vertexBufferAddress = 0;
    uint64_t vertexDecodeBufferAddress = 0;
    uint32_t vertexDecodeIndex = 0;
    uint32_t packedVertexFormat = 0;
    uint32_t materialIndex = kInvalidMaterialIndex;
    uint32_t instanceIndex = 0;
    bool doubleSided = false;
  };

  struct DynamicBufferSlot {
    std::unique_ptr<Buffer> buffer;
    size_t capacityBytes = 0;
  };

  Result<bool, std::string> ensureInitialized();
  Result<bool, std::string> createShaders();
  Result<bool, std::string> ensurePipelines(Format colorFormat,
                                            Format depthFormat);
  Result<bool, std::string> ensureRingBufferCount(uint32_t requiredCount);
  Result<bool, std::string>
  ensureInstanceMatricesRingCapacity(size_t requiredBytes);
  Result<bool, std::string>
  ensureInstanceRemapRingCapacity(size_t requiredBytes);
  Result<bool, std::string> rebuildSceneCache(const RenderScene &scene,
                                              const ResourceManager &resources,
                                              uint32_t materialCount);
  Result<bool, std::string>
  rebuildMaterialTextureAccessCache(const ResourceManager &resources);
  void collectEnvironmentTextureReads(const RenderScene &scene,
                                      const ResourceManager &resources);
  void resetCachedState();
  void resetFrameBuildState();
  void destroyPipelineState();
  void destroyShaders();
  void destroyBuffers();
  [[nodiscard]] bool hasTransmissionContent(const RenderFrameContext &frame);
  [[nodiscard]] RenderPipelineHandle selectMeshPipeline() const;

  GPUDevice &gpu_;
  TransmissionRendererConfig config_{};
  std::pmr::memory_resource *memory_ = std::pmr::get_default_resource();
  std::unique_ptr<Shader> meshShader_;
  std::pmr::vector<DynamicBufferSlot> instanceMatricesRing_;
  std::pmr::vector<DynamicBufferSlot> instanceRemapRing_;

  ShaderHandle meshVertexShader_{};
  ShaderHandle meshFragmentShader_{};
  RenderPipelineHandle meshPipelineHandle_{};
  RenderPipelineHandle meshDoubleSidedPipelineHandle_{};

  Format meshPipelineColorFormat_ = Format::Count;
  Format meshPipelineDepthFormat_ = Format::Count;

  bool initialized_ = false;
  bool loggedMaterialFallbackWarning_ = false;
  uint64_t loggedAddressProbeTopologyVersion_ =
      std::numeric_limits<uint64_t>::max();

  const RenderScene *cachedScene_ = nullptr;
  uint64_t cachedTopologyVersion_ = std::numeric_limits<uint64_t>::max();
  uint64_t cachedMaterialVersion_ = std::numeric_limits<uint64_t>::max();
  uint64_t cachedTransformVersion_ = std::numeric_limits<uint64_t>::max();
  uint64_t cachedGeometryMutationVersion_ =
      std::numeric_limits<uint64_t>::max();
  const RenderScene *cachedTransmissionContentScene_ = nullptr;
  uint64_t cachedTransmissionContentTopologyVersion_ =
      std::numeric_limits<uint64_t>::max();
  uint64_t cachedTransmissionContentMaterialVersion_ =
      std::numeric_limits<uint64_t>::max();
  bool cachedTransmissionContentValid_ = false;
  bool cachedTransmissionContent_ = false;
  EnvironmentHandles cachedEnvironmentHandles_{};

  std::pmr::vector<MeshDrawTemplate> meshDrawTemplates_;
  std::pmr::vector<InstanceData> instanceMatrices_;
  std::pmr::vector<uint32_t> instanceRemap_;
  std::pmr::vector<uint64_t> instanceDataRingUploadVersions_;
  std::pmr::vector<TextureHandle> materialTextureAccessHandles_;
  std::pmr::vector<TextureHandle> environmentTextureAccessHandles_;
  std::pmr::vector<TextureHandle> staticPassTextureReads_;
  std::pmr::vector<MeshPushConstants> meshPushConstants_;
  std::pmr::vector<DrawItem> passDrawItems_;
  std::pmr::vector<TextureHandle> passTextureReads_;
  std::pmr::vector<BufferHandle> passDependencyBuffers_;
  std::pmr::vector<RenderGraphAccessMode> passDependencyBufferAccessModes_;
  std::pmr::vector<BufferHandle> preResolvedTemplateBuffers_;
  std::pmr::vector<BufferHandle> cachedPreResolvedDrawBuffers_;
  uint64_t cachedPreResolvedDrawBufferSignature_ =
      std::numeric_limits<uint64_t>::max();
  std::filesystem::path transmissionVertexPath_{};
  std::filesystem::path transmissionFragmentPath_{};
  TextureHandle preparedSceneColorTexture_{};
  TextureHandle preparedFrameColorTexture_{};
  TextureHandle preparedDepthTexture_{};
  RenderGraphTextureId preparedSceneDepthGraphTexture_{};
};

} // namespace nuri
