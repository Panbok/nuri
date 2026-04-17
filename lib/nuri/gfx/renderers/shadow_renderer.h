#pragma once

#include "nuri/core/result.h"
#include "nuri/core/runtime_config.h"
#include "nuri/defines.h"
#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/render_graph/render_graph.h"
#include "nuri/gfx/renderers/detail/instance_data.h"
#include "nuri/gfx/shader.h"
#include "nuri/resources/cpu/mesh_data.h"
#include "nuri/resources/gpu/buffer.h"
#include "nuri/resources/gpu/material.h"
#include "nuri/resources/gpu/model.h"
#include "nuri/scene/render_scene.h"

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <memory_resource>
#include <optional>
#include <string>
#include <vector>

namespace nuri {

using ShadowRendererConfig = RuntimeOpaqueShaderConfig;
class ResourceManager;

struct BufferDependency {
  BufferHandle handle{};
  RenderGraphAccessMode mode = RenderGraphAccessMode::Read;
};

struct TextureDependency {
  TextureHandle handle{};
  RenderGraphAccessMode mode = RenderGraphAccessMode::Read;
};

class NURI_API ShadowRenderer {
public:
  explicit ShadowRenderer(GPUDevice &gpu, std::pmr::memory_resource *memory =
                                              std::pmr::get_default_resource());
  ShadowRenderer(
      GPUDevice &gpu, const ShadowRendererConfig &config,
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  ~ShadowRenderer();

  ShadowRenderer(const ShadowRenderer &) = delete;
  ShadowRenderer &operator=(const ShadowRenderer &) = delete;
  ShadowRenderer(ShadowRenderer &&) = delete;
  ShadowRenderer &operator=(ShadowRenderer &&) = delete;

  Result<bool, std::string> publishFrameData(RenderFrameContext &frame);
  Result<bool, std::string> prepareShadowGraphPasses(RenderFrameContext &frame);
  [[nodiscard]] bool hasPreparedShadowDepthPasses() const noexcept;
  Result<bool, std::string> appendShadowDepthPasses(RenderFrameContext &frame,
                                                    RenderGraphBuilder &graph);

private:
  using FrameData = ForwardSceneFrameData;
  static_assert(sizeof(FrameData) == 352,
                "ShadowRenderer::FrameData must match shader layout");

  struct PushConstants {
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
    float tessNearDistance = 1.0f;
    float tessFarDistance = 8.0f;
    float tessMinFactor = 1.0f;
    float tessMaxFactor = 1.0f;
    uint32_t debugVisualizationMode = 0;
  };
  static_assert(
      sizeof(PushConstants) <= 128,
      "ShadowRenderer::PushConstants exceeds Vulkan minimum guarantee");

  struct MeshDrawTemplate {
    const Renderable *renderable = nullptr;
    const Submesh *submesh = nullptr;
    uint32_t submeshIndex = 0;
    uint32_t instanceIndex = 0;
    BufferHandle indexBuffer{};
    uint64_t indexBufferOffset = 0;
    IndexFormat indexFormat = IndexFormat::U32;
    BufferHandle baseVertexBuffer{};
    uint64_t vertexBufferByteOffset = 0;
    uint64_t vertexBufferAddress = 0;
    uint64_t vertexDecodeBufferAddress = 0;
    uint32_t vertexDecodeIndex = 0;
    uint32_t packedVertexFormat = 0;
    uint32_t materialIndex = 0;
    bool doubleSided = false;
    bool alphaMasked = false;
  };

  struct DynamicBufferSlot {
    std::unique_ptr<Buffer> buffer;
    size_t capacityBytes = 0;
  };

  struct PreviewPushConstants {
    uint32_t sourceTexId = kInvalidShadowBindlessIndex;
    float depthScale = 1.0f;
    float depthBias = 0.0f;
  };
  static_assert(sizeof(PreviewPushConstants) == 12,
                "ShadowRenderer::PreviewPushConstants layout changed");
  static_assert(
      sizeof(PreviewPushConstants) <= 128,
      "ShadowRenderer::PreviewPushConstants exceeds Vulkan minimum guarantee");

  Result<bool, std::string> ensureInitialized();
  Result<bool, std::string> createShaders();
  Result<bool, std::string> createPreviewShaders();
  Result<bool, std::string> createPipelines();
  Result<bool, std::string> createPreviewPipeline();
  Result<bool, std::string>
  ensureShadowResources(const RenderSettings::ShadowSettings &settings);
  Result<bool, std::string> ensureRingBufferCount(uint32_t requiredCount);
  Result<bool, std::string>
  ensureInstanceMatricesRingCapacity(size_t requiredBytes);
  Result<bool, std::string>
  ensureInstanceRemapRingCapacity(size_t requiredBytes);
  Result<bool, std::string> ensureShadowFrameRingCapacity(size_t requiredBytes);
  Result<bool, std::string>
  ensureRingCapacity(std::pmr::vector<DynamicBufferSlot> &ring,
                     size_t requiredBytes, std::string_view debugName,
                     std::span<uint64_t> uploadVersions);
  Result<bool, std::string> rebuildSceneCache(const RenderScene &scene,
                                              const ResourceManager &resources,
                                              uint32_t materialCount);
  Result<bool, std::string>
  updateShadowFrameData(RenderFrameContext &frame,
                        const RenderSettings::ShadowSettings &settings,
                        uint32_t shadowMapSize);
  Result<bool, std::string>
  buildShadowDraws(RenderFrameContext &frame, uint32_t frameSlot,
                   const ForwardSceneGpuData &sceneGpu);
  void destroyShadowResources();
  void destroyBuffers();
  void destroyShaders();
  void destroyPipelineState();
  void resetCachedState();
  void resetFrameBuildState();

  GPUDevice &gpu_;
  ShadowRendererConfig config_{};
  std::pmr::memory_resource *memory_ = std::pmr::get_default_resource();
  std::unique_ptr<Shader> shadowShader_;
  std::unique_ptr<Shader> depthShader_;
  std::unique_ptr<Shader> depthAlphaShader_;
  std::pmr::vector<DynamicBufferSlot> instanceMatricesRing_;
  std::pmr::vector<DynamicBufferSlot> instanceRemapRing_;
  std::pmr::vector<DynamicBufferSlot> shadowFrameRing_;
  std::pmr::vector<uint64_t> instanceDataRingUploadVersions_;
  std::pmr::vector<uint64_t> shadowFrameUploadSignatures_;
  std::pmr::vector<MeshDrawTemplate> meshDrawTemplates_;
  std::pmr::vector<InstanceData> instanceMatrices_;
  std::pmr::vector<uint32_t> instanceRemap_;
  std::pmr::vector<PushConstants> drawPushConstants_;
  std::pmr::vector<DrawItem> drawItems_;
  std::pmr::vector<BufferDependency> passBufferDependencies_;
  std::pmr::vector<TextureDependency> passTextureDependencies_;
  std::pmr::vector<TextureDependency> previewTextureDependencies_;

  ShaderHandle shadowVertexShader_{};
  ShaderHandle depthFragmentShader_{};
  ShaderHandle depthAlphaFragmentShader_{};
  ShaderHandle previewVertexShader_{};
  ShaderHandle previewFragmentShader_{};
  RenderPipelineHandle shadowPipelineHandle_{};
  RenderPipelineHandle shadowDoubleSidedPipelineHandle_{};
  RenderPipelineHandle shadowAlphaPipelineHandle_{};
  RenderPipelineHandle shadowAlphaDoubleSidedPipelineHandle_{};
  RenderPipelineHandle previewPipelineHandle_{};
  TextureHandle shadowDepthTexture_{};
  TextureHandle shadowDebugPreviewTexture_{};
  SamplerHandle rawDepthSampler_{};
  SamplerHandle compareDepthSampler_{};
  uint32_t shadowMapSize_ = 0u;
  uint32_t preparedShadowDrawCount_ = 0u;
  bool initialized_ = false;
  bool hasPreparedShadowDepthPasses_ = false;
  bool hasPreparedShadowPreviewPass_ = false;
  PreviewPushConstants previewPushConstants_{};
  DrawItem previewDraw_{};

  const RenderScene *cachedScene_ = nullptr;
  uint64_t cachedTopologyVersion_ = std::numeric_limits<uint64_t>::max();
  uint64_t cachedMaterialVersion_ = std::numeric_limits<uint64_t>::max();
  uint64_t cachedTransformVersion_ = std::numeric_limits<uint64_t>::max();
  uint64_t cachedGeometryMutationVersion_ =
      std::numeric_limits<uint64_t>::max();
  ShadowFrameGpuData shadowFrameCpuData_{};
  ShadowDebugFrameData shadowDebugFrameData_{};
};

} // namespace nuri
