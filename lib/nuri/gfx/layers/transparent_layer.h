#pragma once

#include "nuri/core/layer.h"
#include "nuri/core/runtime_config.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/resources/cpu/mesh_data.h"
#include "nuri/resources/gpu/buffer.h"
#include "nuri/resources/gpu/material.h"
#include "nuri/resources/gpu/model.h"
#include "nuri/scene/render_scene.h"

#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <memory_resource>
#include <utility>
#include <vector>

#include <glm/glm.hpp>

namespace nuri {

using TransparentLayerConfig = RuntimeOpaqueShaderConfig;

class ResourceManager;
class Shader;

class NURI_API TransparentLayer final : public Layer {
public:
  explicit TransparentLayer(
      GPUDevice &gpu, TransparentLayerConfig config,
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  ~TransparentLayer() override;

  TransparentLayer(const TransparentLayer &) = delete;
  TransparentLayer &operator=(const TransparentLayer &) = delete;
  TransparentLayer(TransparentLayer &&) = delete;
  TransparentLayer &operator=(TransparentLayer &&) = delete;

  static std::unique_ptr<TransparentLayer>
  create(GPUDevice &gpu, TransparentLayerConfig config,
         std::pmr::memory_resource *memory = std::pmr::get_default_resource()) {
    return std::make_unique<TransparentLayer>(gpu, std::move(config), memory);
  }

  void onAttach() override;
  void onDetach() override;
  void publishFrameData(RenderFrameContext &frame) override;
  Result<bool, std::string>
  buildRenderGraph(RenderFrameContext &frame,
                   RenderGraphBuilder &graph) override;

private:
  struct FrameData {
    glm::mat4 view{1.0f};
    glm::mat4 proj{1.0f};
    glm::vec4 cameraPos{0.0f, 0.0f, 0.0f, 1.0f};
    uint32_t cubemapTexId = 0;
    uint32_t hasCubemap = 0;
    uint32_t irradianceTexId = 0;
    uint32_t prefilteredGgxTexId = 0;
    uint32_t prefilteredCharlieTexId = 0;
    uint32_t brdfLutTexId = 0;
    uint32_t flags = 0;
    uint32_t cubemapSamplerId = 0;

    [[nodiscard]] bool operator==(const FrameData &other) const noexcept {
      for (int column = 0; column < 4; ++column) {
        for (int row = 0; row < 4; ++row) {
          if (view[column][row] != other.view[column][row] ||
              proj[column][row] != other.proj[column][row]) {
            return false;
          }
        }
      }
      return cameraPos.x == other.cameraPos.x &&
             cameraPos.y == other.cameraPos.y &&
             cameraPos.z == other.cameraPos.z &&
             cameraPos.w == other.cameraPos.w &&
             cubemapTexId == other.cubemapTexId &&
             hasCubemap == other.hasCubemap &&
             irradianceTexId == other.irradianceTexId &&
             prefilteredGgxTexId == other.prefilteredGgxTexId &&
             prefilteredCharlieTexId == other.prefilteredCharlieTexId &&
             brdfLutTexId == other.brdfLutTexId &&
             flags == other.flags &&
             cubemapSamplerId == other.cubemapSamplerId;
    }
  };
  static_assert(sizeof(FrameData) == 176,
                "TransparentLayer::FrameData must match shader layout");

  struct PushConstants {
    uint64_t frameDataAddress = 0;
    uint64_t vertexBufferAddress = 0;
    uint64_t instanceMatricesAddress = 0;
    uint64_t instanceRemapAddress = 0;
    uint64_t materialBufferAddress = 0;
    uint64_t instanceCentersPhaseAddress = 0;
    uint64_t instanceBaseMatricesAddress = 0;
    uint32_t instanceCount = 0;
    uint32_t materialIndex = 0;
    float timeSeconds = 0.0f;
    float tessNearDistance = 1.0f;
    float tessFarDistance = 8.0f;
    float tessMinFactor = 1.0f;
    float tessMaxFactor = 6.0f;
    uint32_t debugVisualizationMode = 0;
  };
  static_assert(sizeof(PushConstants) <= 128,
                "TransparentLayer::PushConstants exceeds Vulkan guarantee");

  struct MeshDrawTemplate {
    // These pointers reference scene-owned topology and remain valid only
    // while RenderScene::topologyVersion() is unchanged.
    const Renderable *renderable = nullptr;
    const Submesh *submesh = nullptr;
    uint32_t submeshIndex = 0;
    BufferHandle indexBuffer{};
    uint64_t indexBufferOffset = 0;
    uint64_t vertexBufferAddress = 0;
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
  Result<bool, std::string> ensureFrameDataBufferCapacity(size_t requiredBytes);
  Result<bool, std::string> ensureMaterialBufferCapacity(size_t requiredBytes);
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
  Result<bool, std::string> collectContributorDraws(RenderFrameContext &frame);
  Result<bool, std::string> appendTransparentPass(
      RenderGraphBuilder &graph, TextureHandle depthTexture,
      RenderGraphTextureId sceneDepthGraphTexture,
      std::span<const TransparentStageSortableDraw> sortableDraws,
      std::span<const DrawItem> fixedDraws,
      std::span<const TextureHandle> textureReads,
      std::span<const BufferHandle> dependencyBuffers);
  Result<bool, std::string>
  appendTransparentPickPass(RenderFrameContext &frame,
                            RenderGraphBuilder &graph);
  void collectEnvironmentTextureReads(const RenderScene &scene,
                                      const ResourceManager &resources);
  void resetCachedState();
  void resetFrameBuildState();
  void destroyPipelineState();
  void destroyShaders();
  void destroyBuffers();
  static void
  sortTransparentDraws(std::span<TransparentStageSortableDraw> draws);
  [[nodiscard]] RenderPipelineHandle selectMeshPipeline(bool doubleSided) const;
  [[nodiscard]] RenderPipelineHandle selectPickPipeline(bool doubleSided) const;

  GPUDevice &gpu_;
  TransparentLayerConfig config_{};
  std::pmr::memory_resource *memory_ = std::pmr::get_default_resource();
  std::unique_ptr<Shader> meshShader_;
  std::unique_ptr<Shader> meshPickShader_;
  std::unique_ptr<Buffer> frameDataBuffer_;
  std::unique_ptr<Buffer> materialBuffer_;
  std::pmr::vector<DynamicBufferSlot> instanceMatricesRing_;
  std::pmr::vector<DynamicBufferSlot> instanceRemapRing_;

  ShaderHandle meshVertexShader_{};
  ShaderHandle meshFragmentShader_{};
  ShaderHandle meshPickFragmentShader_{};
  RenderPipelineHandle meshPipelineHandle_{};
  RenderPipelineHandle meshDoubleSidedPipelineHandle_{};
  RenderPipelineHandle meshPickPipelineHandle_{};
  RenderPipelineHandle meshPickDoubleSidedPipelineHandle_{};

  Format meshPipelineColorFormat_ = Format::Count;
  Format meshPipelineDepthFormat_ = Format::Count;
  Format pickPipelineDepthFormat_ = Format::Count;

  size_t frameDataBufferCapacityBytes_ = 0;
  size_t materialBufferCapacityBytes_ = 0;
  bool initialized_ = false;
  bool loggedMaterialFallbackWarning_ = false;

  const RenderScene *cachedScene_ = nullptr;
  uint64_t cachedTopologyVersion_ = std::numeric_limits<uint64_t>::max();
  uint64_t cachedMaterialVersion_ = std::numeric_limits<uint64_t>::max();
  uint64_t cachedTransformVersion_ = std::numeric_limits<uint64_t>::max();
  uint64_t cachedGeometryMutationVersion_ =
      std::numeric_limits<uint64_t>::max();

  std::pmr::vector<MeshDrawTemplate> meshDrawTemplates_;
  std::pmr::vector<glm::mat4> instanceMatrices_;
  std::pmr::vector<uint32_t> instanceRemap_;
  std::pmr::vector<uint64_t> instanceDataRingUploadVersions_;
  std::pmr::vector<MaterialGpuData> materialGpuDataCache_;
  std::pmr::vector<TextureHandle> materialTextureAccessHandles_;
  std::pmr::vector<TextureHandle> environmentTextureAccessHandles_;
  std::pmr::vector<TransparentStageSortableDraw> contributorSortableDraws_;
  std::pmr::vector<DrawItem> contributorFixedDraws_;
  std::pmr::vector<TextureHandle> contributorTextureReads_;
  std::pmr::vector<PushConstants> drawPushConstants_;
  std::pmr::vector<PushConstants> pickPushConstants_;
  std::pmr::vector<TransparentStageSortableDraw> meshSortableDraws_;
  std::pmr::vector<TransparentStageSortableDraw> sortableDraws_;
  std::pmr::vector<DrawItem> fixedDraws_;
  std::pmr::vector<DrawItem> passDrawItems_;
  std::pmr::vector<DrawItem> pickDrawItems_;
  std::pmr::vector<TextureHandle> passTextureReads_;
  std::pmr::vector<BufferHandle> passDependencyBuffers_;
  FrameData frameData_{};
  FrameData uploadedFrameData_{};
  bool frameDataUploadValid_ = false;
  std::filesystem::path alphaPickFragmentPath_{};
};

} // namespace nuri
