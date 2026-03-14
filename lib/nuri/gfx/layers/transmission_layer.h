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

using TransmissionLayerConfig = RuntimeOpaqueShaderConfig;

class ResourceManager;
class Shader;

class NURI_API TransmissionLayer final : public Layer {
public:
  explicit TransmissionLayer(
      GPUDevice &gpu, TransmissionLayerConfig config,
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  ~TransmissionLayer() override;

  TransmissionLayer(const TransmissionLayer &) = delete;
  TransmissionLayer &operator=(const TransmissionLayer &) = delete;
  TransmissionLayer(TransmissionLayer &&) = delete;
  TransmissionLayer &operator=(TransmissionLayer &&) = delete;

  static std::unique_ptr<TransmissionLayer>
  create(GPUDevice &gpu, TransmissionLayerConfig config,
         std::pmr::memory_resource *memory = std::pmr::get_default_resource()) {
    return std::make_unique<TransmissionLayer>(gpu, std::move(config), memory);
  }

  void onAttach() override;
  void onDetach() override;
  void publishFrameData(RenderFrameContext &frame) override;
  Result<bool, std::string>
  buildRenderGraph(RenderFrameContext &frame,
                   RenderGraphBuilder &graph) override;

private:
  enum FrameDataFlags : uint32_t {
    HasIblDiffuse = 1u << 0u,
    HasIblSpecular = 1u << 1u,
    HasIblSheen = 1u << 2u,
    HasBrdfLut = 1u << 3u,
    OutputLinearToSrgb = 1u << 4u,
    HasSceneColor = 1u << 5u,
  };

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
    uint32_t sceneColorTexId = 0;
    uint32_t sceneColorSamplerId = 0;
    uint32_t reserved0 = 0;
    uint32_t reserved1 = 0;

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
             brdfLutTexId == other.brdfLutTexId && flags == other.flags &&
             cubemapSamplerId == other.cubemapSamplerId &&
             sceneColorTexId == other.sceneColorTexId &&
             sceneColorSamplerId == other.sceneColorSamplerId &&
             reserved0 == other.reserved0 && reserved1 == other.reserved1;
    }
  };
  static_assert(sizeof(FrameData) == 192,
                "TransmissionLayer::FrameData must match shader layout");

  struct MeshPushConstants {
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
    float tessMaxFactor = 1.0f;
    uint32_t debugVisualizationMode = 0;
  };
  static_assert(
      sizeof(MeshPushConstants) <= 128,
      "TransmissionLayer::MeshPushConstants exceeds Vulkan guarantee");

  struct CopyPushConstants {
    uint32_t sourceTexId = 0;
    uint32_t sourceSamplerId = 0;
    uint32_t flags = 0;
    uint32_t reserved0 = 0;
  };
  static_assert(
      sizeof(CopyPushConstants) <= 128,
      "TransmissionLayer::CopyPushConstants exceeds Vulkan guarantee");

  struct MeshDrawTemplate {
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
  Result<bool, std::string> ensureSceneColorTexture();
  [[nodiscard]] TextureHandle
  currentSceneColorTexture(uint64_t frameIndex) const;
  [[nodiscard]] TextureHandle currentSceneColorTextureMip(uint64_t frameIndex,
                                                          uint32_t level) const;
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
  [[nodiscard]] RenderPipelineHandle selectMeshPipeline(bool doubleSided) const;

  GPUDevice &gpu_;
  TransmissionLayerConfig config_{};
  std::pmr::memory_resource *memory_ = std::pmr::get_default_resource();
  std::unique_ptr<Shader> meshShader_;
  std::unique_ptr<Shader> copyShader_;
  std::unique_ptr<Buffer> frameDataBuffer_;
  std::unique_ptr<Buffer> materialBuffer_;
  std::pmr::vector<DynamicBufferSlot> instanceMatricesRing_;
  std::pmr::vector<DynamicBufferSlot> instanceRemapRing_;

  ShaderHandle meshVertexShader_{};
  ShaderHandle meshFragmentShader_{};
  ShaderHandle copyVertexShader_{};
  ShaderHandle copyFragmentShader_{};
  RenderPipelineHandle meshPipelineHandle_{};
  RenderPipelineHandle meshDoubleSidedPipelineHandle_{};
  RenderPipelineHandle copyPipelineHandle_{};

  Format meshPipelineColorFormat_ = Format::Count;
  Format meshPipelineDepthFormat_ = Format::Count;
  Format copyPipelineColorFormat_ = Format::Count;
  Format copyPipelineDepthFormat_ = Format::Count;

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
  const RenderScene *cachedTransmissionContentScene_ = nullptr;
  uint64_t cachedTransmissionContentTopologyVersion_ =
      std::numeric_limits<uint64_t>::max();
  uint64_t cachedTransmissionContentMaterialVersion_ =
      std::numeric_limits<uint64_t>::max();
  bool cachedTransmissionContentValid_ = false;
  bool cachedTransmissionContent_ = false;

  std::pmr::vector<MeshDrawTemplate> meshDrawTemplates_;
  std::pmr::vector<glm::mat4> instanceMatrices_;
  std::pmr::vector<uint32_t> instanceRemap_;
  std::pmr::vector<uint64_t> instanceDataRingUploadVersions_;
  std::pmr::vector<MaterialGpuData> materialGpuDataCache_;
  std::pmr::vector<TextureHandle> materialTextureAccessHandles_;
  std::pmr::vector<TextureHandle> environmentTextureAccessHandles_;
  std::pmr::vector<MeshPushConstants> meshPushConstants_;
  std::pmr::vector<DrawItem> passDrawItems_;
  std::pmr::vector<TextureHandle> passTextureReads_;
  std::pmr::vector<BufferHandle> passDependencyBuffers_;
  FrameData frameData_{};
  FrameData uploadedFrameData_{};
  bool frameDataUploadValid_ = false;
  std::pmr::vector<CopyPushConstants> copyPushConstantsRing_;
  std::filesystem::path transmissionFragmentPath_{};
  std::filesystem::path fullscreenCopyVertexPath_{};
  std::filesystem::path sceneCopyFragmentPath_{};
  std::pmr::vector<TextureHandle> sceneColorTextures_;
  std::pmr::vector<TextureHandle> sceneColorMipTextures_;
  Format sceneColorTextureFormat_ = Format::Count;
  uint32_t sceneColorTextureWidth_ = 0;
  uint32_t sceneColorTextureHeight_ = 0;
};

} // namespace nuri
