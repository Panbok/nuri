#pragma once

#include "nuri/core/result.h"
#include "nuri/gfx/gpu_render_types.h"
#include "nuri/gfx/gpu_types.h"
#include "nuri/gfx/render_graph/render_graph.h"
#include "nuri/gfx/sim/animation_scene_frame_data.h"
#include "nuri/scene/light.h"

#include <cstdint>
#include <cstring>
#include <memory_resource>
#include <optional>
#include <vector>

#include <glm/glm.hpp>

namespace nuri {

class RenderScene;
class ResourceManager;
struct RenderFrameContext;

enum class OpaqueDebugVisualization : uint8_t {
  None = 0,
  WireframeOverlay = 1,
  WireframeOnly = 2,
  TessPatchEdgesHeatmap = 3,
};

enum class TextureFilterMode : uint8_t {
  Bilinear = 0,
  Trilinear = 1,
  Anisotropic = 2,
};

[[nodiscard]] constexpr uint8_t
sanitizeTextureFilterAnisotropy(uint8_t anisotropy) noexcept {
  switch (anisotropy) {
  case 2u:
  case 4u:
  case 8u:
  case 16u:
    return anisotropy;
  default:
    return 8u;
  }
}

[[nodiscard]] constexpr TextureFilterMode
sanitizeTextureFilterMode(TextureFilterMode mode) noexcept {
  switch (mode) {
  case TextureFilterMode::Bilinear:
  case TextureFilterMode::Trilinear:
  case TextureFilterMode::Anisotropic:
    return mode;
  default:
    return TextureFilterMode::Trilinear;
  }
}

struct RenderSettings {
  struct SkyboxSettings {
    bool enabled = true;
  };

  struct OpaqueSettings {
    bool enabled = true;
    OpaqueDebugVisualization debugVisualization =
        OpaqueDebugVisualization::None;
    bool enableInstanceCompute = true;
    bool enableIndirectDraw = true;
    bool enableInstancedDraw = true;
    bool enableMeshLod = true;
    int32_t forcedMeshLod = -1;
    glm::vec3 meshLodDistanceThresholds{8.0f, 16.0f, 32.0f};
    bool enableInstanceAnimation = true;
    bool enableTessellation = false;
    float tessNearDistance = 1.0f;
    float tessFarDistance = 8.0f;
    float tessMinFactor = 1.0f;
    float tessMaxFactor = 6.0f;
    // 0 means "no cap".
    uint32_t tessMaxInstances = 256;
  };

  struct DebugSettings {
    bool enabled = false;
    bool modelBounds = false;
    bool grid = false;
    bool lightIcons = true;
  };

  struct TransparentSettings {
    bool enabled = true;
  };

  struct TransmissionSettings {
    bool enabled = true;
  };

  struct TextureFilteringSettings {
    TextureFilterMode mode = TextureFilterMode::Trilinear;
    uint8_t anisotropy = 8u;
  };

  SkyboxSettings skybox{};
  OpaqueSettings opaque{};
  TransmissionSettings transmission{};
  TransparentSettings transparent{};
  DebugSettings debug{};
  TextureFilteringSettings textureFiltering{};
};

inline void sanitizeTextureFilteringSettings(
    RenderSettings::TextureFilteringSettings &settings) {
  settings.mode = sanitizeTextureFilterMode(settings.mode);
  settings.anisotropy = sanitizeTextureFilterAnisotropy(settings.anisotropy);
}

[[nodiscard]] inline TextureFilterMode effectiveTextureFilterMode(
    const RenderSettings::TextureFilteringSettings &settings,
    uint8_t maxSamplerAnisotropy) {
  const TextureFilterMode mode = sanitizeTextureFilterMode(settings.mode);
  if (mode != TextureFilterMode::Anisotropic) {
    return mode;
  }
  return maxSamplerAnisotropy > 1u ? TextureFilterMode::Anisotropic
                                   : TextureFilterMode::Trilinear;
}

struct CameraFrameState {
  glm::mat4 view{1.0f};
  glm::mat4 proj{1.0f};
  glm::vec4 cameraPos{0.0f, 0.0f, 0.0f, 1.0f};
  float aspectRatio = 1.0f;
};

struct ForwardSceneFrameData {
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
  uint32_t materialSamplerId = 0;
  uint32_t sceneColorTexId = 0;
  uint32_t sceneColorSamplerId = 0;
  uint32_t sceneColorHalfResTexId = 0;
  uint32_t sceneColorQuarterResTexId = 0;
  uint64_t directionalLightBufferAddress = 0;
  uint64_t localLightBufferAddress = 0;
  uint64_t materialHeaderBufferAddress = 0;
  uint64_t materialClearcoatBufferAddress = 0;
  uint64_t materialSheenBufferAddress = 0;
  uint64_t materialTransmissionBufferAddress = 0;
  uint64_t materialSpecularBufferAddress = 0;
  uint32_t directionalLightCount = 0;
  uint32_t localLightCount = 0;

  [[nodiscard]] bool
  operator==(const ForwardSceneFrameData &other) const noexcept {
    return std::memcmp(this, &other, sizeof(ForwardSceneFrameData)) == 0;
  }

  [[nodiscard]] bool
  operator!=(const ForwardSceneFrameData &other) const noexcept {
    return !(*this == other);
  }
};
static_assert(sizeof(ForwardSceneFrameData) == 264,
              "ForwardSceneFrameData must match shader FrameDataBuffer layout");

// GPU-side forwarding of the light metadata carried in ForwardSceneFrameData.
// The CPU owns allocation and updates of ForwardSceneFrameData, then derives
// the resolved GPU addresses/counts below before publishing them for consumers.
// Keep this struct in sync with ForwardSceneFrameData's layout/semantics for
// directional/local light buffer addresses and counts to avoid
// desynchronization bugs when the CPU-side contract changes.
struct ForwardSceneGpuData {
  BufferHandle buffer{};
  uint64_t frameDataAddress = 0;
  uint64_t directionalLightBufferAddress = 0;
  uint64_t localLightBufferAddress = 0;
  uint32_t directionalLightCount = 0;
  uint32_t localLightCount = 0;
};

struct MaterialTableGpuData {
  BufferHandle headerBuffer{};
  BufferHandle clearcoatBuffer{};
  BufferHandle sheenBuffer{};
  BufferHandle transmissionBuffer{};
  BufferHandle specularBuffer{};
  uint64_t headerBufferAddress = 0;
  uint64_t clearcoatBufferAddress = 0;
  uint64_t sheenBufferAddress = 0;
  uint64_t transmissionBufferAddress = 0;
  uint64_t specularBufferAddress = 0;
  uint64_t version = 0;
};

struct OpaqueFrameMetrics {
  uint32_t totalInstances = 0;
  uint32_t visibleInstances = 0;
  uint32_t instancedDraws = 0;
  uint32_t indirectDrawCalls = 0;
  uint32_t indirectCommands = 0;
  uint32_t tessellatedDraws = 0;
  uint32_t tessellatedInstances = 0;
  uint32_t debugOverlayDraws = 0;
  uint32_t debugOverlayFallbackDraws = 0;
  uint32_t debugPatchHeatmapDraws = 0;
  uint32_t computeDispatches = 0;
  uint32_t computeDispatchX = 0;
};

struct RenderFrameMetrics {
  OpaqueFrameMetrics opaque{};
  struct TransparentFrameMetrics {
    uint32_t meshDraws = 0;
    uint32_t contributorSortableDraws = 0;
    uint32_t contributorFixedDraws = 0;
    uint32_t pickDraws = 0;
  } transparent{};
};

struct OpaquePickRequest {
  uint32_t x = 0;
  uint32_t y = 0;
  uint64_t requestId = 0;
};

struct OpaquePickResult {
  uint64_t requestId = 0;
  bool hit = false;
  uint32_t renderableIndex = 0;
};

struct FrameSharedResources {
  std::optional<ForwardSceneGpuData> forwardSceneGpuData{};
  std::optional<MaterialTableGpuData> materialTableGpuData{};
  std::optional<AnimationSceneFrameData> animationSceneGpuData{};
  TextureHandle sceneDepthTexture{};
  RenderGraphTextureId sceneDepthGraphTexture{};
  TextureHandle sceneColorTexture{};
  TextureHandle sceneColorHalfResTexture{};
  TextureHandle sceneColorQuarterResTexture{};
  RenderGraphTextureId sceneColorGraphTexture{};
  TextureHandle frameColorTexture{};
  RenderGraphTextureId opaquePickGraphTexture{};
  RenderGraphTextureId opaquePickDepthGraphTexture{};
  std::optional<LightId> selectedLightId{};
  bool transmissionStageEnabled = false;
  bool transparentStageEnabled = false;
};

struct TransparentStageSortableDraw {
  DrawItem draw{};
  float sortDepth = 0.0f;
  uint32_t stableOrder = 0;
};

// These spans are non-owning views into contributor-managed frame storage.
// They are valid only for the current frame and may be invalidated by the
// contributor's next clear()/beginFrame()-style reset. Copy the data if it
// must outlive the current frame.
struct TransparentStageContribution {
  std::span<const TransparentStageSortableDraw> sortableDraws{};
  std::span<const DrawItem> fixedDraws{};
  std::span<const BufferHandle> dependencyBuffers{};
  std::span<const TextureHandle> textureReads{};
};

using TransparentContributionCollectFn = Result<bool, std::string> (*)(
    void *user, RenderFrameContext &frame, TransparentStageContribution &out);

struct TransparentContributionCollector {
  void *user = nullptr;
  TransparentContributionCollectFn collect = nullptr;
};

class TransparentContributionRegistry {
public:
  explicit TransparentContributionRegistry(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : collectors_(memory != nullptr ? memory
                                      : std::pmr::get_default_resource()) {}

  void clear() { collectors_.clear(); }

  void publish(TransparentContributionCollector collector) {
    if (collector.user == nullptr || collector.collect == nullptr) {
      return;
    }
    collectors_.push_back(collector);
  }

  [[nodiscard]] std::span<const TransparentContributionCollector>
  collectors() const noexcept {
    return std::span<const TransparentContributionCollector>(
        collectors_.data(), collectors_.size());
  }

  [[nodiscard]] bool empty() const noexcept { return collectors_.empty(); }

private:
  std::pmr::vector<TransparentContributionCollector> collectors_;
};

struct RenderFrameContext {
  const RenderScene *scene = nullptr;
  CameraFrameState camera{};
  RenderSettings *settings = nullptr;
  RenderFrameMetrics metrics{};
  // Frame-scoped one-shot opaque pick request/result channel.
  std::optional<OpaquePickRequest> opaquePickRequest{};
  std::optional<OpaquePickResult> opaquePickResult{};
  FrameSharedResources sharedResources{};
  TransparentContributionRegistry transparentContributors{};
  TextureHandle sharedDepthTexture{};
  const ResourceManager *resources = nullptr;
  double timeSeconds = 0.0;
  uint64_t frameIndex = 0;
};

[[nodiscard]] inline const RenderSettings &
renderSettingsOrDefault(const RenderFrameContext &frame) {
  static const RenderSettings kDefaultSettings{};
  return frame.settings ? *frame.settings : kDefaultSettings;
}

[[nodiscard]] inline bool
expectsCurrentFrameSceneColor(const RenderFrameContext &frame) {
  const RenderSettings &settings = renderSettingsOrDefault(frame);
  return frame.sharedResources.transmissionStageEnabled &&
         (settings.skybox.enabled || settings.opaque.enabled);
}

inline void resetFrameSharedResources(RenderFrameContext &frame) {
  frame.sharedResources = {};
  frame.transparentContributors.clear();
}

[[nodiscard]] inline TextureHandle
resolveFrameDepthTexture(const RenderFrameContext &frame) {
  if (nuri::isValid(frame.sharedResources.sceneDepthTexture)) {
    return frame.sharedResources.sceneDepthTexture;
  }
  return frame.sharedDepthTexture;
}

[[nodiscard]] inline TextureHandle
resolveFrameColorTexture(const RenderFrameContext &frame) {
  if (nuri::isValid(frame.sharedResources.frameColorTexture)) {
    return frame.sharedResources.frameColorTexture;
  }
  return {};
}

[[nodiscard]] inline TextureHandle
resolveSceneColorTexture(const RenderFrameContext &frame) {
  if (nuri::isValid(frame.sharedResources.sceneColorTexture)) {
    return frame.sharedResources.sceneColorTexture;
  }
  return {};
}

[[nodiscard]] inline TextureHandle
resolveSceneColorHalfResTexture(const RenderFrameContext &frame) {
  if (nuri::isValid(frame.sharedResources.sceneColorHalfResTexture)) {
    return frame.sharedResources.sceneColorHalfResTexture;
  }
  return {};
}

[[nodiscard]] inline TextureHandle
resolveSceneColorQuarterResTexture(const RenderFrameContext &frame) {
  if (nuri::isValid(frame.sharedResources.sceneColorQuarterResTexture)) {
    return frame.sharedResources.sceneColorQuarterResTexture;
  }
  return {};
}

[[nodiscard]] inline bool
resolveTransmissionStageEnabled(const RenderFrameContext &frame) {
  return frame.sharedResources.transmissionStageEnabled;
}

[[nodiscard]] inline bool
resolveTransparentStageEnabled(const RenderFrameContext &frame) {
  return frame.sharedResources.transparentStageEnabled;
}

[[nodiscard]] inline RenderGraphTextureId
resolveSceneDepthGraphTexture(const RenderFrameContext &frame) {
  if (nuri::isValid(frame.sharedResources.sceneDepthGraphTexture)) {
    return frame.sharedResources.sceneDepthGraphTexture;
  }
  return {};
}

[[nodiscard]] inline LightId
resolveSelectedLightId(const RenderFrameContext &frame) {
  if (frame.sharedResources.selectedLightId.has_value() &&
      isValid(*frame.sharedResources.selectedLightId)) {
    return *frame.sharedResources.selectedLightId;
  }
  return kInvalidLightId;
}

} // namespace nuri
