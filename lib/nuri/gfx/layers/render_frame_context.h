#pragma once

#include "nuri/core/result.h"
#include "nuri/gfx/gpu_render_types.h"
#include "nuri/gfx/gpu_types.h"

#include <any>
#include <cstdint>
#include <memory_resource>
#include <optional>
#include <string>
#include <string_view>
#include <typeindex>
#include <vector>

#include <glm/glm.hpp>

namespace nuri {

class LayerStack;
class RenderScene;
class ResourceManager;

enum class OpaqueDebugVisualization : uint8_t {
  None = 0,
  WireframeOverlay = 1,
  WireframeOnly = 2,
  TessPatchEdgesHeatmap = 3,
};

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
  };

  struct TransparentSettings {
    bool enabled = true;
  };

  SkyboxSettings skybox{};
  OpaqueSettings opaque{};
  TransparentSettings transparent{};
  DebugSettings debug{};
};

struct CameraFrameState {
  glm::mat4 view{1.0f};
  glm::mat4 proj{1.0f};
  glm::vec4 cameraPos{0.0f, 0.0f, 0.0f, 1.0f};
  float aspectRatio = 1.0f;
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

class FrameChannelRegistry {
public:
  explicit FrameChannelRegistry(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : memory_(memory != nullptr ? memory : std::pmr::get_default_resource()),
        entries_(memory_) {}

  void clear() { entries_.clear(); }

  template <typename T> void publish(std::string_view key, T value) {
    for (Entry &entry : entries_) {
      if (entry.key == key) {
        entry.type = std::type_index(typeid(T));
        entry.value = std::any(std::move(value));
        return;
      }
    }

    Entry entry(memory_);
    entry.key.assign(key.data(), key.size());
    entry.type = std::type_index(typeid(T));
    entry.value = std::any(std::move(value));
    entries_.push_back(std::move(entry));
  }

  template <typename T>
  [[nodiscard]] const T *tryGet(std::string_view key) const {
    for (const Entry &entry : entries_) {
      if (entry.key != key || entry.type != std::type_index(typeid(T))) {
        continue;
      }
      return std::any_cast<T>(&entry.value);
    }
    return nullptr;
  }

private:
  struct Entry {
    explicit Entry(std::pmr::memory_resource *memory)
        : key(memory != nullptr ? memory : std::pmr::get_default_resource()) {}

    std::pmr::string key;
    std::type_index type = std::type_index(typeid(void));
    std::any value{};
  };

  std::pmr::memory_resource *memory_ = nullptr;
  std::pmr::vector<Entry> entries_;
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

constexpr std::string_view kFrameChannelSceneDepthTexture = "SceneDepthTexture";
constexpr std::string_view kFrameChannelSceneDepthGraphTexture =
    "SceneDepthGraphTexture";
constexpr std::string_view kFrameChannelTransparentStageEnabled =
    "TransparentStageEnabled";
constexpr std::string_view kFrameChannelOpaquePickGraphTexture =
    "OpaquePickGraphTexture";
constexpr std::string_view kFrameChannelOpaquePickDepthGraphTexture =
    "OpaquePickDepthGraphTexture";

struct RenderFrameContext {
  const RenderScene *scene = nullptr;
  CameraFrameState camera{};
  RenderSettings *settings = nullptr;
  RenderFrameMetrics metrics{};
  // Frame-scoped one-shot opaque pick request/result channel.
  std::optional<OpaquePickRequest> opaquePickRequest{};
  std::optional<OpaquePickResult> opaquePickResult{};
  FrameChannelRegistry channels{};
  const LayerStack *layerStack = nullptr;
  TextureHandle sharedDepthTexture{};
  const ResourceManager *resources = nullptr;
  double timeSeconds = 0.0;
  uint64_t frameIndex = 0;
};

[[nodiscard]] inline TextureHandle
resolveFrameDepthTexture(const RenderFrameContext &frame) {
  if (const TextureHandle *depth =
          frame.channels.tryGet<TextureHandle>(kFrameChannelSceneDepthTexture);
      depth != nullptr && nuri::isValid(*depth)) {
    return *depth;
  }
  return frame.sharedDepthTexture;
}

} // namespace nuri
