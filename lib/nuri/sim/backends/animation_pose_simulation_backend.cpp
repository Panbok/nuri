#include "nuri/pch.h"

#include "nuri/sim/backends/animation_pose_simulation_backend.h"

#include "nuri/core/containers/hash_map.h"
#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
#include "nuri/gfx/renderers/detail/instance_data.h"
#include "nuri/gfx/sim/animation_gpu_services.h"
#include "nuri/resources/gpu/resource_manager.h"
#include "nuri/scene_runtime/scene_runtime_host.h"
#include "nuri/sim/animation_pose_simulation.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/matrix_decompose.hpp>

#include <array>
#include <optional>
#include <vector>

namespace nuri {
namespace {

constexpr uint32_t kInvalidIndex = std::numeric_limits<uint32_t>::max();
constexpr uint32_t kComputeWorkgroupSize = 64u;
constexpr size_t kPackedVertexStrideBytes = 32u;
constexpr float kLoopWrapEpsilonSeconds = 1.0e-5f;
constexpr float kBlendWeightEpsilon = 1.0e-5f;
constexpr size_t kClipSlotCount = 2u;
constexpr size_t kPrimaryClipSlot = 0u;
constexpr size_t kSecondaryClipSlot = 1u;
constexpr size_t kAnimationSceneFrameHistorySlots = 3u;

struct ClipBufferLabels {
  std::string_view keyTimes;
  std::string_view values;
  std::string_view channels;
  std::string_view sampledNodes;
  std::string_view sampledWeights;
};

struct AnimationNodeStateGpu {
  glm::vec4 translation{0.0f, 0.0f, 0.0f, 0.0f};
  glm::vec4 rotation{0.0f, 0.0f, 0.0f, 1.0f};
  glm::vec4 scale{1.0f, 1.0f, 1.0f, 0.0f};
};
static_assert(sizeof(AnimationNodeStateGpu) == 48);

struct AnimationNodeMetaGpu {
  uint32_t parentIndex = kInvalidIndex;
  uint32_t weightOffset = 0u;
  uint32_t weightCount = 0u;
  uint32_t reserved = 0u;
};
static_assert(sizeof(AnimationNodeMetaGpu) == 16);

struct AnimationChannelGpu {
  uint32_t keyOffset = 0u;
  uint32_t valueOffset = 0u;
  uint32_t keyCount = 0u;
  uint32_t valueArity = 0u;
  uint32_t targetNodeIndex = 0u;
  uint32_t path = 0u;
  uint32_t interpolation = 0u;
  uint32_t targetWeightOffset = 0u;
};
static_assert(sizeof(AnimationChannelGpu) == 32);

struct AnimationRenderableBindingGpu {
  uint32_t runtimeRenderableIndex = 0u;
  uint32_t nodeIndex = 0u;
  uint32_t reserved0 = 0u;
  uint32_t reserved1 = 0u;
};
static_assert(sizeof(AnimationRenderableBindingGpu) == 16);

struct SamplePushConstants {
  uint64_t channelsAddress = 0u;
  uint64_t keyTimesAddress = 0u;
  uint64_t valuesAddress = 0u;
  uint64_t nodeStatesAddress = 0u;
  uint64_t sampledWeightsAddress = 0u;
  float sampleTimeSeconds = 0.0f;
  uint32_t channelCount = 0u;
  uint32_t reserved = 0u;
};
static_assert(sizeof(SamplePushConstants) <= 128);

struct BlendPushConstants {
  uint64_t sourceNodeStatesAddressA = 0u;
  uint64_t sourceNodeStatesAddressB = 0u;
  uint64_t outputNodeStatesAddress = 0u;
  uint64_t sourceWeightsAddressA = 0u;
  uint64_t sourceWeightsAddressB = 0u;
  uint64_t outputWeightsAddress = 0u;
  float blendWeight = 0.0f;
  uint32_t nodeCount = 0u;
  uint32_t weightCount = 0u;
  uint32_t reserved = 0u;
};
static_assert(sizeof(BlendPushConstants) <= 128);

struct WorldPushConstants {
  uint64_t nodeStatesAddress = 0u;
  uint64_t nodeMetaAddress = 0u;
  uint64_t depthOrderedNodesAddress = 0u;
  uint64_t worldMatricesAddress = 0u;
  glm::mat4 rootTransform{1.0f};
  uint32_t nodeStart = 0u;
  uint32_t nodeCount = 0u;
};
static_assert(sizeof(WorldPushConstants) <= 128);

struct ScatterPushConstants {
  uint64_t renderableBindingsAddress = 0u;
  uint64_t worldMatricesAddress = 0u;
  uint64_t instanceMatricesAddress = 0u;
  uint32_t bindingCount = 0u;
};
static_assert(sizeof(ScatterPushConstants) <= 128);

struct MorphPushConstants {
  uint64_t sourceVertexAddress = 0u;
  uint64_t outputVertexAddress = 0u;
  uint64_t morphDeltaAddress = 0u;
  uint64_t weightAddress = 0u;
  uint32_t weightOffset = 0u;
  uint32_t weightCount = 0u;
  uint32_t morphTargetCount = 0u;
  uint32_t vertexCount = 0u;
};
static_assert(sizeof(MorphPushConstants) <= 128);

struct SkinPalettePushConstants {
  uint64_t worldMatricesAddress = 0u;
  uint64_t jointNodeIndicesAddress = 0u;
  uint64_t inverseBindMatricesAddress = 0u;
  uint64_t paletteAddress = 0u;
  uint32_t renderableNodeIndex = 0u;
  uint32_t jointCount = 0u;
  uint32_t reserved0 = 0u;
  uint32_t reserved1 = 0u;
};
static_assert(sizeof(SkinPalettePushConstants) <= 128);

struct SkinPushConstants {
  uint64_t sourceVertexAddress = 0u;
  uint64_t outputVertexAddress = 0u;
  uint64_t skinInfluenceAddress = 0u;
  uint64_t paletteAddress = 0u;
  uint32_t vertexCount = 0u;
  uint32_t reserved0 = 0u;
  uint32_t reserved1 = 0u;
  uint32_t reserved2 = 0u;
};
static_assert(sizeof(SkinPushConstants) <= 128);

template <typename T>
Result<bool, std::string> uploadVector(GPUDevice &gpu, Buffer &buffer,
                                       const std::pmr::vector<T> &values) {
  if (values.empty()) {
    return Result<bool, std::string>::makeResult(true);
  }
  const std::span<const std::byte> bytes =
      std::as_bytes(std::span<const T>(values.data(), values.size()));
  return gpu.updateBuffer(buffer.handle(), bytes, 0u);
}

void decomposeTransform(const glm::mat4 &matrix, glm::vec3 &translation,
                        glm::quat &rotation, glm::vec3 &scale) {
  glm::vec3 skew(0.0f);
  glm::vec4 perspective(0.0f);
  glm::mat4 local = matrix;
  if (!glm::decompose(local, scale, rotation, translation, skew, perspective)) {
    translation = glm::vec3(matrix[3]);
    rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    scale = glm::vec3(1.0f);
    return;
  }
  rotation = glm::normalize(rotation);
}

AnimationNodeStateGpu makeNodeStateGpu(const glm::mat4 &localFromParent) {
  glm::vec3 translation(0.0f);
  glm::quat rotation(1.0f, 0.0f, 0.0f, 0.0f);
  glm::vec3 scale(1.0f);
  decomposeTransform(localFromParent, translation, rotation, scale);
  return AnimationNodeStateGpu{
      .translation = glm::vec4(translation, 0.0f),
      .rotation = glm::vec4(rotation.x, rotation.y, rotation.z, rotation.w),
      .scale = glm::vec4(scale, 0.0f),
  };
}

uint32_t dispatchCount(uint32_t elementCount) {
  return std::max(1u, (elementCount + (kComputeWorkgroupSize - 1u)) /
                          kComputeWorkgroupSize);
}

template <typename T, size_t N>
void appendComputeDispatch(std::pmr::vector<ComputeDispatchItem> &out,
                           ComputePipelineHandle pipeline, uint32_t x,
                           const T &pushConstants,
                           const std::array<BufferHandle, N> &dependencies,
                           size_t dependencyCount, std::string_view debugLabel,
                           uint32_t debugColor) {
  out.push_back(ComputeDispatchItem{
      .pipeline = pipeline,
      .dispatch = {.x = x, .y = 1u, .z = 1u},
      .pushConstants =
          std::as_bytes(std::span<const T>(&pushConstants, size_t{1u})),
      .dependencyBuffers =
          std::span<const BufferHandle>(dependencies.data(), dependencyCount),
      .debugLabel = debugLabel,
      .debugColor = debugColor,
  });
}

float wrapLoopTime(float timeSeconds, float durationSeconds) {
  if (!(durationSeconds > 0.0f)) {
    return 0.0f;
  }
  const float clampedTime = std::max(0.0f, timeSeconds);
  float wrapped = std::fmod(clampedTime, durationSeconds);
  if (wrapped < 0.0f) {
    wrapped += durationSeconds;
  }
  if (clampedTime > 0.0f && wrapped <= kLoopWrapEpsilonSeconds) {
    return std::nextafter(durationSeconds, 0.0f);
  }
  return wrapped;
}

[[nodiscard]] bool clipStateValid(const ScenePrefab &prefab,
                                  const AnimationPoseClipState &clip) {
  return clip.clipIndex < prefab.animations.size();
}

[[nodiscard]] bool blendEnabled(const ScenePrefab &prefab,
                                const AnimationPoseSimulationParams &params) {
  return params.blendMode == AnimationPoseBlendMode::Lerp &&
         params.blendWeight > kBlendWeightEpsilon &&
         clipStateValid(prefab, params.primary) &&
         clipStateValid(prefab, params.secondary);
}

[[nodiscard]] bool
secondaryOnlyBlendEnabled(const ScenePrefab &prefab,
                          const AnimationPoseSimulationParams &params) {
  return blendEnabled(prefab, params) &&
         params.blendWeight > (1.0f - kBlendWeightEpsilon);
}

[[nodiscard]] ClipBufferLabels clipBufferLabels(size_t clipSlot) {
  switch (clipSlot) {
  case kPrimaryClipSlot:
    return ClipBufferLabels{
        .keyTimes = "animation_pose_primary_key_times",
        .values = "animation_pose_primary_values",
        .channels = "animation_pose_primary_channels",
        .sampledNodes = "animation_pose_primary_sampled_nodes",
        .sampledWeights = "animation_pose_primary_sampled_weights",
    };
  case kSecondaryClipSlot:
    return ClipBufferLabels{
        .keyTimes = "animation_pose_secondary_key_times",
        .values = "animation_pose_secondary_values",
        .channels = "animation_pose_secondary_channels",
        .sampledNodes = "animation_pose_secondary_sampled_nodes",
        .sampledWeights = "animation_pose_secondary_sampled_weights",
    };
  default:
    NURI_ASSERT(false,
                "AnimationPoseSimulationBackend: clip slot %zu is invalid",
                clipSlot);
    return {};
  }
}

void advanceClipTime(AnimationPoseClipState &clipState,
                     const AnimationClipData &clip,
                     float deltaSeconds) noexcept {
  if (!clipState.playing || clip.durationSeconds <= 0.0f) {
    return;
  }
  clipState.timeSeconds = std::max(0.0f, clipState.timeSeconds + deltaSeconds);
  if (clipState.playbackMode == AnimationPosePlaybackMode::Loop) {
    clipState.timeSeconds =
        wrapLoopTime(clipState.timeSeconds, clip.durationSeconds);
    return;
  }
  if (clipState.timeSeconds >= clip.durationSeconds) {
    clipState.timeSeconds = clip.durationSeconds;
    clipState.playing = false;
  }
}

void destroyOwnedBuffer(GPUDevice &gpu, std::unique_ptr<Buffer> &buffer) {
  if (buffer && buffer->valid()) {
    gpu.destroyBuffer(buffer->handle());
  }
  buffer.reset();
}

struct AnimatedRenderableState {
  RenderableId renderableId = kInvalidRenderableId;
  uint32_t runtimeRenderableIndex = kInvalidIndex;
  uint32_t nodeIndex = kInvalidIndex;
  ModelRef model = kInvalidModelRef;
  BufferHandle sourceVertexBuffer{};
  uint64_t sourceVertexByteOffset = 0u;
  uint64_t sourceVertexAddress = 0u;
  BufferHandle skinInfluenceBuffer{};
  BufferHandle morphDeltaBuffer{};
  uint32_t vertexCount = 0u;
  uint32_t morphTargetCount = 0u;
  uint32_t jointNodeOffset = 0u;
  uint32_t jointCount = 0u;
  uint32_t paletteOffset = 0u;
  bool hasMorph = false;
  bool hasSkin = false;
  std::array<std::unique_ptr<Buffer>, kAnimationSceneFrameHistorySlots>
      morphOutputVertexBuffers;
  std::array<std::unique_ptr<Buffer>, kAnimationSceneFrameHistorySlots>
      finalOutputVertexBuffers;
};

struct AnimationClipGpuData {
  explicit AnimationClipGpuData(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : keyTimes(memory), values(memory), channels(memory) {}

  uint32_t clipIndex = kInvalidIndex;
  std::pmr::vector<float> keyTimes;
  std::pmr::vector<float> values;
  std::pmr::vector<AnimationChannelGpu> channels;
  std::unique_ptr<Buffer> keyTimesBuffer;
  std::unique_ptr<Buffer> valuesBuffer;
  std::unique_ptr<Buffer> channelsBuffer;
  std::unique_ptr<Buffer> sampledNodeStatesBuffer;
  std::unique_ptr<Buffer> sampledWeightsBuffer;
};

void destroyClipBuffers(GPUDevice &gpu, AnimationClipGpuData &clip) {
  destroyOwnedBuffer(gpu, clip.keyTimesBuffer);
  destroyOwnedBuffer(gpu, clip.valuesBuffer);
  destroyOwnedBuffer(gpu, clip.channelsBuffer);
  destroyOwnedBuffer(gpu, clip.sampledNodeStatesBuffer);
  destroyOwnedBuffer(gpu, clip.sampledWeightsBuffer);
}

struct AnimationPoseInstance {
  explicit AnimationPoseInstance(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : prefab(memory), instantiationMap(memory), baseNodeStates(memory),
        baseWeights(memory), nodeMeta(memory), depthOrderedNodes(memory),
        depthBucketStarts(memory), depthBucketCounts(memory),
        controlledNodeMask(memory), renderableBindings(memory),
        animatedRenderables(memory), flattenedJointNodeIndices(memory),
        flattenedInverseBindMatrices(memory),
        clips{AnimationClipGpuData(memory), AnimationClipGpuData(memory)} {}

  ScenePrefab prefab;
  SceneInstantiationMap instantiationMap;
  NodeId rootNode = kInvalidNodeId;
  AnimationPoseSimulationParams params{};
  uint64_t cachedSceneTopologyVersion = std::numeric_limits<uint64_t>::max();
  uint64_t cachedGeometryMutationVersion = std::numeric_limits<uint64_t>::max();

  std::pmr::vector<AnimationNodeStateGpu> baseNodeStates;
  std::pmr::vector<float> baseWeights;
  std::pmr::vector<AnimationNodeMetaGpu> nodeMeta;
  std::pmr::vector<uint32_t> depthOrderedNodes;
  std::pmr::vector<uint32_t> depthBucketStarts;
  std::pmr::vector<uint32_t> depthBucketCounts;
  std::pmr::vector<uint8_t> controlledNodeMask;
  std::pmr::vector<AnimationRenderableBindingGpu> renderableBindings;
  std::pmr::vector<AnimatedRenderableState> animatedRenderables;
  std::pmr::vector<uint32_t> flattenedJointNodeIndices;
  std::pmr::vector<glm::mat4> flattenedInverseBindMatrices;
  std::array<AnimationClipGpuData, kClipSlotCount> clips;

  std::unique_ptr<Buffer> nodeMetaBuffer;
  std::unique_ptr<Buffer> depthOrderedNodesBuffer;
  std::unique_ptr<Buffer> renderableBindingsBuffer;
  std::unique_ptr<Buffer> blendedNodeStatesBuffer;
  std::unique_ptr<Buffer> blendedWeightsBuffer;
  std::unique_ptr<Buffer> worldMatricesBuffer;
  std::unique_ptr<Buffer> jointNodeIndicesBuffer;
  std::unique_ptr<Buffer> inverseBindMatricesBuffer;
  std::unique_ptr<Buffer> jointPaletteBuffer;
};

struct SceneFrameState {
  explicit SceneFrameState(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : baseInstances(memory), geometryOverrides(memory),
        previousGeometryOverrides(memory), animatedRenderableIndices(memory),
        preDispatches(memory), samplePushConstants(memory),
        sampleDependencies(memory), blendPushConstants(memory),
        blendDependencies(memory), worldPushConstants(memory),
        worldDependencies(memory), scatterPushConstants(memory),
        scatterDependencies(memory), morphPushConstants(memory),
        morphDependencies(memory), skinPalettePushConstants(memory),
        skinPaletteDependencies(memory), skinPushConstants(memory),
        skinDependencies(memory), pendingBufferDeletes(memory) {}

  std::array<std::unique_ptr<Buffer>, kAnimationSceneFrameHistorySlots>
      instanceMatricesBuffers;
  std::array<size_t, kAnimationSceneFrameHistorySlots>
      instanceMatricesCapacityBytes{};
  std::pmr::vector<InstanceData> baseInstances;
  std::pmr::vector<AnimatedRenderableGeometryOverride> geometryOverrides;
  std::pmr::vector<AnimatedRenderableGeometryOverride>
      previousGeometryOverrides;
  std::pmr::vector<uint32_t> animatedRenderableIndices;
  std::pmr::vector<ComputeDispatchItem> preDispatches;
  std::pmr::vector<SamplePushConstants> samplePushConstants;
  std::pmr::vector<std::array<BufferHandle, 4>> sampleDependencies;
  std::pmr::vector<BlendPushConstants> blendPushConstants;
  std::pmr::vector<std::array<BufferHandle, 6>> blendDependencies;
  std::pmr::vector<WorldPushConstants> worldPushConstants;
  std::pmr::vector<std::array<BufferHandle, 4>> worldDependencies;
  std::pmr::vector<ScatterPushConstants> scatterPushConstants;
  std::pmr::vector<std::array<BufferHandle, 4>> scatterDependencies;
  std::pmr::vector<MorphPushConstants> morphPushConstants;
  std::pmr::vector<std::array<BufferHandle, 4>> morphDependencies;
  std::pmr::vector<SkinPalettePushConstants> skinPalettePushConstants;
  std::pmr::vector<std::array<BufferHandle, 4>> skinPaletteDependencies;
  std::pmr::vector<SkinPushConstants> skinPushConstants;
  std::pmr::vector<std::array<BufferHandle, 4>> skinDependencies;
  struct PendingBufferDelete {
    BufferHandle buffer{};
    uint64_t retireAfterFrame = 0u;
  };
  std::pmr::vector<PendingBufferDelete> pendingBufferDeletes;
  uint64_t version = 0u;
  uint64_t preparedFrameIndex = std::numeric_limits<uint64_t>::max();
  size_t currentHistorySlot = 0u;
  size_t previousHistorySlot = 0u;
  bool previousFrameValid = false;
  const RenderScene *scene = nullptr;
  uint64_t sceneTopologyVersion = 0u;
  size_t renderableCount = 0u;
  mutable AnimationSceneFrameData publishedData{};

  void resetTransientDispatchState() noexcept {
    preDispatches.clear();
    samplePushConstants.clear();
    sampleDependencies.clear();
    blendPushConstants.clear();
    blendDependencies.clear();
    worldPushConstants.clear();
    worldDependencies.clear();
    scatterPushConstants.clear();
    scatterDependencies.clear();
    morphPushConstants.clear();
    morphDependencies.clear();
    skinPalettePushConstants.clear();
    skinPaletteDependencies.clear();
    skinPushConstants.clear();
    skinDependencies.clear();
    animatedRenderableIndices.clear();
  }
};

[[nodiscard]] uint64_t bufferRetireLagFrames(const GPUDevice &gpu) noexcept {
  return static_cast<uint64_t>(std::max(1u, gpu.getSwapchainImageCount())) + 1u;
}

[[nodiscard]] uint64_t
retireAfterPreparedFrame(const GPUDevice &gpu,
                         const SceneFrameState &sceneFrame) noexcept {
  if (sceneFrame.preparedFrameIndex == std::numeric_limits<uint64_t>::max()) {
    return 0u;
  }
  return sceneFrame.preparedFrameIndex + bufferRetireLagFrames(gpu);
}

void destroyOwnedBuffer(GPUDevice &gpu, std::unique_ptr<Buffer> &buffer,
                        SceneFrameState *sceneFrame,
                        uint64_t retireAfterFrame) {
  if (buffer && buffer->valid()) {
    const BufferHandle handle = buffer->handle();
    if (sceneFrame != nullptr && retireAfterFrame != 0u) {
      sceneFrame->pendingBufferDeletes.push_back(
          SceneFrameState::PendingBufferDelete{
              .buffer = handle,
              .retireAfterFrame = retireAfterFrame,
          });
    } else {
      gpu.destroyBuffer(handle);
    }
  }
  buffer.reset();
}

void processPendingBufferDeletes(GPUDevice &gpu, SceneFrameState &sceneFrame,
                                 uint64_t frameIndex) {
  size_t writeIndex = 0u;
  for (size_t readIndex = 0u;
       readIndex < sceneFrame.pendingBufferDeletes.size(); ++readIndex) {
    const SceneFrameState::PendingBufferDelete pending =
        sceneFrame.pendingBufferDeletes[readIndex];
    if (frameIndex >= pending.retireAfterFrame) {
      if (nuri::isValid(pending.buffer)) {
        gpu.destroyBuffer(pending.buffer);
      }
      continue;
    }
    if (writeIndex != readIndex) {
      sceneFrame.pendingBufferDeletes[writeIndex] = pending;
    }
    ++writeIndex;
  }
  sceneFrame.pendingBufferDeletes.resize(writeIndex);
}

void destroyPendingBufferDeletes(GPUDevice &gpu, SceneFrameState &sceneFrame) {
  for (const SceneFrameState::PendingBufferDelete pending :
       sceneFrame.pendingBufferDeletes) {
    if (nuri::isValid(pending.buffer)) {
      gpu.destroyBuffer(pending.buffer);
    }
  }
  sceneFrame.pendingBufferDeletes.clear();
}

void invalidatePreparedSceneFrame(SceneFrameState &sceneFrame) noexcept {
  sceneFrame.resetTransientDispatchState();
  sceneFrame.geometryOverrides.clear();
  sceneFrame.previousGeometryOverrides.clear();
  sceneFrame.animatedRenderableIndices.clear();
  sceneFrame.publishedData = {};
  sceneFrame.preparedFrameIndex = std::numeric_limits<uint64_t>::max();
  sceneFrame.previousFrameValid = false;
  sceneFrame.scene = nullptr;
  sceneFrame.sceneTopologyVersion = 0u;
  sceneFrame.renderableCount = 0u;
}

void destroyAnimatedRenderableBuffers(
    GPUDevice &gpu,
    std::pmr::vector<AnimatedRenderableState> &animatedRenderables,
    SceneFrameState *sceneFrame = nullptr, uint64_t retireAfterFrame = 0u) {
  for (AnimatedRenderableState &renderable : animatedRenderables) {
    for (std::unique_ptr<Buffer> &buffer :
         renderable.morphOutputVertexBuffers) {
      destroyOwnedBuffer(gpu, buffer, sceneFrame, retireAfterFrame);
    }
    for (std::unique_ptr<Buffer> &buffer :
         renderable.finalOutputVertexBuffers) {
      destroyOwnedBuffer(gpu, buffer, sceneFrame, retireAfterFrame);
    }
  }
}

void destroyClipBuffers(GPUDevice &gpu, AnimationClipGpuData &clip,
                        SceneFrameState *sceneFrame,
                        uint64_t retireAfterFrame) {
  destroyOwnedBuffer(gpu, clip.keyTimesBuffer, sceneFrame, retireAfterFrame);
  destroyOwnedBuffer(gpu, clip.valuesBuffer, sceneFrame, retireAfterFrame);
  destroyOwnedBuffer(gpu, clip.channelsBuffer, sceneFrame, retireAfterFrame);
  destroyOwnedBuffer(gpu, clip.sampledNodeStatesBuffer, sceneFrame,
                     retireAfterFrame);
  destroyOwnedBuffer(gpu, clip.sampledWeightsBuffer, sceneFrame,
                     retireAfterFrame);
}

void destroyInstanceBuffers(GPUDevice &gpu, AnimationPoseInstance &instance,
                            SceneFrameState *sceneFrame = nullptr,
                            uint64_t retireAfterFrame = 0u) {
  destroyOwnedBuffer(gpu, instance.nodeMetaBuffer, sceneFrame,
                     retireAfterFrame);
  destroyOwnedBuffer(gpu, instance.depthOrderedNodesBuffer, sceneFrame,
                     retireAfterFrame);
  destroyOwnedBuffer(gpu, instance.renderableBindingsBuffer, sceneFrame,
                     retireAfterFrame);
  for (AnimationClipGpuData &clip : instance.clips) {
    destroyClipBuffers(gpu, clip, sceneFrame, retireAfterFrame);
  }
  destroyOwnedBuffer(gpu, instance.blendedNodeStatesBuffer, sceneFrame,
                     retireAfterFrame);
  destroyOwnedBuffer(gpu, instance.blendedWeightsBuffer, sceneFrame,
                     retireAfterFrame);
  destroyOwnedBuffer(gpu, instance.worldMatricesBuffer, sceneFrame,
                     retireAfterFrame);
  destroyOwnedBuffer(gpu, instance.jointNodeIndicesBuffer, sceneFrame,
                     retireAfterFrame);
  destroyOwnedBuffer(gpu, instance.inverseBindMatricesBuffer, sceneFrame,
                     retireAfterFrame);
  destroyOwnedBuffer(gpu, instance.jointPaletteBuffer, sceneFrame,
                     retireAfterFrame);
  destroyAnimatedRenderableBuffers(gpu, instance.animatedRenderables,
                                   sceneFrame, retireAfterFrame);
}

void destroyBindingBuffers(GPUDevice &gpu, AnimationPoseInstance &instance,
                           SceneFrameState *sceneFrame = nullptr,
                           uint64_t retireAfterFrame = 0u) {
  destroyOwnedBuffer(gpu, instance.renderableBindingsBuffer, sceneFrame,
                     retireAfterFrame);
  destroyOwnedBuffer(gpu, instance.jointNodeIndicesBuffer, sceneFrame,
                     retireAfterFrame);
  destroyOwnedBuffer(gpu, instance.inverseBindMatricesBuffer, sceneFrame,
                     retireAfterFrame);
  destroyOwnedBuffer(gpu, instance.jointPaletteBuffer, sceneFrame,
                     retireAfterFrame);
  destroyAnimatedRenderableBuffers(gpu, instance.animatedRenderables,
                                   sceneFrame, retireAfterFrame);
}

void destroySceneFrameBuffers(GPUDevice &gpu, SceneFrameState &sceneFrame,
                              uint64_t retireAfterFrame = 0u) {
  for (std::unique_ptr<Buffer> &buffer : sceneFrame.instanceMatricesBuffers) {
    destroyOwnedBuffer(gpu, buffer, &sceneFrame, retireAfterFrame);
  }
}

[[nodiscard]] bool controlsPrefabNode(const AnimationPoseInstance &instance,
                                      uint32_t nodeIndex) {
  return nodeIndex < instance.controlledNodeMask.size() &&
         instance.controlledNodeMask[nodeIndex] != 0u;
}

[[nodiscard]] bool
renderableControlledByInstance(const AnimationPoseInstance &instance,
                               const ScenePrefabRenderable &prefabRenderable) {
  if (controlsPrefabNode(instance, prefabRenderable.nodeIndex)) {
    return true;
  }
  if (prefabRenderable.skinIndex >= instance.prefab.skins.size()) {
    return false;
  }
  const SkinData &skin = instance.prefab.skins[prefabRenderable.skinIndex];
  return std::any_of(skin.jointNodeIndices.begin(), skin.jointNodeIndices.end(),
                     [&instance](uint32_t jointNodeIndex) {
                       return controlsPrefabNode(instance, jointNodeIndex);
                     });
}

Result<bool, std::string> buildBaseData(AnimationPoseInstance &instance) {
  const size_t nodeCount = instance.prefab.nodes.size();
  instance.baseNodeStates.clear();
  instance.baseWeights.clear();
  instance.nodeMeta.clear();
  instance.baseNodeStates.reserve(nodeCount);
  instance.nodeMeta.reserve(nodeCount);

  std::pmr::vector<uint32_t> depths(
      instance.prefab.nodes.get_allocator().resource());
  depths.resize(nodeCount, 0u);
  for (uint32_t nodeIndex = 0; nodeIndex < nodeCount; ++nodeIndex) {
    const ScenePrefabNode &node = instance.prefab.nodes[nodeIndex];
    instance.baseNodeStates.push_back(makeNodeStateGpu(node.localFromParent));
    const uint32_t weightOffset =
        static_cast<uint32_t>(instance.baseWeights.size());
    instance.baseWeights.insert(instance.baseWeights.end(),
                                node.morphWeights.begin(),
                                node.morphWeights.end());
    instance.nodeMeta.push_back(AnimationNodeMetaGpu{
        .parentIndex = node.parentIndex,
        .weightOffset = weightOffset,
        .weightCount = static_cast<uint32_t>(node.morphWeights.size()),
    });

    uint32_t depth = 0u;
    uint32_t current = node.parentIndex;
    size_t traversed = 0u;
    while (current != kInvalidScenePrefabIndex &&
           current < instance.prefab.nodes.size()) {
      if (traversed >= instance.prefab.nodes.size()) {
        NURI_LOG_ERROR(
            "AnimationPoseSimulationBackend: detected a parent cycle while "
            "computing depth for node %u",
            nodeIndex);
        depth = kInvalidIndex;
        break;
      }
      ++depth;
      ++traversed;
      current = instance.prefab.nodes[current].parentIndex;
    }
    depths[nodeIndex] = depth;
  }

  instance.depthOrderedNodes.clear();
  instance.depthBucketStarts.clear();
  instance.depthBucketCounts.clear();
  std::pmr::vector<std::pair<uint32_t, uint32_t>> order(
      instance.prefab.nodes.get_allocator().resource());
  order.reserve(nodeCount);
  for (uint32_t nodeIndex = 0; nodeIndex < nodeCount; ++nodeIndex) {
    order.emplace_back(depths[nodeIndex], nodeIndex);
  }
  std::stable_sort(order.begin(), order.end());
  uint32_t currentDepth = kInvalidIndex;
  for (const auto &[depth, nodeIndex] : order) {
    if (depth != currentDepth) {
      currentDepth = depth;
      instance.depthBucketStarts.push_back(
          static_cast<uint32_t>(instance.depthOrderedNodes.size()));
      instance.depthBucketCounts.push_back(0u);
    }
    instance.depthOrderedNodes.push_back(nodeIndex);
    ++instance.depthBucketCounts.back();
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> buildClipData(AnimationPoseInstance &instance,
                                        size_t clipSlot, uint32_t clipIndex) {
  if (instance.prefab.animations.empty()) {
    return Result<bool, std::string>::makeError(
        "AnimationPoseSimulationBackend: prefab has no animations");
  }
  if (clipSlot >= instance.clips.size() ||
      clipIndex >= instance.prefab.animations.size()) {
    return Result<bool, std::string>::makeError(
        "AnimationPoseSimulationBackend: clip selection is out of range");
  }

  AnimationClipGpuData &clipData = instance.clips[clipSlot];
  clipData.clipIndex = clipIndex;
  clipData.keyTimes.clear();
  clipData.values.clear();
  clipData.channels.clear();

  const AnimationClipData &clip = instance.prefab.animations[clipIndex];
  clipData.channels.reserve(clip.channels.size());
  for (const AnimationChannelData &channel : clip.channels) {
    if (channel.samplerIndex >= clip.samplers.size() ||
        channel.targetNodeIndex >= instance.nodeMeta.size() ||
        !controlsPrefabNode(instance, channel.targetNodeIndex)) {
      continue;
    }
    const AnimationSamplerData &sampler = clip.samplers[channel.samplerIndex];
    clipData.channels.push_back(AnimationChannelGpu{
        .keyOffset = static_cast<uint32_t>(clipData.keyTimes.size()),
        .valueOffset = static_cast<uint32_t>(clipData.values.size()),
        .keyCount = static_cast<uint32_t>(sampler.keyTimes.size()),
        .valueArity = sampler.valueArity,
        .targetNodeIndex = channel.targetNodeIndex,
        .path = static_cast<uint32_t>(channel.path),
        .interpolation = static_cast<uint32_t>(sampler.interpolation),
        .targetWeightOffset =
            instance.nodeMeta[channel.targetNodeIndex].weightOffset,
    });
    clipData.keyTimes.insert(clipData.keyTimes.end(), sampler.keyTimes.begin(),
                             sampler.keyTimes.end());
    clipData.values.insert(clipData.values.end(), sampler.values.begin(),
                           sampler.values.end());
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> createClipBuffers(
    AnimationGpuServices &services, const AnimationPoseInstance &instance,
    AnimationClipGpuData &clipData, const ClipBufferLabels &labels) {
  auto &gpu = services.gpu();
  destroyClipBuffers(gpu, clipData);

  auto keyTimesResult = services.createStorageBuffer(
      clipData.keyTimes.size() * sizeof(float), labels.keyTimes);
  if (keyTimesResult.hasError()) {
    return Result<bool, std::string>::makeError(keyTimesResult.error());
  }
  clipData.keyTimesBuffer = std::move(keyTimesResult.value());

  auto valuesResult = services.createStorageBuffer(
      clipData.values.size() * sizeof(float), labels.values);
  if (valuesResult.hasError()) {
    return Result<bool, std::string>::makeError(valuesResult.error());
  }
  clipData.valuesBuffer = std::move(valuesResult.value());

  auto channelResult = services.createStorageBuffer(
      clipData.channels.size() * sizeof(AnimationChannelGpu), labels.channels);
  if (channelResult.hasError()) {
    return Result<bool, std::string>::makeError(channelResult.error());
  }
  clipData.channelsBuffer = std::move(channelResult.value());

  auto nodeStatesResult = services.createStorageBuffer(
      instance.baseNodeStates.size() * sizeof(AnimationNodeStateGpu),
      labels.sampledNodes);
  if (nodeStatesResult.hasError()) {
    return Result<bool, std::string>::makeError(nodeStatesResult.error());
  }
  clipData.sampledNodeStatesBuffer = std::move(nodeStatesResult.value());

  auto sampledWeightsResult = services.createStorageBuffer(
      instance.baseWeights.size() * sizeof(float), labels.sampledWeights);
  if (sampledWeightsResult.hasError()) {
    return Result<bool, std::string>::makeError(sampledWeightsResult.error());
  }
  clipData.sampledWeightsBuffer = std::move(sampledWeightsResult.value());

  auto uploadKeyTimes =
      uploadVector(gpu, *clipData.keyTimesBuffer, clipData.keyTimes);
  if (uploadKeyTimes.hasError()) {
    return uploadKeyTimes;
  }
  auto uploadValues =
      uploadVector(gpu, *clipData.valuesBuffer, clipData.values);
  if (uploadValues.hasError()) {
    return uploadValues;
  }
  return uploadVector(gpu, *clipData.channelsBuffer, clipData.channels);
}

Result<bool, std::string> createStaticBuffers(
    AnimationGpuServices &services, AnimationPoseInstance &instance,
    SceneFrameState *sceneFrame = nullptr, uint64_t retireAfterFrame = 0u) {
  auto &gpu = services.gpu();
  destroyOwnedBuffer(gpu, instance.nodeMetaBuffer, sceneFrame,
                     retireAfterFrame);
  destroyOwnedBuffer(gpu, instance.depthOrderedNodesBuffer, sceneFrame,
                     retireAfterFrame);
  destroyOwnedBuffer(gpu, instance.blendedNodeStatesBuffer, sceneFrame,
                     retireAfterFrame);
  destroyOwnedBuffer(gpu, instance.blendedWeightsBuffer, sceneFrame,
                     retireAfterFrame);
  destroyOwnedBuffer(gpu, instance.worldMatricesBuffer, sceneFrame,
                     retireAfterFrame);
  for (AnimationClipGpuData &clip : instance.clips) {
    destroyClipBuffers(gpu, clip, sceneFrame, retireAfterFrame);
  }

  auto nodeMetaResult = services.createStorageBuffer(
      instance.nodeMeta.size() * sizeof(AnimationNodeMetaGpu),
      "animation_pose_node_meta");
  if (nodeMetaResult.hasError()) {
    return Result<bool, std::string>::makeError(nodeMetaResult.error());
  }
  instance.nodeMetaBuffer = std::move(nodeMetaResult.value());

  auto depthNodesResult = services.createStorageBuffer(
      instance.depthOrderedNodes.size() * sizeof(uint32_t),
      "animation_pose_depth_nodes");
  if (depthNodesResult.hasError()) {
    return Result<bool, std::string>::makeError(depthNodesResult.error());
  }
  instance.depthOrderedNodesBuffer = std::move(depthNodesResult.value());

  auto blendedNodesResult = services.createStorageBuffer(
      instance.baseNodeStates.size() * sizeof(AnimationNodeStateGpu),
      "animation_pose_blended_nodes");
  if (blendedNodesResult.hasError()) {
    return Result<bool, std::string>::makeError(blendedNodesResult.error());
  }
  instance.blendedNodeStatesBuffer = std::move(blendedNodesResult.value());

  auto blendedWeightsResult =
      services.createStorageBuffer(instance.baseWeights.size() * sizeof(float),
                                   "animation_pose_blended_weights");
  if (blendedWeightsResult.hasError()) {
    return Result<bool, std::string>::makeError(blendedWeightsResult.error());
  }
  instance.blendedWeightsBuffer = std::move(blendedWeightsResult.value());

  auto worldResult = services.createStorageBuffer(
      instance.baseNodeStates.size() * sizeof(glm::mat4),
      "animation_pose_world_matrices");
  if (worldResult.hasError()) {
    return Result<bool, std::string>::makeError(worldResult.error());
  }
  instance.worldMatricesBuffer = std::move(worldResult.value());

  auto uploadNodeMeta =
      uploadVector(gpu, *instance.nodeMetaBuffer, instance.nodeMeta);
  if (uploadNodeMeta.hasError()) {
    return uploadNodeMeta;
  }
  auto uploadDepthNodes = uploadVector(gpu, *instance.depthOrderedNodesBuffer,
                                       instance.depthOrderedNodes);
  if (uploadDepthNodes.hasError()) {
    return uploadDepthNodes;
  }

  auto createClipSlot =
      [&](size_t clipSlot,
          const AnimationPoseClipState &state) -> Result<bool, std::string> {
    AnimationClipGpuData &clipData = instance.clips[clipSlot];
    clipData.clipIndex = kInvalidIndex;
    clipData.keyTimes.clear();
    clipData.values.clear();
    clipData.channels.clear();
    if (!clipStateValid(instance.prefab, state)) {
      return Result<bool, std::string>::makeResult(true);
    }
    auto buildResult = buildClipData(instance, clipSlot, state.clipIndex);
    if (buildResult.hasError()) {
      return buildResult;
    }
    return createClipBuffers(services, instance, clipData,
                             clipBufferLabels(clipSlot));
  };

  auto primaryResult =
      createClipSlot(kPrimaryClipSlot, instance.params.primary);
  if (primaryResult.hasError()) {
    return primaryResult;
  }
  if (!blendEnabled(instance.prefab, instance.params)) {
    return Result<bool, std::string>::makeResult(true);
  }
  return createClipSlot(kSecondaryClipSlot, instance.params.secondary);
}

Result<bool, std::string> appendAnimatedRenderableBinding(
    const RenderScene &scene, const ResourceManager &resources,
    AnimationGpuServices &services, AnimationPoseInstance &instance,
    uint32_t prefabRenderableIndex, uint32_t &totalPaletteCount) {
  if (prefabRenderableIndex >= instance.instantiationMap.renderables.size()) {
    return Result<bool, std::string>::makeResult(true);
  }

  const RenderableId renderableId =
      instance.instantiationMap.renderables[prefabRenderableIndex];
  if (!isValid(renderableId)) {
    return Result<bool, std::string>::makeResult(true);
  }

  const std::optional<uint32_t> runtimeRenderableIndex =
      scene.findRenderableIndex(renderableId);
  if (!runtimeRenderableIndex.has_value()) {
    return Result<bool, std::string>::makeError(
        "AnimationPoseSimulationBackend: failed to resolve runtime renderable "
        "index");
  }

  const ScenePrefabRenderable &prefabRenderable =
      instance.prefab.renderables[prefabRenderableIndex];
  if (!renderableControlledByInstance(instance, prefabRenderable)) {
    return Result<bool, std::string>::makeResult(true);
  }

  const auto duplicateBindingIt = std::find_if(
      instance.renderableBindings.begin(), instance.renderableBindings.end(),
      [runtimeRenderableIndex](
          const AnimationRenderableBindingGpu &existingBinding) {
        return existingBinding.runtimeRenderableIndex ==
               *runtimeRenderableIndex;
      });
  if (duplicateBindingIt != instance.renderableBindings.end()) {
    return Result<bool, std::string>::makeError(
        "AnimationPoseSimulationBackend: duplicate runtime renderable index "
        "in scatter bindings");
  }

  AnimationRenderableBindingGpu binding{};
  binding.runtimeRenderableIndex = *runtimeRenderableIndex;
  binding.nodeIndex = prefabRenderable.nodeIndex;
  instance.renderableBindings.push_back(binding);

  ModelRef modelRef = kInvalidModelRef;
  if (!scene.graph().getRenderableModel(renderableId, modelRef) ||
      !isValid(modelRef)) {
    return Result<bool, std::string>::makeResult(true);
  }

  const ModelRecord *modelRecord = resources.tryGet(modelRef);
  if (modelRecord == nullptr || modelRecord->model == nullptr) {
    return Result<bool, std::string>::makeResult(true);
  }

  const Model &model = *modelRecord->model;
  const Model::ModelAnimationGpuView &animationView = model.animationGpuView();
  GeometryAllocationView geometry{};
  if (!services.gpu().resolveGeometry(model.geometryHandle(), geometry)) {
    return Result<bool, std::string>::makeError(
        "AnimationPoseSimulationBackend: failed to resolve geometry");
  }

  const uint64_t sourceVertexAddress = services.gpu().getBufferDeviceAddress(
      geometry.vertexBuffer, geometry.vertexByteOffset);
  if (sourceVertexAddress == 0u) {
    return Result<bool, std::string>::makeError(
        "AnimationPoseSimulationBackend: invalid geometry vertex address");
  }

  const bool hasMorphWeights =
      prefabRenderable.nodeIndex < instance.nodeMeta.size() &&
      instance.nodeMeta[prefabRenderable.nodeIndex].weightCount > 0u;
  const bool hasMorph = hasMorphWeights &&
                        isValid(animationView.morphDeltaBuffer) &&
                        animationView.morphTargetCount > 0u;

  const SkinData *skin = nullptr;
  bool hasSkin = false;
  if (prefabRenderable.skinIndex < instance.prefab.skins.size()) {
    skin = &instance.prefab.skins[prefabRenderable.skinIndex];
    hasSkin = isValid(animationView.skinInfluenceBuffer) &&
              animationView.skinInfluenceCount > 0u &&
              !skin->jointNodeIndices.empty();
    if (!hasSkin) {
      NURI_LOG_WARNING(
          "AnimationPoseSimulationBackend: renderable %u references skin %u "
          "but GPU skin influences are unavailable; using non-skinned geometry",
          prefabRenderableIndex, prefabRenderable.skinIndex);
    }
  }

  if (!hasMorph && !hasSkin) {
    return Result<bool, std::string>::makeResult(true);
  }

  AnimatedRenderableState animated{};
  animated.renderableId = renderableId;
  animated.runtimeRenderableIndex = *runtimeRenderableIndex;
  animated.nodeIndex = prefabRenderable.nodeIndex;
  animated.model = modelRef;
  animated.sourceVertexBuffer = geometry.vertexBuffer;
  animated.sourceVertexByteOffset = geometry.vertexByteOffset;
  animated.sourceVertexAddress = sourceVertexAddress;
  animated.skinInfluenceBuffer = animationView.skinInfluenceBuffer;
  animated.morphDeltaBuffer = animationView.morphDeltaBuffer;
  animated.vertexCount = model.vertexCount();
  animated.morphTargetCount = animationView.morphTargetCount;
  animated.hasMorph = hasMorph;
  animated.hasSkin = hasSkin;

  const size_t requiredBytes =
      static_cast<size_t>(animated.vertexCount) * kPackedVertexStrideBytes;
  if (animated.hasMorph) {
    for (size_t slot = 0u; slot < kAnimationSceneFrameHistorySlots; ++slot) {
      auto morphOutputResult = services.createStorageVertexBuffer(
          requiredBytes, "animation_pose_morph_output");
      if (morphOutputResult.hasError()) {
        return Result<bool, std::string>::makeError(morphOutputResult.error());
      }
      animated.morphOutputVertexBuffers[slot] =
          std::move(morphOutputResult.value());
    }
  }

  if (animated.hasSkin) {
    if (skin->jointNodeIndices.size() != skin->inverseBindMatrices.size()) {
      return Result<bool, std::string>::makeError(
          "AnimationPoseSimulationBackend: skin jointNodeIndices and "
          "inverseBindMatrices counts do not match");
    }
    for (size_t slot = 0u; slot < kAnimationSceneFrameHistorySlots; ++slot) {
      auto finalOutputResult = services.createStorageVertexBuffer(
          requiredBytes, "animation_pose_skin_output");
      if (finalOutputResult.hasError()) {
        return Result<bool, std::string>::makeError(finalOutputResult.error());
      }
      animated.finalOutputVertexBuffers[slot] =
          std::move(finalOutputResult.value());
    }
    animated.jointNodeOffset =
        static_cast<uint32_t>(instance.flattenedJointNodeIndices.size());
    animated.jointCount = static_cast<uint32_t>(skin->jointNodeIndices.size());
    animated.paletteOffset = totalPaletteCount;
    instance.flattenedJointNodeIndices.insert(
        instance.flattenedJointNodeIndices.end(),
        skin->jointNodeIndices.begin(), skin->jointNodeIndices.end());
    instance.flattenedInverseBindMatrices.insert(
        instance.flattenedInverseBindMatrices.end(),
        skin->inverseBindMatrices.begin(), skin->inverseBindMatrices.end());
    totalPaletteCount += animated.jointCount;
  }

  instance.animatedRenderables.push_back(std::move(animated));
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> createRenderableBindingBuffers(
    AnimationGpuServices &services, const RenderScene &scene,
    AnimationPoseInstance &instance, uint32_t totalPaletteCount) {
  auto bindingBufferResult =
      services.createStorageBuffer(instance.renderableBindings.size() *
                                       sizeof(AnimationRenderableBindingGpu),
                                   "animation_pose_renderable_bindings");
  if (bindingBufferResult.hasError()) {
    return Result<bool, std::string>::makeError(bindingBufferResult.error());
  }
  instance.renderableBindingsBuffer = std::move(bindingBufferResult.value());

  if (!instance.flattenedJointNodeIndices.empty()) {
    auto jointNodeBufferResult = services.createStorageBuffer(
        instance.flattenedJointNodeIndices.size() * sizeof(uint32_t),
        "animation_pose_skin_joint_nodes");
    if (jointNodeBufferResult.hasError()) {
      return Result<bool, std::string>::makeError(
          jointNodeBufferResult.error());
    }
    instance.jointNodeIndicesBuffer = std::move(jointNodeBufferResult.value());

    auto inverseBindResult = services.createStorageBuffer(
        instance.flattenedInverseBindMatrices.size() * sizeof(glm::mat4),
        "animation_pose_skin_inverse_bind");
    if (inverseBindResult.hasError()) {
      return Result<bool, std::string>::makeError(inverseBindResult.error());
    }
    instance.inverseBindMatricesBuffer = std::move(inverseBindResult.value());

    auto paletteResult = services.createStorageBuffer(
        static_cast<size_t>(totalPaletteCount) * sizeof(glm::mat4),
        "animation_pose_skin_palette");
    if (paletteResult.hasError()) {
      return Result<bool, std::string>::makeError(paletteResult.error());
    }
    instance.jointPaletteBuffer = std::move(paletteResult.value());
  }

  instance.cachedSceneTopologyVersion = scene.topologyVersion();
  instance.cachedGeometryMutationVersion =
      services.gpu().geometryMutationVersion();

  auto uploadBindings =
      uploadVector(services.gpu(), *instance.renderableBindingsBuffer,
                   instance.renderableBindings);
  if (uploadBindings.hasError()) {
    return uploadBindings;
  }
  if (instance.jointNodeIndicesBuffer != nullptr) {
    auto uploadJointNodes =
        uploadVector(services.gpu(), *instance.jointNodeIndicesBuffer,
                     instance.flattenedJointNodeIndices);
    if (uploadJointNodes.hasError()) {
      return uploadJointNodes;
    }
  }
  if (instance.inverseBindMatricesBuffer != nullptr) {
    return uploadVector(services.gpu(), *instance.inverseBindMatricesBuffer,
                        instance.flattenedInverseBindMatrices);
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
ensureRenderableBindings(SceneRuntimeHost &host, AnimationGpuServices &services,
                         AnimationPoseInstance &instance,
                         SceneFrameState *sceneFrame = nullptr,
                         uint64_t retireAfterFrame = 0u) {
  const RenderScene *scene = host.scene();
  if (scene == nullptr) {
    return Result<bool, std::string>::makeError(
        "AnimationPoseSimulationBackend: host scene is null");
  }
  const ResourceManager *resources = scene->resources();
  if (resources == nullptr) {
    return Result<bool, std::string>::makeError(
        "AnimationPoseSimulationBackend: render scene resources are null");
  }
  destroyBindingBuffers(services.gpu(), instance, sceneFrame, retireAfterFrame);
  instance.renderableBindings.clear();
  instance.animatedRenderables.clear();
  instance.flattenedJointNodeIndices.clear();
  instance.flattenedInverseBindMatrices.clear();
  instance.renderableBindings.reserve(instance.prefab.renderables.size());
  instance.animatedRenderables.reserve(instance.prefab.renderables.size());

  uint32_t totalPaletteCount = 0u;
  for (uint32_t prefabRenderableIndex = 0u;
       prefabRenderableIndex < instance.prefab.renderables.size();
       ++prefabRenderableIndex) {
    auto appendResult = appendAnimatedRenderableBinding(
        *scene, *resources, services, instance, prefabRenderableIndex,
        totalPaletteCount);
    if (appendResult.hasError()) {
      return appendResult;
    }
  }
  return createRenderableBindingBuffers(services, *scene, instance,
                                        totalPaletteCount);
}

Result<bool, std::string> ensureSceneFrameBuffer(AnimationGpuServices &services,
                                                 const RenderScene &scene,
                                                 SceneFrameState &frameState,
                                                 size_t currentHistorySlot) {
  const size_t requiredBytes = std::max(
      scene.renderables().size() * sizeof(InstanceData), sizeof(InstanceData));
  if (currentHistorySlot >= frameState.instanceMatricesBuffers.size()) {
    return Result<bool, std::string>::makeError(
        "AnimationPoseSimulationBackend: invalid scene frame history slot");
  }
  for (size_t slot = 0u; slot < frameState.instanceMatricesBuffers.size();
       ++slot) {
    if (frameState.instanceMatricesBuffers[slot] &&
        frameState.instanceMatricesCapacityBytes[slot] >= requiredBytes) {
      continue;
    }
    destroyOwnedBuffer(services.gpu(), frameState.instanceMatricesBuffers[slot],
                       &frameState,
                       retireAfterPreparedFrame(services.gpu(), frameState));
    auto bufferResult = services.createStorageBuffer(
        requiredBytes, "animation_scene_instances_" + std::to_string(slot));
    if (bufferResult.hasError()) {
      return Result<bool, std::string>::makeError(bufferResult.error());
    }
    frameState.instanceMatricesBuffers[slot] = std::move(bufferResult.value());
    frameState.instanceMatricesCapacityBytes[slot] = requiredBytes;
  }

  frameState.baseInstances.clear();
  frameState.geometryOverrides.clear();
  frameState.previousGeometryOverrides.clear();
  frameState.animatedRenderableIndices.clear();
  frameState.baseInstances.reserve(scene.renderables().size());
  frameState.geometryOverrides.resize(scene.renderables().size());
  frameState.previousGeometryOverrides.resize(scene.renderables().size());
  for (const Renderable &renderable : scene.renderables()) {
    frameState.baseInstances.push_back(
        makeInstanceData(renderable.modelMatrix));
  }

  auto uploadResult = uploadVector(
      services.gpu(), *frameState.instanceMatricesBuffers[currentHistorySlot],
      frameState.baseInstances);
  if (uploadResult.hasError()) {
    return uploadResult;
  }
  return Result<bool, std::string>::makeResult(true);
}

} // namespace

struct AnimationPoseSimulationBackend::Impl {
  explicit Impl(std::pmr::memory_resource *memory)
      : instances(memory), sceneFrame(memory) {}

  PmrHashMap<SimulationHandle, AnimationPoseInstance> instances;
  SceneFrameState sceneFrame;
  uint64_t sceneFrameVersionCounter = 0u;
};

AnimationPoseSimulationBackend::AnimationPoseSimulationBackend(
    std::pmr::memory_resource *memory)
    : memory_(memory != nullptr ? memory : std::pmr::get_default_resource()),
      impl_(std::make_unique<Impl>(memory_)) {}

AnimationPoseSimulationBackend::~AnimationPoseSimulationBackend() = default;
AnimationPoseSimulationBackend::AnimationPoseSimulationBackend(
    AnimationPoseSimulationBackend &&) noexcept = default;
AnimationPoseSimulationBackend &AnimationPoseSimulationBackend::operator=(
    AnimationPoseSimulationBackend &&) noexcept = default;

Result<bool, std::string>
AnimationPoseSimulationBackend::createInstance(SceneRuntimeHost &host,
                                               SimulationHandle handle,
                                               const SimulationDesc &desc) {
  if (services_ == nullptr) {
    return Result<bool, std::string>::makeError(
        "AnimationPoseSimulationBackend: animation GPU services are not "
        "attached");
  }
  auto initResult = services_->ensureInitialized();
  if (initResult.hasError()) {
    return Result<bool, std::string>::makeError(initResult.error());
  }

  auto paramsResult = decodeAnimationPoseSimulationParams(desc.initialParams);
  if (paramsResult.hasError()) {
    return Result<bool, std::string>::makeError(paramsResult.error());
  }
  AnimationPoseSimulationParams params = paramsResult.value();
  sanitizeAnimationPoseSimulationParams(params);
  if (!host.pendingAnimationPoseCreatePayload_.has_value()) {
    return Result<bool, std::string>::makeError(
        "AnimationPoseSimulationBackend: missing transient create payload; use "
        "SceneRuntimeHost::createAnimationPoseSimulation");
  }
  const AnimationPoseSimulationCreatePayload &payload =
      *host.pendingAnimationPoseCreatePayload_;
  if (payload.prefab == nullptr || payload.instantiationMap == nullptr) {
    return Result<bool, std::string>::makeError(
        "AnimationPoseSimulationBackend: transient create payload pointers are "
        "null");
  }

  AnimationPoseInstance instance(memory_);
  instance.prefab = *payload.prefab;
  instance.instantiationMap = *payload.instantiationMap;
  instance.rootNode = desc.binding.primaryTarget.prefabRoot;
  instance.params = params;
  instance.controlledNodeMask.assign(instance.prefab.nodes.size(), uint8_t{0u});
  if (payload.controlledPrefabNodeIndices.empty()) {
    std::fill(instance.controlledNodeMask.begin(),
              instance.controlledNodeMask.end(), uint8_t{1u});
  } else {
    for (const uint32_t nodeIndex : payload.controlledPrefabNodeIndices) {
      if (nodeIndex < instance.controlledNodeMask.size()) {
        instance.controlledNodeMask[nodeIndex] = 1u;
      }
    }
  }
  auto validateResult =
      validateAnimationPoseSimulationParams(instance.prefab, params);
  if (validateResult.hasError()) {
    return Result<bool, std::string>::makeError(validateResult.error());
  }
  auto baseResult = buildBaseData(instance);
  if (baseResult.hasError()) {
    return baseResult;
  }
  auto staticBufferResult = createStaticBuffers(*services_, instance);
  if (staticBufferResult.hasError()) {
    return staticBufferResult;
  }
  auto bindingResult = ensureRenderableBindings(host, *services_, instance);
  if (bindingResult.hasError()) {
    return bindingResult;
  }

  impl_->instances.insert_or_assign(handle, std::move(instance));
  invalidatePreparedSceneFrame(impl_->sceneFrame);
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
AnimationPoseSimulationBackend::destroyInstance(SceneRuntimeHost &,
                                                SimulationHandle handle) {
  if (impl_ == nullptr) {
    return Result<bool, std::string>::makeResult(false);
  }
  auto it = impl_->instances.find(handle);
  if (it == impl_->instances.end()) {
    return Result<bool, std::string>::makeResult(false);
  }
  if (services_ != nullptr) {
    const uint64_t retireAfterFrame =
        retireAfterPreparedFrame(services_->gpu(), impl_->sceneFrame);
    destroyInstanceBuffers(services_->gpu(), it->second, &impl_->sceneFrame,
                           retireAfterFrame);
  }
  impl_->instances.erase(it);
  if (impl_->instances.empty()) {
    if (services_ != nullptr) {
      destroySceneFrameBuffers(
          services_->gpu(), impl_->sceneFrame,
          retireAfterPreparedFrame(services_->gpu(), impl_->sceneFrame));
    }
    impl_->sceneFrame.publishedData = {};
    impl_->sceneFrame.version = 0u;
    impl_->sceneFrame.preparedFrameIndex = std::numeric_limits<uint64_t>::max();
    impl_->sceneFrame.previousFrameValid = false;
    impl_->sceneFrame.scene = nullptr;
    impl_->sceneFrame.sceneTopologyVersion = 0u;
    impl_->sceneFrame.renderableCount = 0u;
  } else {
    invalidatePreparedSceneFrame(impl_->sceneFrame);
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> AnimationPoseSimulationBackend::updateParams(
    SceneRuntimeHost &, SimulationHandle handle,
    std::span<const std::byte> params) {
  if (impl_ == nullptr) {
    return Result<bool, std::string>::makeError(
        "AnimationPoseSimulationBackend: backend state is null");
  }
  auto it = impl_->instances.find(handle);
  if (it == impl_->instances.end()) {
    return Result<bool, std::string>::makeError(
        "AnimationPoseSimulationBackend: instance handle is invalid");
  }
  if (services_ == nullptr) {
    return Result<bool, std::string>::makeError(
        "AnimationPoseSimulationBackend: services_ is null");
  }
  auto decodedResult = decodeAnimationPoseSimulationParams(params);
  if (decodedResult.hasError()) {
    return Result<bool, std::string>::makeError(decodedResult.error());
  }
  AnimationPoseSimulationParams decoded = decodedResult.value();
  sanitizeAnimationPoseSimulationParams(decoded);
  auto validateResult =
      validateAnimationPoseSimulationParams(it->second.prefab, decoded);
  if (validateResult.hasError()) {
    return Result<bool, std::string>::makeError(validateResult.error());
  }
  const bool primaryClipChanged =
      decoded.primary.clipIndex != it->second.params.primary.clipIndex;
  const bool wasBlendEnabled =
      blendEnabled(it->second.prefab, it->second.params);
  const bool isBlendEnabledNow = blendEnabled(it->second.prefab, decoded);
  const bool secondaryClipChanged =
      decoded.secondary.clipIndex != it->second.params.secondary.clipIndex;
  it->second.params = decoded;
  if (primaryClipChanged || wasBlendEnabled != isBlendEnabledNow ||
      (isBlendEnabledNow && secondaryClipChanged)) {
    auto recreateResult = createStaticBuffers(
        *services_, it->second, &impl_->sceneFrame,
        retireAfterPreparedFrame(services_->gpu(), impl_->sceneFrame));
    if (recreateResult.hasError()) {
      return recreateResult;
    }
  }
  invalidatePreparedSceneFrame(impl_->sceneFrame);
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> AnimationPoseSimulationBackend::executePhase(
    SceneRuntimeHost &host, SimulationHandle handle, SimulationPhase phase,
    const SimulationExecutionContext &context) {
  if (phase != SimulationPhase::Finalize) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (services_ == nullptr || impl_ == nullptr || host.scene() == nullptr) {
    return Result<bool, std::string>::makeError(
        "AnimationPoseSimulationBackend: missing required runtime state");
  }

  auto initResult = services_->ensureInitialized();
  if (initResult.hasError()) {
    return Result<bool, std::string>::makeError(initResult.error());
  }

  auto it = impl_->instances.find(handle);
  if (it == impl_->instances.end()) {
    return Result<bool, std::string>::makeError(
        "AnimationPoseSimulationBackend: instance handle is invalid");
  }
  AnimationPoseInstance &instance = it->second;
  if (clipStateValid(instance.prefab, instance.params.primary)) {
    advanceClipTime(
        instance.params.primary,
        instance.prefab.animations[instance.params.primary.clipIndex],
        static_cast<float>(context.effectiveDeltaSeconds));
  }
  if (clipStateValid(instance.prefab, instance.params.secondary)) {
    advanceClipTime(
        instance.params.secondary,
        instance.prefab.animations[instance.params.secondary.clipIndex],
        static_cast<float>(context.effectiveDeltaSeconds));
  }
  return Result<bool, std::string>::makeResult(true);
}

namespace {

bool isPlaybackRunning(const SimulationRegistry::Record &record,
                       const AnimationPoseClipState &clipState) noexcept {
  return record.enabled && !record.faulted &&
         record.state == SimulationState::Running && clipState.playing;
}

float computeRenderSampleTime(const SceneRuntimeHost &host,
                              const SimulationRegistry::Record &record,
                              const AnimationPoseClipState &clipState,
                              const AnimationClipData &clip) noexcept {
  float sampleTime = clipState.timeSeconds;
  if (!isPlaybackRunning(record, clipState) || clip.durationSeconds <= 0.0f) {
    return sampleTime;
  }
  const double accumulatorSeconds =
      std::max(0.0, host.remainingAccumulatorSeconds());
  sampleTime +=
      static_cast<float>(accumulatorSeconds * std::max(record.timeScale, 0.0f));
  if (clipState.playbackMode == AnimationPosePlaybackMode::Loop) {
    return wrapLoopTime(sampleTime, clip.durationSeconds);
  }
  return std::min(sampleTime, clip.durationSeconds);
}

float computeNormalizedPhase(float sampleTime,
                             const AnimationClipData &clip) noexcept {
  if (!(clip.durationSeconds > 0.0f)) {
    return 0.0f;
  }
  return glm::clamp(sampleTime / clip.durationSeconds, 0.0f, 1.0f);
}

} // namespace

Result<bool, std::string>
AnimationPoseSimulationBackend::prepareSceneFrame(SceneRuntimeHost &host,
                                                  uint64_t frameIndex) {
  NURI_PROFILER_FUNCTION();
  if (impl_ == nullptr || services_ == nullptr || host.scene() == nullptr) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (impl_->sceneFrame.preparedFrameIndex == frameIndex) {
    return Result<bool, std::string>::makeResult(true);
  }

  auto initResult = services_->ensureInitialized();
  if (initResult.hasError()) {
    return Result<bool, std::string>::makeError(initResult.error());
  }

  SceneFrameState &sceneFrame = impl_->sceneFrame;
  processPendingBufferDeletes(services_->gpu(), sceneFrame, frameIndex);
  sceneFrame.resetTransientDispatchState();

  if (impl_->instances.empty()) {
    sceneFrame.geometryOverrides.clear();
    sceneFrame.previousGeometryOverrides.clear();
    sceneFrame.publishedData = {};
    sceneFrame.version = 0u;
    sceneFrame.preparedFrameIndex = frameIndex;
    sceneFrame.previousFrameValid = false;
    sceneFrame.scene = nullptr;
    sceneFrame.sceneTopologyVersion = 0u;
    sceneFrame.renderableCount = 0u;
    return Result<bool, std::string>::makeResult(true);
  }

  const RenderScene &scene = *host.scene();
  const bool previousFrameValid =
      sceneFrame.preparedFrameIndex != std::numeric_limits<uint64_t>::max() &&
      sceneFrame.preparedFrameIndex + 1u == frameIndex &&
      sceneFrame.scene == &scene &&
      sceneFrame.sceneTopologyVersion == scene.topologyVersion() &&
      sceneFrame.renderableCount == scene.renderables().size();
  const size_t previousHistorySlot = sceneFrame.currentHistorySlot;
  const size_t currentHistorySlot =
      static_cast<size_t>(frameIndex % kAnimationSceneFrameHistorySlots);
  auto frameResult =
      ensureSceneFrameBuffer(*services_, scene, sceneFrame, currentHistorySlot);
  if (frameResult.hasError()) {
    return frameResult;
  }
  sceneFrame.currentHistorySlot = currentHistorySlot;
  sceneFrame.previousHistorySlot =
      previousFrameValid ? previousHistorySlot : currentHistorySlot;
  sceneFrame.previousFrameValid =
      previousFrameValid &&
      sceneFrame.previousHistorySlot != currentHistorySlot;
  sceneFrame.scene = &scene;
  sceneFrame.sceneTopologyVersion = scene.topologyVersion();
  sceneFrame.renderableCount = scene.renderables().size();

  size_t sampleDispatchCount = 0u;
  size_t blendDispatchCount = 0u;
  size_t worldDispatchCount = 0u;
  size_t scatterDispatchCount = 0u;
  size_t animatedRenderableIndexCount = 0u;
  size_t morphDispatchCount = 0u;
  size_t skinPaletteDispatchCount = 0u;
  size_t skinDispatchCount = 0u;
  for (auto &[handle, instance] : impl_->instances) {
    (void)handle;
    if (scene.topologyVersion() != instance.cachedSceneTopologyVersion ||
        services_->gpu().geometryMutationVersion() !=
            instance.cachedGeometryMutationVersion) {
      auto bindingResult = ensureRenderableBindings(
          host, *services_, instance, &sceneFrame,
          retireAfterPreparedFrame(services_->gpu(), sceneFrame));
      if (bindingResult.hasError()) {
        return bindingResult;
      }
    }
    const AnimationClipGpuData &primaryClip = instance.clips[kPrimaryClipSlot];
    const bool instanceBlendEnabled =
        blendEnabled(instance.prefab, instance.params);
    const bool useSecondaryOnly =
        secondaryOnlyBlendEnabled(instance.prefab, instance.params);
    if (!useSecondaryOnly && !primaryClip.channels.empty()) {
      ++sampleDispatchCount;
    }
    if (instanceBlendEnabled) {
      const AnimationClipGpuData &secondaryClip =
          instance.clips[kSecondaryClipSlot];
      if (!secondaryClip.channels.empty()) {
        ++sampleDispatchCount;
      }
      if (!useSecondaryOnly) {
        ++blendDispatchCount;
      }
    }
    worldDispatchCount += instance.depthBucketStarts.size();
    if (!instance.renderableBindings.empty()) {
      ++scatterDispatchCount;
      animatedRenderableIndexCount += instance.renderableBindings.size();
    }
    for (const AnimatedRenderableState &animated :
         instance.animatedRenderables) {
      morphDispatchCount += animated.hasMorph ? 1u : 0u;
      skinPaletteDispatchCount += animated.hasSkin ? 1u : 0u;
      skinDispatchCount += animated.hasSkin ? 1u : 0u;
    }
  }

  sceneFrame.preDispatches.reserve(
      sampleDispatchCount + blendDispatchCount + worldDispatchCount +
      scatterDispatchCount + morphDispatchCount + skinPaletteDispatchCount +
      skinDispatchCount);
  sceneFrame.samplePushConstants.reserve(sampleDispatchCount);
  sceneFrame.sampleDependencies.reserve(sampleDispatchCount);
  sceneFrame.blendPushConstants.reserve(blendDispatchCount);
  sceneFrame.blendDependencies.reserve(blendDispatchCount);
  sceneFrame.worldPushConstants.reserve(worldDispatchCount);
  sceneFrame.worldDependencies.reserve(worldDispatchCount);
  sceneFrame.scatterPushConstants.reserve(scatterDispatchCount);
  sceneFrame.scatterDependencies.reserve(scatterDispatchCount);
  sceneFrame.animatedRenderableIndices.reserve(animatedRenderableIndexCount);
  sceneFrame.morphPushConstants.reserve(morphDispatchCount);
  sceneFrame.morphDependencies.reserve(morphDispatchCount);
  sceneFrame.skinPalettePushConstants.reserve(skinPaletteDispatchCount);
  sceneFrame.skinPaletteDependencies.reserve(skinPaletteDispatchCount);
  sceneFrame.skinPushConstants.reserve(skinDispatchCount);
  sceneFrame.skinDependencies.reserve(skinDispatchCount);

  const uint64_t instanceMatricesAddress =
      services_->gpu().getBufferDeviceAddress(
          sceneFrame.instanceMatricesBuffers[sceneFrame.currentHistorySlot]
              ->handle());
  if (instanceMatricesAddress == 0u) {
    return Result<bool, std::string>::makeError(
        "AnimationPoseSimulationBackend: invalid scene frame buffer address");
  }

  for (auto &[handle, instance] : impl_->instances) {
    const SimulationRegistry::Record *record = host.registry_.tryGet(handle);
    if (record == nullptr || record->faulted) {
      continue;
    }
    const AnimationClipGpuData &primaryClipGpu =
        instance.clips[kPrimaryClipSlot];
    NURI_ASSERT(primaryClipGpu.sampledNodeStatesBuffer != nullptr &&
                    primaryClipGpu.sampledWeightsBuffer != nullptr,
                "AnimationPoseSimulationBackend: primary clip buffers are "
                "missing");
    const AnimationClipData &primaryClip =
        instance.prefab.animations[instance.params.primary.clipIndex];
    const float primarySampleTime = computeRenderSampleTime(
        host, *record, instance.params.primary, primaryClip);
    const bool instanceBlendEnabled =
        blendEnabled(instance.prefab, instance.params);
    const bool useSecondaryOnly =
        secondaryOnlyBlendEnabled(instance.prefab, instance.params);

    if (!useSecondaryOnly) {
      auto uploadPrimaryNodeStates = uploadVector(
          services_->gpu(), *primaryClipGpu.sampledNodeStatesBuffer,
          instance.baseNodeStates);
      if (uploadPrimaryNodeStates.hasError()) {
        return uploadPrimaryNodeStates;
      }
      auto uploadPrimaryWeights =
          uploadVector(services_->gpu(), *primaryClipGpu.sampledWeightsBuffer,
                       instance.baseWeights);
      if (uploadPrimaryWeights.hasError()) {
        return uploadPrimaryWeights;
      }
    }

    const AnimationClipGpuData *secondaryClipGpu = nullptr;
    const AnimationClipData *secondaryClip = nullptr;
    float secondarySampleTime = 0.0f;
    if (instanceBlendEnabled) {
      secondaryClipGpu = &instance.clips[kSecondaryClipSlot];
      NURI_ASSERT(secondaryClipGpu->sampledNodeStatesBuffer != nullptr &&
                      secondaryClipGpu->sampledWeightsBuffer != nullptr,
                  "AnimationPoseSimulationBackend: secondary clip buffers are "
                  "missing");
      secondaryClip =
          &instance.prefab.animations[instance.params.secondary.clipIndex];
      secondarySampleTime = computeRenderSampleTime(
          host, *record, instance.params.secondary, *secondaryClip);
      if (instance.params.blendSyncMode ==
          AnimationPoseBlendSyncMode::NormalizedTime) {
        secondarySampleTime =
            computeNormalizedPhase(primarySampleTime, primaryClip) *
            std::max(secondaryClip->durationSeconds, 0.0f);
      }

      auto uploadSecondaryNodeStates = uploadVector(
          services_->gpu(), *secondaryClipGpu->sampledNodeStatesBuffer,
          instance.baseNodeStates);
      if (uploadSecondaryNodeStates.hasError()) {
        return uploadSecondaryNodeStates;
      }
      auto uploadSecondaryWeights = uploadVector(
          services_->gpu(), *secondaryClipGpu->sampledWeightsBuffer,
          instance.baseWeights);
      if (uploadSecondaryWeights.hasError()) {
        return uploadSecondaryWeights;
      }
    }

    uint64_t primaryNodeStatesAddress = 0u;
    uint64_t primaryWeightsAddress = 0u;
    if (!useSecondaryOnly) {
      primaryNodeStatesAddress = services_->gpu().getBufferDeviceAddress(
          primaryClipGpu.sampledNodeStatesBuffer->handle());
      primaryWeightsAddress = services_->gpu().getBufferDeviceAddress(
          primaryClipGpu.sampledWeightsBuffer->handle());
      if (primaryNodeStatesAddress == 0u || primaryWeightsAddress == 0u) {
        return Result<bool, std::string>::makeError(
            "AnimationPoseSimulationBackend: invalid primary sample buffer "
            "address");
      }
    }

    glm::mat4 rootTransform(1.0f);
    (void)scene.graph().getCachedNodeWorldTransform(instance.rootNode,
                                                    rootTransform);
    const uint64_t worldMatricesAddress =
        services_->gpu().getBufferDeviceAddress(
            instance.worldMatricesBuffer->handle());
    if (worldMatricesAddress == 0u) {
      return Result<bool, std::string>::makeError(
          "AnimationPoseSimulationBackend: invalid dynamic buffer address");
    }

    auto appendSampleDispatch =
        [&](const AnimationClipGpuData &clipGpu,
            float sampleTimeSeconds) -> Result<bool, std::string> {
      if (clipGpu.channels.empty()) {
        return Result<bool, std::string>::makeResult(true);
      }
      sceneFrame.samplePushConstants.push_back(SamplePushConstants{
          .channelsAddress = services_->gpu().getBufferDeviceAddress(
              clipGpu.channelsBuffer->handle()),
          .keyTimesAddress = services_->gpu().getBufferDeviceAddress(
              clipGpu.keyTimesBuffer->handle()),
          .valuesAddress = services_->gpu().getBufferDeviceAddress(
              clipGpu.valuesBuffer->handle()),
          .nodeStatesAddress = services_->gpu().getBufferDeviceAddress(
              clipGpu.sampledNodeStatesBuffer->handle()),
          .sampledWeightsAddress = services_->gpu().getBufferDeviceAddress(
              clipGpu.sampledWeightsBuffer->handle()),
          .sampleTimeSeconds = sampleTimeSeconds,
          .channelCount = static_cast<uint32_t>(clipGpu.channels.size()),
      });
      sceneFrame.sampleDependencies.push_back(
          {clipGpu.channelsBuffer->handle(), clipGpu.keyTimesBuffer->handle(),
           clipGpu.sampledNodeStatesBuffer->handle(),
           clipGpu.sampledWeightsBuffer->handle()});
      appendComputeDispatch(
          sceneFrame.preDispatches, services_->samplePipeline(),
          dispatchCount(sceneFrame.samplePushConstants.back().channelCount),
          sceneFrame.samplePushConstants.back(),
          sceneFrame.sampleDependencies.back(),
          sceneFrame.sampleDependencies.back().size(), "AnimationPose Sample",
          0xff5599ffu);
      return Result<bool, std::string>::makeResult(true);
    };

    if (!useSecondaryOnly) {
      auto primarySampleResult =
          appendSampleDispatch(primaryClipGpu, primarySampleTime);
      if (primarySampleResult.hasError()) {
        return primarySampleResult;
      }
    }

    uint64_t nodeStatesAddress = primaryNodeStatesAddress;
    uint64_t sampledWeightsAddress = primaryWeightsAddress;
    BufferHandle nodeStatesBuffer =
        !useSecondaryOnly ? primaryClipGpu.sampledNodeStatesBuffer->handle()
                          : BufferHandle{};
    BufferHandle sampledWeightsBuffer =
        !useSecondaryOnly ? primaryClipGpu.sampledWeightsBuffer->handle()
                          : BufferHandle{};

    if (instanceBlendEnabled && secondaryClipGpu != nullptr &&
        secondaryClip != nullptr) {
      auto secondarySampleResult =
          appendSampleDispatch(*secondaryClipGpu, secondarySampleTime);
      if (secondarySampleResult.hasError()) {
        return secondarySampleResult;
      }

      const uint64_t secondaryNodeStatesAddress =
          services_->gpu().getBufferDeviceAddress(
              secondaryClipGpu->sampledNodeStatesBuffer->handle());
      const uint64_t secondaryWeightsAddress =
          services_->gpu().getBufferDeviceAddress(
              secondaryClipGpu->sampledWeightsBuffer->handle());
      const uint64_t blendedNodeStatesAddress =
          services_->gpu().getBufferDeviceAddress(
              instance.blendedNodeStatesBuffer->handle());
      const uint64_t blendedWeightsAddress =
          services_->gpu().getBufferDeviceAddress(
              instance.blendedWeightsBuffer->handle());
      if (secondaryNodeStatesAddress == 0u || secondaryWeightsAddress == 0u) {
        return Result<bool, std::string>::makeError(
            "AnimationPoseSimulationBackend: invalid secondary sample buffer "
            "address");
      }

      if (useSecondaryOnly) {
        nodeStatesAddress = secondaryNodeStatesAddress;
        sampledWeightsAddress = secondaryWeightsAddress;
        nodeStatesBuffer = secondaryClipGpu->sampledNodeStatesBuffer->handle();
        sampledWeightsBuffer = secondaryClipGpu->sampledWeightsBuffer->handle();
      } else {
        if (blendedNodeStatesAddress == 0u || blendedWeightsAddress == 0u) {
          return Result<bool, std::string>::makeError(
              "AnimationPoseSimulationBackend: invalid blend buffer address");
        }
        sceneFrame.blendPushConstants.push_back(BlendPushConstants{
            .sourceNodeStatesAddressA = primaryNodeStatesAddress,
            .sourceNodeStatesAddressB = secondaryNodeStatesAddress,
            .outputNodeStatesAddress = blendedNodeStatesAddress,
            .sourceWeightsAddressA = primaryWeightsAddress,
            .sourceWeightsAddressB = secondaryWeightsAddress,
            .outputWeightsAddress = blendedWeightsAddress,
            .blendWeight = instance.params.blendWeight,
            .nodeCount = static_cast<uint32_t>(instance.baseNodeStates.size()),
            .weightCount = static_cast<uint32_t>(instance.baseWeights.size()),
        });
        sceneFrame.blendDependencies.push_back(
            {primaryClipGpu.sampledNodeStatesBuffer->handle(),
             secondaryClipGpu->sampledNodeStatesBuffer->handle(),
             instance.blendedNodeStatesBuffer->handle(),
             primaryClipGpu.sampledWeightsBuffer->handle(),
             secondaryClipGpu->sampledWeightsBuffer->handle(),
             instance.blendedWeightsBuffer->handle()});
        appendComputeDispatch(
            sceneFrame.preDispatches, services_->blendPipeline(),
            dispatchCount(
                std::max(sceneFrame.blendPushConstants.back().nodeCount,
                         sceneFrame.blendPushConstants.back().weightCount)),
            sceneFrame.blendPushConstants.back(),
            sceneFrame.blendDependencies.back(),
            sceneFrame.blendDependencies.back().size(), "AnimationPose Blend",
            0xff6688ffu);

        nodeStatesAddress = blendedNodeStatesAddress;
        sampledWeightsAddress = blendedWeightsAddress;
        nodeStatesBuffer = instance.blendedNodeStatesBuffer->handle();
        sampledWeightsBuffer = instance.blendedWeightsBuffer->handle();
      }
    }

    for (size_t depthBucketIndex = 0;
         depthBucketIndex < instance.depthBucketStarts.size();
         ++depthBucketIndex) {
      const uint32_t nodeCount = instance.depthBucketCounts[depthBucketIndex];
      if (nodeCount == 0u) {
        continue;
      }
      sceneFrame.worldPushConstants.push_back(WorldPushConstants{
          .nodeStatesAddress = nodeStatesAddress,
          .nodeMetaAddress = services_->gpu().getBufferDeviceAddress(
              instance.nodeMetaBuffer->handle()),
          .depthOrderedNodesAddress = services_->gpu().getBufferDeviceAddress(
              instance.depthOrderedNodesBuffer->handle()),
          .worldMatricesAddress = worldMatricesAddress,
          .rootTransform = rootTransform,
          .nodeStart = instance.depthBucketStarts[depthBucketIndex],
          .nodeCount = nodeCount,
      });
      sceneFrame.worldDependencies.push_back(
          {nodeStatesBuffer, instance.nodeMetaBuffer->handle(),
           instance.depthOrderedNodesBuffer->handle(),
           instance.worldMatricesBuffer->handle()});
      appendComputeDispatch(
          sceneFrame.preDispatches, services_->worldPipeline(),
          dispatchCount(nodeCount), sceneFrame.worldPushConstants.back(),
          sceneFrame.worldDependencies.back(),
          sceneFrame.worldDependencies.back().size(), "AnimationPose World",
          0xff33aa55u);
    }

    if (!instance.renderableBindings.empty()) {
      for (const AnimationRenderableBindingGpu &binding :
           instance.renderableBindings) {
        sceneFrame.animatedRenderableIndices.push_back(
            binding.runtimeRenderableIndex);
      }
      sceneFrame.scatterPushConstants.push_back(ScatterPushConstants{
          .renderableBindingsAddress = services_->gpu().getBufferDeviceAddress(
              instance.renderableBindingsBuffer->handle()),
          .worldMatricesAddress = worldMatricesAddress,
          .instanceMatricesAddress = instanceMatricesAddress,
          .bindingCount =
              static_cast<uint32_t>(instance.renderableBindings.size()),
      });
      sceneFrame.scatterDependencies.push_back(
          {instance.renderableBindingsBuffer->handle(),
           instance.worldMatricesBuffer->handle(),
           sceneFrame.instanceMatricesBuffers[sceneFrame.currentHistorySlot]
               ->handle(),
           BufferHandle{}});
      appendComputeDispatch(
          sceneFrame.preDispatches, services_->scatterPipeline(),
          dispatchCount(sceneFrame.scatterPushConstants.back().bindingCount),
          sceneFrame.scatterPushConstants.back(),
          sceneFrame.scatterDependencies.back(), size_t{3u},
          "AnimationPose Scatter", 0xff33cc88u);
    }

    for (AnimatedRenderableState &animated : instance.animatedRenderables) {
      BufferHandle overrideBuffer{};
      BufferHandle previousOverrideBuffer{};
      Buffer *currentMorphOutput =
          animated.hasMorph
              ? animated.morphOutputVertexBuffers[sceneFrame.currentHistorySlot]
                    .get()
              : nullptr;
      Buffer *previousMorphOutput =
          sceneFrame.previousFrameValid && animated.hasMorph
              ? animated
                    .morphOutputVertexBuffers[sceneFrame.previousHistorySlot]
                    .get()
              : nullptr;
      Buffer *currentSkinOutput =
          animated.hasSkin
              ? animated.finalOutputVertexBuffers[sceneFrame.currentHistorySlot]
                    .get()
              : nullptr;
      Buffer *previousSkinOutput =
          sceneFrame.previousFrameValid && animated.hasSkin
              ? animated
                    .finalOutputVertexBuffers[sceneFrame.previousHistorySlot]
                    .get()
              : nullptr;
      if (previousSkinOutput != nullptr && previousSkinOutput->valid()) {
        previousOverrideBuffer = previousSkinOutput->handle();
      } else if (previousMorphOutput != nullptr &&
                 previousMorphOutput->valid()) {
        previousOverrideBuffer = previousMorphOutput->handle();
      }

      if (animated.hasSkin && instance.jointNodeIndicesBuffer &&
          instance.inverseBindMatricesBuffer && instance.jointPaletteBuffer &&
          animated.jointCount > 0u) {
        const uint64_t jointNodeAddress =
            services_->gpu().getBufferDeviceAddress(
                instance.jointNodeIndicesBuffer->handle(),
                static_cast<size_t>(animated.jointNodeOffset) *
                    sizeof(uint32_t));
        const uint64_t inverseBindAddress =
            services_->gpu().getBufferDeviceAddress(
                instance.inverseBindMatricesBuffer->handle(),
                static_cast<size_t>(animated.jointNodeOffset) *
                    sizeof(glm::mat4));
        const uint64_t paletteAddress = services_->gpu().getBufferDeviceAddress(
            instance.jointPaletteBuffer->handle(),
            static_cast<size_t>(animated.paletteOffset) * sizeof(glm::mat4));
        if (jointNodeAddress != 0u && inverseBindAddress != 0u &&
            paletteAddress != 0u) {
          sceneFrame.skinPalettePushConstants.push_back(
              SkinPalettePushConstants{
                  .worldMatricesAddress = worldMatricesAddress,
                  .jointNodeIndicesAddress = jointNodeAddress,
                  .inverseBindMatricesAddress = inverseBindAddress,
                  .paletteAddress = paletteAddress,
                  .renderableNodeIndex = animated.nodeIndex,
                  .jointCount = animated.jointCount,
              });
          sceneFrame.skinPaletteDependencies.push_back(
              {instance.worldMatricesBuffer->handle(),
               instance.jointNodeIndicesBuffer->handle(),
               instance.inverseBindMatricesBuffer->handle(),
               instance.jointPaletteBuffer->handle()});
          appendComputeDispatch(
              sceneFrame.preDispatches, services_->skinPalettePipeline(),
              dispatchCount(animated.jointCount),
              sceneFrame.skinPalettePushConstants.back(),
              sceneFrame.skinPaletteDependencies.back(),
              sceneFrame.skinPaletteDependencies.back().size(),
              "AnimationPose SkinPalette", 0xffaa55ccu);
        }
      }

      if (animated.hasMorph && currentMorphOutput != nullptr &&
          currentMorphOutput->valid() &&
          animated.nodeIndex < instance.nodeMeta.size()) {
        const uint64_t outputVertexAddress =
            services_->gpu().getBufferDeviceAddress(
                currentMorphOutput->handle());
        const uint64_t morphDeltaAddress =
            services_->gpu().getBufferDeviceAddress(animated.morphDeltaBuffer);
        if (outputVertexAddress != 0u && morphDeltaAddress != 0u &&
            animated.sourceVertexAddress != 0u) {
          sceneFrame.morphPushConstants.push_back(MorphPushConstants{
              .sourceVertexAddress = animated.sourceVertexAddress,
              .outputVertexAddress = outputVertexAddress,
              .morphDeltaAddress = morphDeltaAddress,
              .weightAddress = sampledWeightsAddress,
              .weightOffset =
                  instance.nodeMeta[animated.nodeIndex].weightOffset,
              .weightCount = instance.nodeMeta[animated.nodeIndex].weightCount,
              .morphTargetCount = animated.morphTargetCount,
              .vertexCount = animated.vertexCount,
          });
          sceneFrame.morphDependencies.push_back(
              {animated.sourceVertexBuffer, animated.morphDeltaBuffer,
               sampledWeightsBuffer, currentMorphOutput->handle()});
          appendComputeDispatch(sceneFrame.preDispatches,
                                services_->morphPipeline(),
                                dispatchCount(animated.vertexCount),
                                sceneFrame.morphPushConstants.back(),
                                sceneFrame.morphDependencies.back(),
                                sceneFrame.morphDependencies.back().size(),
                                "AnimationPose Morph", 0xffaa7733u);
          if (!animated.hasSkin) {
            overrideBuffer = currentMorphOutput->handle();
          }
        }
      }

      if (animated.hasSkin && currentSkinOutput != nullptr &&
          currentSkinOutput->valid()) {
        const uint64_t skinSourceAddress =
            animated.hasMorph && currentMorphOutput != nullptr
                ? services_->gpu().getBufferDeviceAddress(
                      currentMorphOutput->handle())
                : animated.sourceVertexAddress;
        const BufferHandle skinSourceBuffer =
            animated.hasMorph && currentMorphOutput != nullptr
                ? currentMorphOutput->handle()
                : animated.sourceVertexBuffer;
        const uint64_t outputVertexAddress =
            services_->gpu().getBufferDeviceAddress(
                currentSkinOutput->handle());
        const uint64_t skinInfluenceAddress =
            services_->gpu().getBufferDeviceAddress(
                animated.skinInfluenceBuffer);
        const uint64_t paletteAddress =
            instance.jointPaletteBuffer
                ? services_->gpu().getBufferDeviceAddress(
                      instance.jointPaletteBuffer->handle(),
                      static_cast<size_t>(animated.paletteOffset) *
                          sizeof(glm::mat4))
                : 0u;
        if (skinSourceAddress != 0u && outputVertexAddress != 0u &&
            skinInfluenceAddress != 0u && paletteAddress != 0u) {
          sceneFrame.skinPushConstants.push_back(SkinPushConstants{
              .sourceVertexAddress = skinSourceAddress,
              .outputVertexAddress = outputVertexAddress,
              .skinInfluenceAddress = skinInfluenceAddress,
              .paletteAddress = paletteAddress,
              .vertexCount = animated.vertexCount,
          });
          sceneFrame.skinDependencies.push_back(
              {skinSourceBuffer, animated.skinInfluenceBuffer,
               instance.jointPaletteBuffer->handle(),
               currentSkinOutput->handle()});
          appendComputeDispatch(sceneFrame.preDispatches,
                                services_->skinPipeline(),
                                dispatchCount(animated.vertexCount),
                                sceneFrame.skinPushConstants.back(),
                                sceneFrame.skinDependencies.back(),
                                sceneFrame.skinDependencies.back().size(),
                                "AnimationPose Skin", 0xffcc8844u);
          overrideBuffer = currentSkinOutput->handle();
        }
      }

      if (animated.runtimeRenderableIndex <
          sceneFrame.geometryOverrides.size()) {
        if (nuri::isValid(overrideBuffer)) {
          sceneFrame.geometryOverrides[animated.runtimeRenderableIndex] =
              AnimatedRenderableGeometryOverride{
                  .vertexBuffer = overrideBuffer,
                  .vertexByteOffset = 0u,
                  .vertexCount = animated.vertexCount,
                  .enabled = true,
              };
        }
        if (sceneFrame.previousFrameValid &&
            nuri::isValid(previousOverrideBuffer) &&
            animated.runtimeRenderableIndex <
                sceneFrame.previousGeometryOverrides.size()) {
          sceneFrame
              .previousGeometryOverrides[animated.runtimeRenderableIndex] =
              AnimatedRenderableGeometryOverride{
                  .vertexBuffer = previousOverrideBuffer,
                  .vertexByteOffset = 0u,
                  .vertexCount = animated.vertexCount,
                  .enabled = true,
              };
        }
      }
    }
  }

  sceneFrame.version = ++impl_->sceneFrameVersionCounter;
  sceneFrame.preparedFrameIndex = frameIndex;
  return Result<bool, std::string>::makeResult(true);
}

void AnimationPoseSimulationBackend::reset() {
  if (impl_ != nullptr) {
    if (services_ != nullptr) {
      for (auto &[handle, instance] : impl_->instances) {
        (void)handle;
        destroyInstanceBuffers(services_->gpu(), instance);
      }
      destroySceneFrameBuffers(services_->gpu(), impl_->sceneFrame);
      destroyPendingBufferDeletes(services_->gpu(), impl_->sceneFrame);
    }
    impl_->instances.clear();
    impl_->sceneFrame = SceneFrameState(memory_);
  }
}

const AnimationSceneFrameData *
AnimationPoseSimulationBackend::currentSceneFrameData() const noexcept {
  if (impl_ == nullptr || impl_->sceneFrame.version == 0u ||
      impl_->instances.empty() || services_ == nullptr) {
    return nullptr;
  }
  const SceneFrameState &sceneFrame = impl_->sceneFrame;
  if (sceneFrame.currentHistorySlot >=
          sceneFrame.instanceMatricesBuffers.size() ||
      !sceneFrame.instanceMatricesBuffers[sceneFrame.currentHistorySlot] ||
      !sceneFrame.instanceMatricesBuffers[sceneFrame.currentHistorySlot]
           ->valid()) {
    return nullptr;
  }
  AnimationSceneFrameData &frameData = impl_->sceneFrame.publishedData;
  frameData.instanceMatricesBuffer =
      sceneFrame.instanceMatricesBuffers[sceneFrame.currentHistorySlot]
          ->handle();
  frameData.instanceMatricesAddress =
      services_->gpu().getBufferDeviceAddress(frameData.instanceMatricesBuffer);
  frameData.previousInstanceMatricesBuffer = {};
  frameData.previousInstanceMatricesAddress = 0u;
  if (sceneFrame.previousFrameValid &&
      sceneFrame.previousHistorySlot <
          sceneFrame.instanceMatricesBuffers.size() &&
      sceneFrame.instanceMatricesBuffers[sceneFrame.previousHistorySlot] &&
      sceneFrame.instanceMatricesBuffers[sceneFrame.previousHistorySlot]
          ->valid()) {
    frameData.previousInstanceMatricesBuffer =
        sceneFrame.instanceMatricesBuffers[sceneFrame.previousHistorySlot]
            ->handle();
    frameData.previousInstanceMatricesAddress =
        services_->gpu().getBufferDeviceAddress(
            frameData.previousInstanceMatricesBuffer);
  }
  frameData.preDispatches = std::span<const ComputeDispatchItem>(
      impl_->sceneFrame.preDispatches.data(),
      impl_->sceneFrame.preDispatches.size());
  frameData.geometryOverridesByRenderable =
      std::span<const AnimatedRenderableGeometryOverride>(
          impl_->sceneFrame.geometryOverrides.data(),
          impl_->sceneFrame.geometryOverrides.size());
  if (sceneFrame.previousFrameValid) {
    frameData.previousGeometryOverridesByRenderable =
        std::span<const AnimatedRenderableGeometryOverride>(
            impl_->sceneFrame.previousGeometryOverrides.data(),
            impl_->sceneFrame.previousGeometryOverrides.size());
  } else {
    frameData.previousGeometryOverridesByRenderable = {};
  }
  frameData.animatedRenderableIndices = std::span<const uint32_t>(
      impl_->sceneFrame.animatedRenderableIndices.data(),
      impl_->sceneFrame.animatedRenderableIndices.size());
  frameData.scene = impl_->sceneFrame.scene;
  frameData.sceneTopologyVersion = impl_->sceneFrame.sceneTopologyVersion;
  frameData.renderableCount = impl_->sceneFrame.renderableCount;
  frameData.version = impl_->sceneFrame.version;
  return frameData.instanceMatricesAddress != 0u ? &frameData : nullptr;
}

} // namespace nuri
