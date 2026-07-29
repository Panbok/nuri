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
#include <array>
#include <glm/gtx/matrix_decompose.hpp>
#include <optional>
#include <variant>
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
constexpr RenderGraphAccessMode kRead = RenderGraphAccessMode::Read;
constexpr RenderGraphAccessMode kWrite = RenderGraphAccessMode::Write;
constexpr std::array kSampleDependencyAccess = {kRead, kRead, kWrite, kWrite};
constexpr std::array kBlendDependencyAccess = {kRead, kRead, kWrite,
                                               kRead, kRead, kWrite};
constexpr std::array kWorldDependencyAccess = {kRead, kRead, kRead, kWrite};
constexpr std::array kScatterDependencyAccess = {kRead, kRead, kWrite};
constexpr std::array kSkinPaletteDependencyAccess = {kRead, kRead, kRead,
                                                     kWrite};
constexpr std::array kMorphDependencyAccess = {kRead, kRead, kRead, kWrite};
constexpr std::array kSkinDependencyAccess = {kRead, kRead, kRead, kWrite};
template <typename T, size_t N>
void appendComputeDispatch(std::pmr::vector<ComputeDispatchItem> &out,
                           ComputePipelineHandle pipeline, uint32_t x,
                           const T &pushConstants,
                           const std::array<BufferHandle, N> &dependencies,
                           size_t dependencyCount,
                           std::span<const RenderGraphAccessMode> accessModes,
                           std::string_view debugLabel, uint32_t debugColor) {
  out.push_back(ComputeDispatchItem{
      .pipeline = pipeline,
      .dispatch = {.x = x, .y = 1u, .z = 1u},
      .pushConstants =
          std::as_bytes(std::span<const T>(&pushConstants, size_t{1u})),
      .dependencyBuffers =
          std::span<const BufferHandle>(dependencies.data(), dependencyCount),
      .dependencyBufferAccessModes = accessModes,
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
struct BufferAllocation {
  std::unique_ptr<Buffer> *buffer;
  size_t bytes;
  std::string_view label;
};
Result<bool, std::string>
allocateStorageBuffers(AnimationGpuServices &services,
                       std::span<const BufferAllocation> allocations) {
  for (const BufferAllocation &allocation : allocations) {
    auto result =
        services.createStorageBuffer(allocation.bytes, allocation.label);
    if (result.hasError()) {
      return Result<bool, std::string>::makeError(result.error());
    }
    *allocation.buffer = std::move(result.value());
  }
  return Result<bool, std::string>::makeResult(true);
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
void resetClipBuffers(AnimationClipGpuData &clip) {
  clip.keyTimesBuffer.reset();
  clip.valuesBuffer.reset();
  clip.channelsBuffer.reset();
  clip.sampledNodeStatesBuffer.reset();
  clip.sampledWeightsBuffer.reset();
}
struct AnimationPoseInstance {
  AnimationPoseInstance(
      std::shared_ptr<const ScenePrefab> prefabOwnerIn,
      std::shared_ptr<const SceneInstantiationMap> instantiationOwnerIn,
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : prefabOwner(std::move(prefabOwnerIn)),
        instantiationOwner(std::move(instantiationOwnerIn)),
        prefab(*prefabOwner), instantiationMap(*instantiationOwner),
        baseNodeStates(memory), baseWeights(memory), nodeMeta(memory),
        depthOrderedNodes(memory), depthBucketStarts(memory),
        depthBucketCounts(memory), controlledNodeMask(memory),
        renderableBindings(memory), animatedRenderables(memory),
        flattenedJointNodeIndices(memory), flattenedInverseBindMatrices(memory),
        clips{AnimationClipGpuData(memory), AnimationClipGpuData(memory)} {}
  AnimationPoseInstance(AnimationPoseInstance &&) noexcept = default;
  AnimationPoseInstance &operator=(AnimationPoseInstance &&other) noexcept {
    if (this != &other) {
      std::destroy_at(this);
      std::construct_at(this, std::move(other));
    }
    return *this;
  }
  std::shared_ptr<const ScenePrefab> prefabOwner;
  std::shared_ptr<const SceneInstantiationMap> instantiationOwner;
  const ScenePrefab &prefab;
  const SceneInstantiationMap &instantiationMap;
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
template <typename PushConstants, size_t DependencyCount>
struct AnimationDispatchRecord {
  PushConstants push{};
  std::array<BufferHandle, DependencyCount> dependencies{};
};
using AnimationDispatch =
    std::variant<AnimationDispatchRecord<SamplePushConstants, 4>,
                 AnimationDispatchRecord<BlendPushConstants, 6>,
                 AnimationDispatchRecord<WorldPushConstants, 4>,
                 AnimationDispatchRecord<ScatterPushConstants, 4>,
                 AnimationDispatchRecord<MorphPushConstants, 4>,
                 AnimationDispatchRecord<SkinPalettePushConstants, 4>,
                 AnimationDispatchRecord<SkinPushConstants, 4>>;

template <typename Record>
Record &appendAnimationDispatch(std::pmr::vector<AnimationDispatch> &records,
                                Record record) {
  records.emplace_back(std::move(record));
  return std::get<Record>(records.back());
}
struct SceneFrameState {
  explicit SceneFrameState(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : baseInstances(memory), geometryOverrides(memory),
        previousGeometryOverrides(memory), animatedRenderableIndices(memory),
        preDispatches(memory), dispatchRecords(memory) {}
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
  std::pmr::vector<AnimationDispatch> dispatchRecords;
  uint64_t version = 0u;
  uint64_t preparedFrameIndex = std::numeric_limits<uint64_t>::max();
  size_t currentHistorySlot = 0u;
  size_t previousHistorySlot = 0u;
  bool previousFrameValid = false;
  const RenderScene *scene = nullptr;
  uint64_t sceneTopologyVersion = 0u;
  size_t renderableCount = 0u;
  size_t committedHistorySlot = 0u;
  const RenderScene *committedScene = nullptr;
  uint64_t committedSceneTopologyVersion = 0u;
  size_t committedRenderableCount = 0u;
  mutable AnimationSceneFrameData publishedData{};
  void resetTransientDispatchState() noexcept {
    preDispatches.clear();
    dispatchRecords.clear();
    animatedRenderableIndices.clear();
  }
};
void invalidatePreparedSceneFrame(SceneFrameState &sceneFrame) noexcept {
  sceneFrame.resetTransientDispatchState();
  sceneFrame.geometryOverrides.clear();
  sceneFrame.previousGeometryOverrides.clear();
  sceneFrame.publishedData = {};
  sceneFrame.preparedFrameIndex = std::numeric_limits<uint64_t>::max();
  sceneFrame.previousFrameValid = false;
  sceneFrame.scene = nullptr;
  sceneFrame.sceneTopologyVersion = 0u;
  sceneFrame.renderableCount = 0u;
  sceneFrame.committedHistorySlot = 0u;
  sceneFrame.committedScene = nullptr;
  sceneFrame.committedSceneTopologyVersion = 0u;
  sceneFrame.committedRenderableCount = 0u;
}
void resetBindingBuffers(AnimationPoseInstance &instance) {
  instance.renderableBindingsBuffer.reset();
  instance.jointNodeIndicesBuffer.reset();
  instance.inverseBindMatricesBuffer.reset();
  instance.jointPaletteBuffer.reset();
}
[[nodiscard]] bool controlsPrefabNode(const AnimationPoseInstance &instance,
                                      uint32_t nodeIndex) {
  return instance.controlledNodeMask[nodeIndex] != 0u;
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
void buildBaseData(AnimationPoseInstance &instance) {
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
    while (current != kInvalidScenePrefabIndex) {
      ++depth;
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
}
void buildClipData(AnimationPoseInstance &instance, size_t clipSlot,
                   uint32_t clipIndex) {
  AnimationClipGpuData &clipData = instance.clips[clipSlot];
  clipData.clipIndex = clipIndex;
  clipData.keyTimes.clear();
  clipData.values.clear();
  clipData.channels.clear();
  const AnimationClipData &clip = instance.prefab.animations[clipIndex];
  clipData.channels.reserve(clip.channels.size());
  for (const AnimationChannelData &channel : clip.channels) {
    if (!controlsPrefabNode(instance, channel.targetNodeIndex)) {
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
}
Result<bool, std::string> createClipBuffers(
    AnimationGpuServices &services, const AnimationPoseInstance &instance,
    AnimationClipGpuData &clipData, const ClipBufferLabels &labels) {
  auto &gpu = services.gpu();
  resetClipBuffers(clipData);
  const std::array allocations{
      BufferAllocation{&clipData.keyTimesBuffer,
                       clipData.keyTimes.size() * sizeof(float),
                       labels.keyTimes},
      BufferAllocation{&clipData.valuesBuffer,
                       clipData.values.size() * sizeof(float), labels.values},
      BufferAllocation{&clipData.channelsBuffer,
                       clipData.channels.size() * sizeof(AnimationChannelGpu),
                       labels.channels},
      BufferAllocation{&clipData.sampledNodeStatesBuffer,
                       instance.baseNodeStates.size() *
                           sizeof(AnimationNodeStateGpu),
                       labels.sampledNodes},
      BufferAllocation{&clipData.sampledWeightsBuffer,
                       instance.baseWeights.size() * sizeof(float),
                       labels.sampledWeights},
  };
  auto allocation = allocateStorageBuffers(services, allocations);
  if (allocation.hasError()) {
    return allocation;
  }
  const std::array updates{
      BufferUpdate{.buffer = clipData.keyTimesBuffer->handle(),
                   .data = std::as_bytes(std::span(clipData.keyTimes))},
      BufferUpdate{.buffer = clipData.valuesBuffer->handle(),
                   .data = std::as_bytes(std::span(clipData.values))},
      BufferUpdate{.buffer = clipData.channelsBuffer->handle(),
                   .data = std::as_bytes(std::span(clipData.channels))},
  };
  return gpu.updateBuffers(updates);
}
Result<bool, std::string> createStaticBuffers(AnimationGpuServices &services,
                                              AnimationPoseInstance &instance) {
  auto &gpu = services.gpu();
  instance.nodeMetaBuffer.reset();
  instance.depthOrderedNodesBuffer.reset();
  instance.blendedNodeStatesBuffer.reset();
  instance.blendedWeightsBuffer.reset();
  instance.worldMatricesBuffer.reset();
  for (AnimationClipGpuData &clip : instance.clips) {
    resetClipBuffers(clip);
  }
  const std::array allocations{
      BufferAllocation{&instance.nodeMetaBuffer,
                       instance.nodeMeta.size() * sizeof(AnimationNodeMetaGpu),
                       "animation_pose_node_meta"},
      BufferAllocation{&instance.depthOrderedNodesBuffer,
                       instance.depthOrderedNodes.size() * sizeof(uint32_t),
                       "animation_pose_depth_nodes"},
      BufferAllocation{&instance.blendedNodeStatesBuffer,
                       instance.baseNodeStates.size() *
                           sizeof(AnimationNodeStateGpu),
                       "animation_pose_blended_nodes"},
      BufferAllocation{&instance.blendedWeightsBuffer,
                       instance.baseWeights.size() * sizeof(float),
                       "animation_pose_blended_weights"},
      BufferAllocation{&instance.worldMatricesBuffer,
                       instance.baseNodeStates.size() * sizeof(glm::mat4),
                       "animation_pose_world_matrices"},
  };
  auto allocation = allocateStorageBuffers(services, allocations);
  if (allocation.hasError()) {
    return allocation;
  }
  const std::array updates{
      BufferUpdate{.buffer = instance.nodeMetaBuffer->handle(),
                   .data = std::as_bytes(std::span(instance.nodeMeta))},
      BufferUpdate{.buffer = instance.depthOrderedNodesBuffer->handle(),
                   .data =
                       std::as_bytes(std::span(instance.depthOrderedNodes))},
  };
  auto upload = gpu.updateBuffers(updates);
  if (upload.hasError()) {
    return upload;
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
    buildClipData(instance, clipSlot, state.clipIndex);
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
  const RenderableId renderableId =
      instance.instantiationMap.renderables[prefabRenderableIndex];
  if (!isValid(renderableId)) {
    return Result<bool, std::string>::makeResult(true);
  }
  const uint32_t runtimeRenderableIndex =
      *scene.findRenderableIndex(renderableId);
  const ScenePrefabRenderable &prefabRenderable =
      instance.prefab.renderables[prefabRenderableIndex];
  if (!renderableControlledByInstance(instance, prefabRenderable)) {
    return Result<bool, std::string>::makeResult(true);
  }
  AnimationRenderableBindingGpu binding{};
  binding.runtimeRenderableIndex = runtimeRenderableIndex;
  binding.nodeIndex = prefabRenderable.nodeIndex;
  instance.renderableBindings.push_back(binding);
  ModelRef modelRef = kInvalidModelRef;
  (void)scene.graph().getRenderableModel(renderableId, modelRef);
  const Model &model = *resources.tryGet(modelRef)->model;
  const Model::ModelAnimationGpuView &animationView = model.animationGpuView();
  GeometryAllocationView geometry{};
  (void)services.gpu().resolveGeometry(model.geometryHandle(), geometry);
  const uint64_t sourceVertexAddress = services.gpu().getBufferDeviceAddress(
      geometry.vertexBuffer, geometry.vertexByteOffset);
  const bool hasMorphWeights =
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
  }
  if (!hasMorph && !hasSkin) {
    return Result<bool, std::string>::makeResult(true);
  }
  AnimatedRenderableState animated{};
  animated.renderableId = renderableId;
  animated.runtimeRenderableIndex = runtimeRenderableIndex;
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
  std::array<BufferAllocation, 4> allocations{{
      {&instance.renderableBindingsBuffer,
       instance.renderableBindings.size() *
           sizeof(AnimationRenderableBindingGpu),
       "animation_pose_renderable_bindings"},
  }};
  size_t allocationCount = 1u;
  const bool hasSkin = !instance.flattenedJointNodeIndices.empty();
  if (hasSkin) {
    allocations[allocationCount++] = {
        &instance.jointNodeIndicesBuffer,
        instance.flattenedJointNodeIndices.size() * sizeof(uint32_t),
        "animation_pose_skin_joint_nodes"};
    allocations[allocationCount++] = {
        &instance.inverseBindMatricesBuffer,
        instance.flattenedInverseBindMatrices.size() * sizeof(glm::mat4),
        "animation_pose_skin_inverse_bind"};
    allocations[allocationCount++] = {&instance.jointPaletteBuffer,
                                      static_cast<size_t>(totalPaletteCount) *
                                          sizeof(glm::mat4),
                                      "animation_pose_skin_palette"};
  }
  auto allocation =
      allocateStorageBuffers(services, {allocations.data(), allocationCount});
  if (allocation.hasError()) {
    return allocation;
  }
  instance.cachedSceneTopologyVersion = scene.topologyVersion();
  instance.cachedGeometryMutationVersion =
      services.gpu().geometryMutationVersion();
  std::array<BufferUpdate, 3> updates{{
      {.buffer = instance.renderableBindingsBuffer->handle(),
       .data = std::as_bytes(std::span(instance.renderableBindings))},
  }};
  size_t updateCount = 1u;
  if (hasSkin) {
    updates[updateCount++] = {
        .buffer = instance.jointNodeIndicesBuffer->handle(),
        .data = std::as_bytes(std::span(instance.flattenedJointNodeIndices))};
    updates[updateCount++] = {
        .buffer = instance.inverseBindMatricesBuffer->handle(),
        .data =
            std::as_bytes(std::span(instance.flattenedInverseBindMatrices))};
  }
  return services.gpu().updateBuffers({updates.data(), updateCount});
}
Result<bool, std::string>
ensureRenderableBindings(SceneRuntimeHost &host, AnimationGpuServices &services,
                         AnimationPoseInstance &instance) {
  const RenderScene *scene = host.scene();
  const ResourceManager *resources = scene->resources();
  resetBindingBuffers(instance);
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
  for (size_t slot = 0u; slot < frameState.instanceMatricesBuffers.size();
       ++slot) {
    if (frameState.instanceMatricesBuffers[slot] &&
        frameState.instanceMatricesCapacityBytes[slot] >= requiredBytes) {
      continue;
    }
    auto bufferResult = services.createStorageBuffer(
        requiredBytes, "animation_scene_instances_" + std::to_string(slot));
    if (bufferResult.hasError()) {
      return Result<bool, std::string>::makeError(bufferResult.error());
    }
    std::unique_ptr<Buffer> previous =
        std::move(frameState.instanceMatricesBuffers[slot]);
    frameState.instanceMatricesBuffers[slot] = std::move(bufferResult.value());
    frameState.instanceMatricesCapacityBytes[slot] = requiredBytes;
    previous.reset();
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

Result<bool, std::string>
AnimationPoseSimulationBackend::createInstance(SceneRuntimeHost &host,
                                               SimulationHandle handle,
                                               const SimulationDesc &desc) {
  if (services_ == nullptr)
    return Result<bool, std::string>::makeError(
        "AnimationPoseSimulationBackend: animation GPU services are not "
        "attached");
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
  const AnimationPoseSimulationCreatePayload &payload =
      *host.pendingAnimationPoseCreatePayload_;
  AnimationPoseInstance instance(payload.prefab, payload.instantiationMap,
                                 memory_);
  instance.rootNode = desc.binding.primaryTarget.prefabRoot;
  instance.params = params;
  instance.controlledNodeMask.assign(instance.prefab.nodes.size(), uint8_t{0u});
  if (payload.controlledPrefabNodeIndices.empty()) {
    std::fill(instance.controlledNodeMask.begin(),
              instance.controlledNodeMask.end(), uint8_t{1u});
  } else {
    for (const uint32_t nodeIndex : payload.controlledPrefabNodeIndices) {
      instance.controlledNodeMask[nodeIndex] = 1u;
    }
  }
  auto validateResult =
      validateAnimationPoseSimulationParams(instance.prefab, params);
  if (validateResult.hasError()) {
    return Result<bool, std::string>::makeError(validateResult.error());
  }
  buildBaseData(instance);
  auto staticBufferResult = createStaticBuffers(*services_, instance);
  if (staticBufferResult.hasError()) {
    return staticBufferResult;
  }
  auto bindingResult = ensureRenderableBindings(host, *services_, instance);
  if (bindingResult.hasError()) {
    return bindingResult;
  }
  impl_->instances.emplace(handle, std::move(instance));
  invalidatePreparedSceneFrame(impl_->sceneFrame);
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
AnimationPoseSimulationBackend::destroyInstance(SceneRuntimeHost &,
                                                SimulationHandle handle) {
  impl_->instances.erase(handle);
  if (impl_->instances.empty()) {
    impl_->sceneFrame = SceneFrameState(memory_);
  } else {
    invalidatePreparedSceneFrame(impl_->sceneFrame);
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> AnimationPoseSimulationBackend::updateParams(
    SceneRuntimeHost &, SimulationHandle handle,
    std::span<const std::byte> params) {
  auto it = impl_->instances.find(handle);
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
    auto recreateResult = createStaticBuffers(*services_, it->second);
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
  auto it = impl_->instances.find(handle);
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
  return record.enabled && !record.stats.faulted &&
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
  if (impl_->sceneFrame.preparedFrameIndex == frameIndex) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (impl_->sceneFrame.preparedFrameIndex !=
      std::numeric_limits<uint64_t>::max()) {
    abandonSceneFrame(impl_->sceneFrame.preparedFrameIndex);
  }
  SceneFrameState &sceneFrame = impl_->sceneFrame;
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
  const uint64_t geometryMutationVersion =
      services_->gpu().geometryMutationVersion();
  const bool renderableBindingsValid = std::all_of(
      impl_->instances.begin(), impl_->instances.end(),
      [&scene, geometryMutationVersion](const auto &entry) {
        const AnimationPoseInstance &instance = entry.second;
        return instance.cachedSceneTopologyVersion == scene.topologyVersion() &&
               instance.cachedGeometryMutationVersion ==
                   geometryMutationVersion;
      });
  const bool previousFrameValid =
      sceneFrame.committedScene == &scene &&
      sceneFrame.committedSceneTopologyVersion == scene.topologyVersion() &&
      sceneFrame.committedRenderableCount == scene.renderables().size() &&
      renderableBindingsValid;
  const size_t previousHistorySlot = sceneFrame.committedHistorySlot;
  const size_t currentHistorySlot =
      sceneFrame.committedScene != nullptr
          ? (sceneFrame.committedHistorySlot + 1u) %
                kAnimationSceneFrameHistorySlots
          : static_cast<size_t>(frameIndex % kAnimationSceneFrameHistorySlots);
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
  struct DispatchCounts {
    size_t sample = 0u, blend = 0u, world = 0u, scatter = 0u;
    size_t animated = 0u, morph = 0u, palette = 0u, skin = 0u;
  } counts;
  for (auto &[handle, instance] : impl_->instances) {
    if (scene.topologyVersion() != instance.cachedSceneTopologyVersion ||
        geometryMutationVersion != instance.cachedGeometryMutationVersion) {
      auto result = ensureRenderableBindings(host, *services_, instance);
      if (result.hasError()) {
        return result;
      }
    }
    const bool blending = blendEnabled(instance.prefab, instance.params);
    const bool secondaryOnly =
        secondaryOnlyBlendEnabled(instance.prefab, instance.params);
    counts.sample +=
        !secondaryOnly && !instance.clips[kPrimaryClipSlot].channels.empty();
    counts.sample +=
        blending && !instance.clips[kSecondaryClipSlot].channels.empty();
    counts.blend += blending && !secondaryOnly;
    counts.world += instance.depthBucketStarts.size();
    counts.scatter += !instance.renderableBindings.empty();
    counts.animated += instance.renderableBindings.size();
    for (const AnimatedRenderableState &animated :
         instance.animatedRenderables) {
      counts.morph += animated.hasMorph;
      counts.palette += animated.hasSkin;
      counts.skin += animated.hasSkin;
    }
  }
  sceneFrame.preDispatches.reserve(counts.sample + counts.blend + counts.world +
                                   counts.scatter + counts.morph +
                                   counts.palette + counts.skin);
  sceneFrame.dispatchRecords.reserve(sceneFrame.preDispatches.capacity());
  sceneFrame.animatedRenderableIndices.reserve(counts.animated);
  const uint64_t instanceMatricesAddress =
      services_->gpu().getBufferDeviceAddress(
          sceneFrame.instanceMatricesBuffers[sceneFrame.currentHistorySlot]
              ->handle());
  for (auto &[handle, instance] : impl_->instances) {
    const SimulationRegistry::Record &record = *host.registry_.tryGet(handle);
    if (record.stats.faulted) {
      continue;
    }
    const AnimationClipGpuData &primaryClipGpu =
        instance.clips[kPrimaryClipSlot];
    const AnimationClipData &primaryClip =
        instance.prefab.animations[instance.params.primary.clipIndex];
    const float primarySampleTime = computeRenderSampleTime(
        host, record, instance.params.primary, primaryClip);
    const bool instanceBlendEnabled =
        blendEnabled(instance.prefab, instance.params);
    const bool useSecondaryOnly =
        secondaryOnlyBlendEnabled(instance.prefab, instance.params);
    std::array<BufferUpdate, 4u> sampleUpdates{};
    size_t sampleUpdateCount = 0u;
    const auto appendSampleUpdate = [&sampleUpdates, &sampleUpdateCount](
                                        Buffer &buffer, const auto &values) {
      if (values.empty()) {
        return;
      }
      sampleUpdates[sampleUpdateCount++] = BufferUpdate{
          .buffer = buffer.handle(),
          .data = std::as_bytes(std::span(values.data(), values.size())),
      };
    };
    if (!useSecondaryOnly) {
      appendSampleUpdate(*primaryClipGpu.sampledNodeStatesBuffer,
                         instance.baseNodeStates);
      appendSampleUpdate(*primaryClipGpu.sampledWeightsBuffer,
                         instance.baseWeights);
    }
    const AnimationClipGpuData *secondaryClipGpu = nullptr;
    const AnimationClipData *secondaryClip = nullptr;
    float secondarySampleTime = 0.0f;
    if (instanceBlendEnabled) {
      secondaryClipGpu = &instance.clips[kSecondaryClipSlot];
      secondaryClip =
          &instance.prefab.animations[instance.params.secondary.clipIndex];
      secondarySampleTime = computeRenderSampleTime(
          host, record, instance.params.secondary, *secondaryClip);
      if (instance.params.blendSyncMode ==
          AnimationPoseBlendSyncMode::NormalizedTime) {
        secondarySampleTime =
            computeNormalizedPhase(primarySampleTime, primaryClip) *
            std::max(secondaryClip->durationSeconds, 0.0f);
      }
      appendSampleUpdate(*secondaryClipGpu->sampledNodeStatesBuffer,
                         instance.baseNodeStates);
      appendSampleUpdate(*secondaryClipGpu->sampledWeightsBuffer,
                         instance.baseWeights);
    }
    if (sampleUpdateCount != 0u) {
      auto uploadResult =
          services_->gpu().updateBuffers(std::span<const BufferUpdate>(
              sampleUpdates.data(), sampleUpdateCount));
      if (uploadResult.hasError()) {
        return uploadResult;
      }
    }
    uint64_t primaryNodeStatesAddress = 0u;
    uint64_t primaryWeightsAddress = 0u;
    if (!useSecondaryOnly) {
      primaryNodeStatesAddress = services_->gpu().getBufferDeviceAddress(
          primaryClipGpu.sampledNodeStatesBuffer->handle());
      primaryWeightsAddress = services_->gpu().getBufferDeviceAddress(
          primaryClipGpu.sampledWeightsBuffer->handle());
    }
    glm::mat4 rootTransform(1.0f);
    (void)scene.graph().getCachedNodeWorldTransform(instance.rootNode,
                                                    rootTransform);
    const size_t mappedNodeCount = std::min(
        instance.prefab.nodes.size(), instance.instantiationMap.nodes.size());
    for (size_t nodeIndex = 0u; nodeIndex < mappedNodeCount; ++nodeIndex) {
      const ScenePrefabNode &prefabNode = instance.prefab.nodes[nodeIndex];
      if (instance.instantiationMap.nodes[nodeIndex] != instance.rootNode ||
          prefabNode.parentIndex != kInvalidScenePrefabIndex) {
        continue;
      }
      const float determinant = glm::determinant(prefabNode.localFromParent);
      if (std::abs(determinant) > std::numeric_limits<float>::epsilon()) {
        rootTransform *= glm::inverse(prefabNode.localFromParent);
      }
      break;
    }
    const uint64_t worldMatricesAddress =
        services_->gpu().getBufferDeviceAddress(
            instance.worldMatricesBuffer->handle());
    auto appendSampleDispatch = [&](const AnimationClipGpuData &clipGpu,
                                    float sampleTimeSeconds) {
      if (clipGpu.channels.empty()) {
        return;
      }
      const auto &dispatch = appendAnimationDispatch(
          sceneFrame.dispatchRecords,
          AnimationDispatchRecord<SamplePushConstants, 4>{
              .push =
                  {
                      .channelsAddress =
                          services_->gpu().getBufferDeviceAddress(
                              clipGpu.channelsBuffer->handle()),
                      .keyTimesAddress =
                          services_->gpu().getBufferDeviceAddress(
                              clipGpu.keyTimesBuffer->handle()),
                      .valuesAddress = services_->gpu().getBufferDeviceAddress(
                          clipGpu.valuesBuffer->handle()),
                      .nodeStatesAddress =
                          services_->gpu().getBufferDeviceAddress(
                              clipGpu.sampledNodeStatesBuffer->handle()),
                      .sampledWeightsAddress =
                          services_->gpu().getBufferDeviceAddress(
                              clipGpu.sampledWeightsBuffer->handle()),
                      .sampleTimeSeconds = sampleTimeSeconds,
                      .channelCount =
                          static_cast<uint32_t>(clipGpu.channels.size()),
                  },
              .dependencies = {clipGpu.channelsBuffer->handle(),
                               clipGpu.keyTimesBuffer->handle(),
                               clipGpu.sampledNodeStatesBuffer->handle(),
                               clipGpu.sampledWeightsBuffer->handle()},
          });
      appendComputeDispatch(
          sceneFrame.preDispatches, services_->samplePipeline(),
          dispatchCount(dispatch.push.channelCount), dispatch.push,
          dispatch.dependencies, dispatch.dependencies.size(),
          kSampleDependencyAccess, "AnimationPose Sample", 0xff5599ffu);
    };
    if (!useSecondaryOnly) {
      appendSampleDispatch(primaryClipGpu, primarySampleTime);
    }
    uint64_t nodeStatesAddress = primaryNodeStatesAddress;
    uint64_t sampledWeightsAddress = primaryWeightsAddress;
    BufferHandle nodeStatesBuffer =
        !useSecondaryOnly ? primaryClipGpu.sampledNodeStatesBuffer->handle()
                          : BufferHandle{};
    BufferHandle sampledWeightsBuffer =
        !useSecondaryOnly ? primaryClipGpu.sampledWeightsBuffer->handle()
                          : BufferHandle{};
    if (instanceBlendEnabled) {
      appendSampleDispatch(*secondaryClipGpu, secondarySampleTime);
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
      if (useSecondaryOnly) {
        nodeStatesAddress = secondaryNodeStatesAddress;
        sampledWeightsAddress = secondaryWeightsAddress;
        nodeStatesBuffer = secondaryClipGpu->sampledNodeStatesBuffer->handle();
        sampledWeightsBuffer = secondaryClipGpu->sampledWeightsBuffer->handle();
      } else {
        const auto &dispatch = appendAnimationDispatch(
            sceneFrame.dispatchRecords,
            AnimationDispatchRecord<BlendPushConstants, 6>{
                .push =
                    {
                        .sourceNodeStatesAddressA = primaryNodeStatesAddress,
                        .sourceNodeStatesAddressB = secondaryNodeStatesAddress,
                        .outputNodeStatesAddress = blendedNodeStatesAddress,
                        .sourceWeightsAddressA = primaryWeightsAddress,
                        .sourceWeightsAddressB = secondaryWeightsAddress,
                        .outputWeightsAddress = blendedWeightsAddress,
                        .blendWeight = instance.params.blendWeight,
                        .nodeCount = static_cast<uint32_t>(
                            instance.baseNodeStates.size()),
                        .weightCount =
                            static_cast<uint32_t>(instance.baseWeights.size()),
                    },
                .dependencies =
                    {primaryClipGpu.sampledNodeStatesBuffer->handle(),
                     secondaryClipGpu->sampledNodeStatesBuffer->handle(),
                     instance.blendedNodeStatesBuffer->handle(),
                     primaryClipGpu.sampledWeightsBuffer->handle(),
                     secondaryClipGpu->sampledWeightsBuffer->handle(),
                     instance.blendedWeightsBuffer->handle()},
            });
        appendComputeDispatch(
            sceneFrame.preDispatches, services_->blendPipeline(),
            dispatchCount(
                std::max(dispatch.push.nodeCount, dispatch.push.weightCount)),
            dispatch.push, dispatch.dependencies, dispatch.dependencies.size(),
            kBlendDependencyAccess, "AnimationPose Blend", 0xff6688ffu);
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
      const auto &dispatch = appendAnimationDispatch(
          sceneFrame.dispatchRecords,
          AnimationDispatchRecord<WorldPushConstants, 4>{
              .push =
                  {
                      .nodeStatesAddress = nodeStatesAddress,
                      .nodeMetaAddress =
                          services_->gpu().getBufferDeviceAddress(
                              instance.nodeMetaBuffer->handle()),
                      .depthOrderedNodesAddress =
                          services_->gpu().getBufferDeviceAddress(
                              instance.depthOrderedNodesBuffer->handle()),
                      .worldMatricesAddress = worldMatricesAddress,
                      .rootTransform = rootTransform,
                      .nodeStart = instance.depthBucketStarts[depthBucketIndex],
                      .nodeCount = nodeCount,
                  },
              .dependencies = {nodeStatesBuffer,
                               instance.nodeMetaBuffer->handle(),
                               instance.depthOrderedNodesBuffer->handle(),
                               instance.worldMatricesBuffer->handle()},
          });
      appendComputeDispatch(
          sceneFrame.preDispatches, services_->worldPipeline(),
          dispatchCount(nodeCount), dispatch.push, dispatch.dependencies,
          dispatch.dependencies.size(), kWorldDependencyAccess,
          "AnimationPose World", 0xff33aa55u);
    }
    if (!instance.renderableBindings.empty()) {
      for (const AnimationRenderableBindingGpu &binding :
           instance.renderableBindings) {
        sceneFrame.animatedRenderableIndices.push_back(
            binding.runtimeRenderableIndex);
      }
      const auto &dispatch = appendAnimationDispatch(
          sceneFrame.dispatchRecords,
          AnimationDispatchRecord<ScatterPushConstants, 4>{
              .push =
                  {
                      .renderableBindingsAddress =
                          services_->gpu().getBufferDeviceAddress(
                              instance.renderableBindingsBuffer->handle()),
                      .worldMatricesAddress = worldMatricesAddress,
                      .instanceMatricesAddress = instanceMatricesAddress,
                      .bindingCount = static_cast<uint32_t>(
                          instance.renderableBindings.size()),
                  },
              .dependencies =
                  {instance.renderableBindingsBuffer->handle(),
                   instance.worldMatricesBuffer->handle(),
                   sceneFrame
                       .instanceMatricesBuffers[sceneFrame.currentHistorySlot]
                       ->handle(),
                   BufferHandle{}},
          });
      appendComputeDispatch(
          sceneFrame.preDispatches, services_->scatterPipeline(),
          dispatchCount(dispatch.push.bindingCount), dispatch.push,
          dispatch.dependencies, size_t{3u}, kScatterDependencyAccess,
          "AnimationPose Scatter", 0xff33cc88u);
    }
    for (AnimatedRenderableState &animated : instance.animatedRenderables) {
      BufferHandle overrideBuffer{};
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
      if (animated.hasSkin) {
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
        const auto &dispatch = appendAnimationDispatch(
            sceneFrame.dispatchRecords,
            AnimationDispatchRecord<SkinPalettePushConstants, 4>{
                .push =
                    {
                        .worldMatricesAddress = worldMatricesAddress,
                        .jointNodeIndicesAddress = jointNodeAddress,
                        .inverseBindMatricesAddress = inverseBindAddress,
                        .paletteAddress = paletteAddress,
                        .renderableNodeIndex = animated.nodeIndex,
                        .jointCount = animated.jointCount,
                    },
                .dependencies = {instance.worldMatricesBuffer->handle(),
                                 instance.jointNodeIndicesBuffer->handle(),
                                 instance.inverseBindMatricesBuffer->handle(),
                                 instance.jointPaletteBuffer->handle()},
            });
        appendComputeDispatch(
            sceneFrame.preDispatches, services_->skinPalettePipeline(),
            dispatchCount(animated.jointCount), dispatch.push,
            dispatch.dependencies, dispatch.dependencies.size(),
            kSkinPaletteDependencyAccess, "AnimationPose SkinPalette",
            0xffaa55ccu);
      }
      if (animated.hasMorph) {
        const uint64_t outputVertexAddress =
            services_->gpu().getBufferDeviceAddress(
                currentMorphOutput->handle());
        const uint64_t morphDeltaAddress =
            services_->gpu().getBufferDeviceAddress(animated.morphDeltaBuffer);
        const auto &dispatch = appendAnimationDispatch(
            sceneFrame.dispatchRecords,
            AnimationDispatchRecord<MorphPushConstants, 4>{
                .push =
                    {
                        .sourceVertexAddress = animated.sourceVertexAddress,
                        .outputVertexAddress = outputVertexAddress,
                        .morphDeltaAddress = morphDeltaAddress,
                        .weightAddress = sampledWeightsAddress,
                        .weightOffset =
                            instance.nodeMeta[animated.nodeIndex].weightOffset,
                        .weightCount =
                            instance.nodeMeta[animated.nodeIndex].weightCount,
                        .morphTargetCount = animated.morphTargetCount,
                        .vertexCount = animated.vertexCount,
                    },
                .dependencies = {animated.sourceVertexBuffer,
                                 animated.morphDeltaBuffer,
                                 sampledWeightsBuffer,
                                 currentMorphOutput->handle()},
            });
        appendComputeDispatch(
            sceneFrame.preDispatches, services_->morphPipeline(),
            dispatchCount(animated.vertexCount), dispatch.push,
            dispatch.dependencies, dispatch.dependencies.size(),
            kMorphDependencyAccess, "AnimationPose Morph", 0xffaa7733u);
        if (!animated.hasSkin) {
          overrideBuffer = currentMorphOutput->handle();
        }
      }
      if (animated.hasSkin) {
        const uint64_t skinSourceAddress =
            animated.hasMorph ? services_->gpu().getBufferDeviceAddress(
                                    currentMorphOutput->handle())
                              : animated.sourceVertexAddress;
        const BufferHandle skinSourceBuffer = animated.hasMorph
                                                  ? currentMorphOutput->handle()
                                                  : animated.sourceVertexBuffer;
        const uint64_t outputVertexAddress =
            services_->gpu().getBufferDeviceAddress(
                currentSkinOutput->handle());
        const uint64_t skinInfluenceAddress =
            services_->gpu().getBufferDeviceAddress(
                animated.skinInfluenceBuffer);
        const uint64_t paletteAddress = services_->gpu().getBufferDeviceAddress(
            instance.jointPaletteBuffer->handle(),
            static_cast<size_t>(animated.paletteOffset) * sizeof(glm::mat4));
        const auto &dispatch = appendAnimationDispatch(
            sceneFrame.dispatchRecords,
            AnimationDispatchRecord<SkinPushConstants, 4>{
                .push =
                    {
                        .sourceVertexAddress = skinSourceAddress,
                        .outputVertexAddress = outputVertexAddress,
                        .skinInfluenceAddress = skinInfluenceAddress,
                        .paletteAddress = paletteAddress,
                        .vertexCount = animated.vertexCount,
                    },
                .dependencies = {skinSourceBuffer, animated.skinInfluenceBuffer,
                                 instance.jointPaletteBuffer->handle(),
                                 currentSkinOutput->handle()},
            });
        appendComputeDispatch(
            sceneFrame.preDispatches, services_->skinPipeline(),
            dispatchCount(animated.vertexCount), dispatch.push,
            dispatch.dependencies, dispatch.dependencies.size(),
            kSkinDependencyAccess, "AnimationPose Skin", 0xffcc8844u);
        overrideBuffer = currentSkinOutput->handle();
      }
      sceneFrame.geometryOverrides[animated.runtimeRenderableIndex] =
          AnimatedRenderableGeometryOverride{
              .vertexBuffer = overrideBuffer,
              .vertexByteOffset = 0u,
              .vertexCount = animated.vertexCount,
              .enabled = true,
          };
      if (sceneFrame.previousFrameValid) {
        const BufferHandle previousOverrideBuffer =
            (previousSkinOutput ? previousSkinOutput : previousMorphOutput)
                ->handle();
        sceneFrame.previousGeometryOverrides[animated.runtimeRenderableIndex] =
            AnimatedRenderableGeometryOverride{
                .vertexBuffer = previousOverrideBuffer,
                .vertexByteOffset = 0u,
                .vertexCount = animated.vertexCount,
                .enabled = true,
            };
      }
    }
  }
  sceneFrame.version = ++impl_->sceneFrameVersionCounter;
  sceneFrame.preparedFrameIndex = frameIndex;
  return Result<bool, std::string>::makeResult(true);
}

void AnimationPoseSimulationBackend::commitSceneFrame(
    uint64_t frameIndex) noexcept {
  if (impl_->sceneFrame.preparedFrameIndex != frameIndex) {
    return;
  }
  SceneFrameState &sceneFrame = impl_->sceneFrame;
  if (sceneFrame.version != 0u) {
    sceneFrame.committedHistorySlot = sceneFrame.currentHistorySlot;
    sceneFrame.committedScene = sceneFrame.scene;
    sceneFrame.committedSceneTopologyVersion = sceneFrame.sceneTopologyVersion;
    sceneFrame.committedRenderableCount = sceneFrame.renderableCount;
  } else {
    sceneFrame.committedHistorySlot = 0u;
    sceneFrame.committedScene = nullptr;
    sceneFrame.committedSceneTopologyVersion = 0u;
    sceneFrame.committedRenderableCount = 0u;
  }
  sceneFrame.preparedFrameIndex = std::numeric_limits<uint64_t>::max();
}

void AnimationPoseSimulationBackend::abandonSceneFrame(
    uint64_t frameIndex) noexcept {
  if (impl_->sceneFrame.preparedFrameIndex != frameIndex) {
    return;
  }
  SceneFrameState &sceneFrame = impl_->sceneFrame;
  sceneFrame.resetTransientDispatchState();
  sceneFrame.geometryOverrides.clear();
  sceneFrame.previousGeometryOverrides.clear();
  sceneFrame.publishedData = {};
  sceneFrame.version = 0u;
  sceneFrame.preparedFrameIndex = std::numeric_limits<uint64_t>::max();
  sceneFrame.previousFrameValid = false;
  sceneFrame.scene = nullptr;
  sceneFrame.sceneTopologyVersion = 0u;
  sceneFrame.renderableCount = 0u;
}

void AnimationPoseSimulationBackend::reset() {
  impl_->instances.clear();
  impl_->sceneFrame = SceneFrameState(memory_);
}

const AnimationSceneFrameData *
AnimationPoseSimulationBackend::currentSceneFrameData() const noexcept {
  if (impl_->sceneFrame.version == 0u || impl_->instances.empty()) {
    return nullptr;
  }
  const SceneFrameState &sceneFrame = impl_->sceneFrame;
  AnimationSceneFrameData &frameData = impl_->sceneFrame.publishedData;
  frameData.instanceMatricesBuffer =
      sceneFrame.instanceMatricesBuffers[sceneFrame.currentHistorySlot]
          ->handle();
  frameData.instanceMatricesAddress =
      services_->gpu().getBufferDeviceAddress(frameData.instanceMatricesBuffer);
  frameData.previousInstanceMatricesBuffer = {};
  frameData.previousInstanceMatricesAddress = 0u;
  if (sceneFrame.previousFrameValid) {
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
  return &frameData;
}

} // namespace nuri
