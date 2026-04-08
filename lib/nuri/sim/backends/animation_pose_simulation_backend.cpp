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

#include <optional>
#include <vector>

namespace nuri {
namespace {

constexpr uint32_t kInvalidIndex = std::numeric_limits<uint32_t>::max();
constexpr uint32_t kComputeWorkgroupSize = 64u;
constexpr size_t kPackedVertexStrideBytes = 24u;
constexpr float kLoopWrapEpsilonSeconds = 1.0e-5f;

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

bool readParams(std::span<const std::byte> bytes,
                AnimationPoseSimulationParams &out) {
  if (bytes.size() != sizeof(AnimationPoseSimulationParams)) {
    return false;
  }
  std::memcpy(&out, bytes.data(), sizeof(out));
  return true;
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
  std::unique_ptr<Buffer> morphOutputVertexBuffer;
  std::unique_ptr<Buffer> finalOutputVertexBuffer;
};

struct AnimationPoseInstance {
  explicit AnimationPoseInstance(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : prefab(memory), instantiationMap(memory), baseNodeStates(memory),
        baseWeights(memory), nodeMeta(memory), depthOrderedNodes(memory),
        depthBucketStarts(memory), depthBucketCounts(memory), keyTimes(memory),
        values(memory), channels(memory), renderableBindings(memory),
        animatedRenderables(memory), flattenedJointNodeIndices(memory),
        flattenedInverseBindMatrices(memory) {}

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
  std::pmr::vector<float> keyTimes;
  std::pmr::vector<float> values;
  std::pmr::vector<AnimationChannelGpu> channels;
  std::pmr::vector<AnimationRenderableBindingGpu> renderableBindings;
  std::pmr::vector<AnimatedRenderableState> animatedRenderables;
  std::pmr::vector<uint32_t> flattenedJointNodeIndices;
  std::pmr::vector<glm::mat4> flattenedInverseBindMatrices;

  std::unique_ptr<Buffer> nodeMetaBuffer;
  std::unique_ptr<Buffer> depthOrderedNodesBuffer;
  std::unique_ptr<Buffer> keyTimesBuffer;
  std::unique_ptr<Buffer> valuesBuffer;
  std::unique_ptr<Buffer> channelsBuffer;
  std::unique_ptr<Buffer> renderableBindingsBuffer;
  std::unique_ptr<Buffer> sampledNodeStatesBuffer;
  std::unique_ptr<Buffer> sampledWeightsBuffer;
  std::unique_ptr<Buffer> worldMatricesBuffer;
  std::unique_ptr<Buffer> jointNodeIndicesBuffer;
  std::unique_ptr<Buffer> inverseBindMatricesBuffer;
  std::unique_ptr<Buffer> jointPaletteBuffer;
};

struct SceneFrameState {
  explicit SceneFrameState(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : baseInstances(memory), geometryOverrides(memory), preDispatches(memory),
        samplePushConstants(memory), sampleDependencies(memory),
        worldPushConstants(memory), worldDependencies(memory),
        scatterPushConstants(memory), scatterDependencies(memory),
        morphPushConstants(memory), morphDependencies(memory),
        skinPalettePushConstants(memory), skinPaletteDependencies(memory),
        skinPushConstants(memory), skinDependencies(memory) {}

  std::unique_ptr<Buffer> instanceMatricesBuffer;
  size_t instanceMatricesCapacityBytes = 0u;
  std::pmr::vector<InstanceData> baseInstances;
  std::pmr::vector<AnimatedRenderableGeometryOverride> geometryOverrides;
  std::pmr::vector<ComputeDispatchItem> preDispatches;
  std::pmr::vector<SamplePushConstants> samplePushConstants;
  std::pmr::vector<std::array<BufferHandle, 4>> sampleDependencies;
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
  uint64_t version = 0u;
  uint64_t preparedFrameIndex = std::numeric_limits<uint64_t>::max();
  const RenderScene *scene = nullptr;
  uint64_t sceneTopologyVersion = 0u;
  size_t renderableCount = 0u;
  mutable AnimationSceneFrameData publishedData{};

  void resetTransientDispatchState() noexcept {
    preDispatches.clear();
    samplePushConstants.clear();
    sampleDependencies.clear();
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
  }
};

void destroyAnimatedRenderableBuffers(
    GPUDevice &gpu,
    std::pmr::vector<AnimatedRenderableState> &animatedRenderables) {
  for (AnimatedRenderableState &renderable : animatedRenderables) {
    destroyOwnedBuffer(gpu, renderable.morphOutputVertexBuffer);
    destroyOwnedBuffer(gpu, renderable.finalOutputVertexBuffer);
  }
}

void destroyInstanceBuffers(GPUDevice &gpu, AnimationPoseInstance &instance) {
  destroyOwnedBuffer(gpu, instance.nodeMetaBuffer);
  destroyOwnedBuffer(gpu, instance.depthOrderedNodesBuffer);
  destroyOwnedBuffer(gpu, instance.keyTimesBuffer);
  destroyOwnedBuffer(gpu, instance.valuesBuffer);
  destroyOwnedBuffer(gpu, instance.channelsBuffer);
  destroyOwnedBuffer(gpu, instance.renderableBindingsBuffer);
  destroyOwnedBuffer(gpu, instance.sampledNodeStatesBuffer);
  destroyOwnedBuffer(gpu, instance.sampledWeightsBuffer);
  destroyOwnedBuffer(gpu, instance.worldMatricesBuffer);
  destroyOwnedBuffer(gpu, instance.jointNodeIndicesBuffer);
  destroyOwnedBuffer(gpu, instance.inverseBindMatricesBuffer);
  destroyOwnedBuffer(gpu, instance.jointPaletteBuffer);
  destroyAnimatedRenderableBuffers(gpu, instance.animatedRenderables);
}

void destroyBindingBuffers(GPUDevice &gpu, AnimationPoseInstance &instance) {
  destroyOwnedBuffer(gpu, instance.renderableBindingsBuffer);
  destroyOwnedBuffer(gpu, instance.jointNodeIndicesBuffer);
  destroyOwnedBuffer(gpu, instance.inverseBindMatricesBuffer);
  destroyOwnedBuffer(gpu, instance.jointPaletteBuffer);
  destroyAnimatedRenderableBuffers(gpu, instance.animatedRenderables);
}

void destroySceneFrameBuffers(GPUDevice &gpu, SceneFrameState &sceneFrame) {
  destroyOwnedBuffer(gpu, sceneFrame.instanceMatricesBuffer);
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

Result<bool, std::string> buildClipData(AnimationPoseInstance &instance) {
  if (instance.prefab.animations.empty()) {
    return Result<bool, std::string>::makeError(
        "AnimationPoseSimulationBackend: prefab has no animations");
  }
  if (instance.params.clipIndex >= instance.prefab.animations.size()) {
    return Result<bool, std::string>::makeError(
        "AnimationPoseSimulationBackend: clip index is out of range");
  }

  const AnimationClipData &clip =
      instance.prefab.animations[instance.params.clipIndex];
  instance.keyTimes.clear();
  instance.values.clear();
  instance.channels.clear();
  instance.channels.reserve(clip.channels.size());
  for (const AnimationChannelData &channel : clip.channels) {
    if (channel.samplerIndex >= clip.samplers.size() ||
        channel.targetNodeIndex >= instance.nodeMeta.size()) {
      continue;
    }
    const AnimationSamplerData &sampler = clip.samplers[channel.samplerIndex];
    instance.channels.push_back(AnimationChannelGpu{
        .keyOffset = static_cast<uint32_t>(instance.keyTimes.size()),
        .valueOffset = static_cast<uint32_t>(instance.values.size()),
        .keyCount = static_cast<uint32_t>(sampler.keyTimes.size()),
        .valueArity = sampler.valueArity,
        .targetNodeIndex = channel.targetNodeIndex,
        .path = static_cast<uint32_t>(channel.path),
        .interpolation = static_cast<uint32_t>(sampler.interpolation),
        .targetWeightOffset =
            instance.nodeMeta[channel.targetNodeIndex].weightOffset,
    });
    instance.keyTimes.insert(instance.keyTimes.end(), sampler.keyTimes.begin(),
                             sampler.keyTimes.end());
    instance.values.insert(instance.values.end(), sampler.values.begin(),
                           sampler.values.end());
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> createStaticBuffers(AnimationGpuServices &services,
                                              AnimationPoseInstance &instance) {
  auto &gpu = services.gpu();
  destroyOwnedBuffer(gpu, instance.nodeMetaBuffer);
  destroyOwnedBuffer(gpu, instance.depthOrderedNodesBuffer);
  destroyOwnedBuffer(gpu, instance.keyTimesBuffer);
  destroyOwnedBuffer(gpu, instance.valuesBuffer);
  destroyOwnedBuffer(gpu, instance.channelsBuffer);
  destroyOwnedBuffer(gpu, instance.sampledNodeStatesBuffer);
  destroyOwnedBuffer(gpu, instance.sampledWeightsBuffer);
  destroyOwnedBuffer(gpu, instance.worldMatricesBuffer);
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
  auto keyTimesResult = services.createStorageBuffer(
      instance.keyTimes.size() * sizeof(float), "animation_pose_key_times");
  if (keyTimesResult.hasError()) {
    return Result<bool, std::string>::makeError(keyTimesResult.error());
  }
  instance.keyTimesBuffer = std::move(keyTimesResult.value());
  auto valuesResult = services.createStorageBuffer(
      instance.values.size() * sizeof(float), "animation_pose_values");
  if (valuesResult.hasError()) {
    return Result<bool, std::string>::makeError(valuesResult.error());
  }
  instance.valuesBuffer = std::move(valuesResult.value());
  auto channelResult = services.createStorageBuffer(
      instance.channels.size() * sizeof(AnimationChannelGpu),
      "animation_pose_channels");
  if (channelResult.hasError()) {
    return Result<bool, std::string>::makeError(channelResult.error());
  }
  instance.channelsBuffer = std::move(channelResult.value());
  auto nodeStatesResult = services.createStorageBuffer(
      instance.baseNodeStates.size() * sizeof(AnimationNodeStateGpu),
      "animation_pose_sampled_nodes");
  if (nodeStatesResult.hasError()) {
    return Result<bool, std::string>::makeError(nodeStatesResult.error());
  }
  instance.sampledNodeStatesBuffer = std::move(nodeStatesResult.value());
  auto sampledWeightsResult =
      services.createStorageBuffer(instance.baseWeights.size() * sizeof(float),
                                   "animation_pose_sampled_weights");
  if (sampledWeightsResult.hasError()) {
    return Result<bool, std::string>::makeError(sampledWeightsResult.error());
  }
  instance.sampledWeightsBuffer = std::move(sampledWeightsResult.value());
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
  auto uploadKeyTimes =
      uploadVector(gpu, *instance.keyTimesBuffer, instance.keyTimes);
  if (uploadKeyTimes.hasError()) {
    return uploadKeyTimes;
  }
  auto uploadValues =
      uploadVector(gpu, *instance.valuesBuffer, instance.values);
  if (uploadValues.hasError()) {
    return uploadValues;
  }
  return uploadVector(gpu, *instance.channelsBuffer, instance.channels);
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
    auto morphOutputResult = services.createStorageVertexBuffer(
        requiredBytes, "animation_pose_morph_output");
    if (morphOutputResult.hasError()) {
      return Result<bool, std::string>::makeError(morphOutputResult.error());
    }
    animated.morphOutputVertexBuffer = std::move(morphOutputResult.value());
  }

  if (animated.hasSkin) {
    if (skin->jointNodeIndices.size() != skin->inverseBindMatrices.size()) {
      return Result<bool, std::string>::makeError(
          "AnimationPoseSimulationBackend: skin jointNodeIndices and "
          "inverseBindMatrices counts do not match");
    }
    auto finalOutputResult = services.createStorageVertexBuffer(
        requiredBytes, "animation_pose_skin_output");
    if (finalOutputResult.hasError()) {
      return Result<bool, std::string>::makeError(finalOutputResult.error());
    }
    animated.finalOutputVertexBuffer = std::move(finalOutputResult.value());
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
                         AnimationPoseInstance &instance) {
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
  destroyBindingBuffers(services.gpu(), instance);
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
                                                 SceneFrameState &frameState) {
  const size_t requiredBytes = std::max(
      scene.renderables().size() * sizeof(InstanceData), sizeof(InstanceData));
  if (!frameState.instanceMatricesBuffer ||
      frameState.instanceMatricesCapacityBytes < requiredBytes) {
    destroyOwnedBuffer(services.gpu(), frameState.instanceMatricesBuffer);
    auto bufferResult = services.createStorageBuffer(
        requiredBytes, "animation_scene_instances");
    if (bufferResult.hasError()) {
      return Result<bool, std::string>::makeError(bufferResult.error());
    }
    frameState.instanceMatricesBuffer = std::move(bufferResult.value());
    frameState.instanceMatricesCapacityBytes = requiredBytes;
  }

  frameState.baseInstances.clear();
  frameState.geometryOverrides.clear();
  frameState.baseInstances.reserve(scene.renderables().size());
  frameState.geometryOverrides.resize(scene.renderables().size());
  for (const Renderable &renderable : scene.renderables()) {
    frameState.baseInstances.push_back(
        makeInstanceData(renderable.modelMatrix));
  }

  auto uploadResult =
      uploadVector(services.gpu(), *frameState.instanceMatricesBuffer,
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

  AnimationPoseSimulationParams params{};
  if (!readParams(desc.initialParams, params)) {
    return Result<bool, std::string>::makeError(
        "AnimationPoseSimulationBackend: invalid params payload");
  }
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
  auto baseResult = buildBaseData(instance);
  if (baseResult.hasError()) {
    return baseResult;
  }
  auto clipResult = buildClipData(instance);
  if (clipResult.hasError()) {
    return clipResult;
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
    destroyInstanceBuffers(services_->gpu(), it->second);
  }
  impl_->instances.erase(it);
  if (impl_->instances.empty()) {
    if (services_ != nullptr) {
      destroySceneFrameBuffers(services_->gpu(), impl_->sceneFrame);
    }
    impl_->sceneFrame.publishedData = {};
    impl_->sceneFrame.version = 0u;
    impl_->sceneFrame.preparedFrameIndex = std::numeric_limits<uint64_t>::max();
    impl_->sceneFrame.scene = nullptr;
    impl_->sceneFrame.sceneTopologyVersion = 0u;
    impl_->sceneFrame.renderableCount = 0u;
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
  AnimationPoseSimulationParams decoded{};
  if (!readParams(params, decoded)) {
    return Result<bool, std::string>::makeError(
        "AnimationPoseSimulationBackend: invalid params payload");
  }
  it->second.params = decoded;
  auto clipResult = buildClipData(it->second);
  if (clipResult.hasError()) {
    return clipResult;
  }
  return createStaticBuffers(*services_, it->second);
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
  const AnimationClipData &clip =
      instance.prefab.animations[instance.params.clipIndex];
  if (instance.params.playing && clip.durationSeconds > 0.0f) {
    instance.params.timeSeconds =
        std::max(0.0f, instance.params.timeSeconds +
                           static_cast<float>(context.effectiveDeltaSeconds));
    if (instance.params.playbackMode == AnimationPosePlaybackMode::Loop) {
      instance.params.timeSeconds =
          wrapLoopTime(instance.params.timeSeconds, clip.durationSeconds);
    } else if (instance.params.timeSeconds >= clip.durationSeconds) {
      instance.params.timeSeconds = clip.durationSeconds;
      instance.params.playing = false;
    }
  }
  return Result<bool, std::string>::makeResult(true);
}

namespace {

bool isPlaybackRunning(const SimulationRegistry::Record &record,
                       const AnimationPoseInstance &instance) noexcept {
  return record.enabled && !record.faulted &&
         record.state == SimulationState::Running && instance.params.playing;
}

float computeRenderSampleTime(const SceneRuntimeHost &host,
                              const SimulationRegistry::Record &record,
                              const AnimationPoseInstance &instance,
                              const AnimationClipData &clip) noexcept {
  float sampleTime = instance.params.timeSeconds;
  if (!isPlaybackRunning(record, instance) || clip.durationSeconds <= 0.0f) {
    return sampleTime;
  }
  const double accumulatorSeconds =
      std::max(0.0, host.remainingAccumulatorSeconds());
  sampleTime +=
      static_cast<float>(accumulatorSeconds * std::max(record.timeScale, 0.0f));
  if (instance.params.playbackMode == AnimationPosePlaybackMode::Loop) {
    return wrapLoopTime(sampleTime, clip.durationSeconds);
  }
  return std::min(sampleTime, clip.durationSeconds);
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
  sceneFrame.resetTransientDispatchState();

  if (impl_->instances.empty()) {
    sceneFrame.geometryOverrides.clear();
    sceneFrame.publishedData = {};
    sceneFrame.version = 0u;
    sceneFrame.preparedFrameIndex = frameIndex;
    sceneFrame.scene = nullptr;
    sceneFrame.sceneTopologyVersion = 0u;
    sceneFrame.renderableCount = 0u;
    return Result<bool, std::string>::makeResult(true);
  }

  const RenderScene &scene = *host.scene();
  auto frameResult = ensureSceneFrameBuffer(*services_, scene, sceneFrame);
  if (frameResult.hasError()) {
    return frameResult;
  }
  sceneFrame.scene = &scene;
  sceneFrame.sceneTopologyVersion = scene.topologyVersion();
  sceneFrame.renderableCount = scene.renderables().size();

  size_t sampleDispatchCount = 0u;
  size_t worldDispatchCount = 0u;
  size_t scatterDispatchCount = 0u;
  size_t morphDispatchCount = 0u;
  size_t skinPaletteDispatchCount = 0u;
  size_t skinDispatchCount = 0u;
  for (auto &[handle, instance] : impl_->instances) {
    (void)handle;
    if (scene.topologyVersion() != instance.cachedSceneTopologyVersion ||
        services_->gpu().geometryMutationVersion() !=
            instance.cachedGeometryMutationVersion) {
      auto bindingResult = ensureRenderableBindings(host, *services_, instance);
      if (bindingResult.hasError()) {
        return bindingResult;
      }
    }
    if (!instance.channels.empty()) {
      ++sampleDispatchCount;
    }
    worldDispatchCount += instance.depthBucketStarts.size();
    if (!instance.renderableBindings.empty()) {
      ++scatterDispatchCount;
    }
    for (const AnimatedRenderableState &animated :
         instance.animatedRenderables) {
      morphDispatchCount += animated.hasMorph ? 1u : 0u;
      skinPaletteDispatchCount += animated.hasSkin ? 1u : 0u;
      skinDispatchCount += animated.hasSkin ? 1u : 0u;
    }
  }

  sceneFrame.preDispatches.reserve(
      sampleDispatchCount + worldDispatchCount + scatterDispatchCount +
      morphDispatchCount + skinPaletteDispatchCount + skinDispatchCount);
  sceneFrame.samplePushConstants.reserve(sampleDispatchCount);
  sceneFrame.sampleDependencies.reserve(sampleDispatchCount);
  sceneFrame.worldPushConstants.reserve(worldDispatchCount);
  sceneFrame.worldDependencies.reserve(worldDispatchCount);
  sceneFrame.scatterPushConstants.reserve(scatterDispatchCount);
  sceneFrame.scatterDependencies.reserve(scatterDispatchCount);
  sceneFrame.morphPushConstants.reserve(morphDispatchCount);
  sceneFrame.morphDependencies.reserve(morphDispatchCount);
  sceneFrame.skinPalettePushConstants.reserve(skinPaletteDispatchCount);
  sceneFrame.skinPaletteDependencies.reserve(skinPaletteDispatchCount);
  sceneFrame.skinPushConstants.reserve(skinDispatchCount);
  sceneFrame.skinDependencies.reserve(skinDispatchCount);

  const uint64_t instanceMatricesAddress =
      services_->gpu().getBufferDeviceAddress(
          sceneFrame.instanceMatricesBuffer->handle());
  if (instanceMatricesAddress == 0u) {
    return Result<bool, std::string>::makeError(
        "AnimationPoseSimulationBackend: invalid scene frame buffer address");
  }

  for (auto &[handle, instance] : impl_->instances) {
    const SimulationRegistry::Record *record = host.registry_.tryGet(handle);
    if (record == nullptr || record->faulted) {
      continue;
    }
    const AnimationClipData &clip =
        instance.prefab.animations[instance.params.clipIndex];
    const float sampleTimeSeconds =
        computeRenderSampleTime(host, *record, instance, clip);

    auto uploadNodeStates =
        uploadVector(services_->gpu(), *instance.sampledNodeStatesBuffer,
                     instance.baseNodeStates);
    if (uploadNodeStates.hasError()) {
      return uploadNodeStates;
    }
    auto uploadWeights = uploadVector(
        services_->gpu(), *instance.sampledWeightsBuffer, instance.baseWeights);
    if (uploadWeights.hasError()) {
      return uploadWeights;
    }

    glm::mat4 rootTransform(1.0f);
    (void)scene.graph().getCachedNodeWorldTransform(instance.rootNode,
                                                    rootTransform);
    const uint64_t nodeStatesAddress = services_->gpu().getBufferDeviceAddress(
        instance.sampledNodeStatesBuffer->handle());
    const uint64_t sampledWeightsAddress =
        services_->gpu().getBufferDeviceAddress(
            instance.sampledWeightsBuffer->handle());
    const uint64_t worldMatricesAddress =
        services_->gpu().getBufferDeviceAddress(
            instance.worldMatricesBuffer->handle());
    if (nodeStatesAddress == 0u || sampledWeightsAddress == 0u ||
        worldMatricesAddress == 0u) {
      return Result<bool, std::string>::makeError(
          "AnimationPoseSimulationBackend: invalid dynamic buffer address");
    }

    if (!instance.channels.empty()) {
      sceneFrame.samplePushConstants.push_back(SamplePushConstants{
          .channelsAddress = services_->gpu().getBufferDeviceAddress(
              instance.channelsBuffer->handle()),
          .keyTimesAddress = services_->gpu().getBufferDeviceAddress(
              instance.keyTimesBuffer->handle()),
          .valuesAddress = services_->gpu().getBufferDeviceAddress(
              instance.valuesBuffer->handle()),
          .nodeStatesAddress = nodeStatesAddress,
          .sampledWeightsAddress = sampledWeightsAddress,
          .sampleTimeSeconds = sampleTimeSeconds,
          .channelCount = static_cast<uint32_t>(instance.channels.size()),
      });
      sceneFrame.sampleDependencies.push_back(
          {instance.channelsBuffer->handle(), instance.keyTimesBuffer->handle(),
           instance.sampledNodeStatesBuffer->handle(),
           instance.sampledWeightsBuffer->handle()});
      appendComputeDispatch(
          sceneFrame.preDispatches, services_->samplePipeline(),
          dispatchCount(sceneFrame.samplePushConstants.back().channelCount),
          sceneFrame.samplePushConstants.back(),
          sceneFrame.sampleDependencies.back(),
          sceneFrame.sampleDependencies.back().size(), "AnimationPose Sample",
          0xff5599ffu);
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
          {instance.sampledNodeStatesBuffer->handle(),
           instance.nodeMetaBuffer->handle(),
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
           sceneFrame.instanceMatricesBuffer->handle(), BufferHandle{}});
      appendComputeDispatch(
          sceneFrame.preDispatches, services_->scatterPipeline(),
          dispatchCount(sceneFrame.scatterPushConstants.back().bindingCount),
          sceneFrame.scatterPushConstants.back(),
          sceneFrame.scatterDependencies.back(), size_t{3u},
          "AnimationPose Scatter", 0xff33cc88u);
    }

    for (AnimatedRenderableState &animated : instance.animatedRenderables) {
      BufferHandle overrideBuffer{};

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

      if (animated.hasMorph && animated.morphOutputVertexBuffer &&
          animated.morphOutputVertexBuffer->valid() &&
          animated.nodeIndex < instance.nodeMeta.size()) {
        const uint64_t outputVertexAddress =
            services_->gpu().getBufferDeviceAddress(
                animated.morphOutputVertexBuffer->handle());
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
               instance.sampledWeightsBuffer->handle(),
               animated.morphOutputVertexBuffer->handle()});
          appendComputeDispatch(sceneFrame.preDispatches,
                                services_->morphPipeline(),
                                dispatchCount(animated.vertexCount),
                                sceneFrame.morphPushConstants.back(),
                                sceneFrame.morphDependencies.back(),
                                sceneFrame.morphDependencies.back().size(),
                                "AnimationPose Morph", 0xffaa7733u);
          if (!animated.hasSkin) {
            overrideBuffer = animated.morphOutputVertexBuffer->handle();
          }
        }
      }

      if (animated.hasSkin && animated.finalOutputVertexBuffer &&
          animated.finalOutputVertexBuffer->valid()) {
        const uint64_t skinSourceAddress =
            animated.hasMorph && animated.morphOutputVertexBuffer
                ? services_->gpu().getBufferDeviceAddress(
                      animated.morphOutputVertexBuffer->handle())
                : animated.sourceVertexAddress;
        const BufferHandle skinSourceBuffer =
            animated.hasMorph && animated.morphOutputVertexBuffer
                ? animated.morphOutputVertexBuffer->handle()
                : animated.sourceVertexBuffer;
        const uint64_t outputVertexAddress =
            services_->gpu().getBufferDeviceAddress(
                animated.finalOutputVertexBuffer->handle());
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
               animated.finalOutputVertexBuffer->handle()});
          appendComputeDispatch(sceneFrame.preDispatches,
                                services_->skinPipeline(),
                                dispatchCount(animated.vertexCount),
                                sceneFrame.skinPushConstants.back(),
                                sceneFrame.skinDependencies.back(),
                                sceneFrame.skinDependencies.back().size(),
                                "AnimationPose Skin", 0xffcc8844u);
          overrideBuffer = animated.finalOutputVertexBuffer->handle();
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
      }
    }
  }

  ++sceneFrame.version;
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
    }
    impl_->instances.clear();
    impl_->sceneFrame = SceneFrameState(memory_);
  }
}

const AnimationSceneFrameData *
AnimationPoseSimulationBackend::currentSceneFrameData() const noexcept {
  if (impl_ == nullptr || impl_->sceneFrame.version == 0u ||
      impl_->instances.empty() || !impl_->sceneFrame.instanceMatricesBuffer ||
      !impl_->sceneFrame.instanceMatricesBuffer->valid() ||
      services_ == nullptr) {
    return nullptr;
  }
  AnimationSceneFrameData &frameData = impl_->sceneFrame.publishedData;
  frameData.instanceMatricesBuffer =
      impl_->sceneFrame.instanceMatricesBuffer->handle();
  frameData.instanceMatricesAddress = services_->gpu().getBufferDeviceAddress(
      impl_->sceneFrame.instanceMatricesBuffer->handle());
  frameData.preDispatches = std::span<const ComputeDispatchItem>(
      impl_->sceneFrame.preDispatches.data(),
      impl_->sceneFrame.preDispatches.size());
  frameData.geometryOverridesByRenderable =
      std::span<const AnimatedRenderableGeometryOverride>(
          impl_->sceneFrame.geometryOverrides.data(),
          impl_->sceneFrame.geometryOverrides.size());
  frameData.scene = impl_->sceneFrame.scene;
  frameData.sceneTopologyVersion = impl_->sceneFrame.sceneTopologyVersion;
  frameData.renderableCount = impl_->sceneFrame.renderableCount;
  frameData.version = impl_->sceneFrame.version;
  return frameData.instanceMatricesAddress != 0u ? &frameData : nullptr;
}

} // namespace nuri
