#include "nuri/scene/scene_animation_player.h"
#include "nuri/core/pmr_scratch.h"
#include "nuri/math/utils.h"
#include "nuri/pch.h"
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/matrix_decompose.hpp>
namespace nuri {
namespace {
struct SampleWindow {
  uint32_t leftIndex = 0;
  uint32_t rightIndex = 0;
  float t = 0.0f;
};
[[nodiscard]] std::pmr::memory_resource *
memoryOrDefault(std::pmr::memory_resource *memory) noexcept {
  return memory != nullptr ? memory : std::pmr::get_default_resource();
}
[[nodiscard]] glm::mat4 composeTransform(const glm::vec3 &translation,
                                         const glm::quat &rotation,
                                         const glm::vec3 &scale) {
  return glm::translate(glm::mat4(1.0f), translation) *
         glm::mat4_cast(rotation) * glm::scale(glm::mat4(1.0f), scale);
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
[[nodiscard]] SampleWindow findSampleWindow(std::span<const float> keyTimes,
                                            float timeSeconds) {
  if (keyTimes.size() == 1u || timeSeconds <= keyTimes.front()) {
    return {};
  }
  if (timeSeconds >= keyTimes.back()) {
    return SampleWindow{
        .leftIndex = static_cast<uint32_t>(keyTimes.size() - 1u),
        .rightIndex = static_cast<uint32_t>(keyTimes.size() - 1u),
        .t = 0.0f,
    };
  }
  const auto upper =
      std::upper_bound(keyTimes.begin(), keyTimes.end(), timeSeconds);
  const uint32_t rightIndex =
      static_cast<uint32_t>(std::distance(keyTimes.begin(), upper));
  const uint32_t leftIndex = rightIndex - 1u;
  const float leftTime = keyTimes[leftIndex];
  const float rightTime = keyTimes[rightIndex];
  const float duration = rightTime - leftTime;
  const float t = duration > 0.0f ? (timeSeconds - leftTime) / duration : 0.0f;
  return SampleWindow{.leftIndex = leftIndex, .rightIndex = rightIndex, .t = t};
}
[[nodiscard]] size_t sampleValueOffset(const AnimationSamplerData &sampler,
                                       uint32_t keyIndex) noexcept {
  const size_t keyStride =
      sampler.interpolation == AnimationInterpolation::CubicSpline ? 3u : 1u;
  const size_t valueIndex =
      static_cast<size_t>(keyIndex) * keyStride + (keyStride == 3u ? 1u : 0u);
  return valueIndex * sampler.valueArity;
}
[[nodiscard]] glm::quat readQuaternionValue(const AnimationSamplerData &sampler,
                                            uint32_t keyIndex) {
  const size_t base = sampleValueOffset(sampler, keyIndex);
  return glm::normalize(
      glm::quat(sampler.values[base + 3u], sampler.values[base],
                sampler.values[base + 1u], sampler.values[base + 2u]));
}
[[nodiscard]] glm::quat
sampleRotationSampler(const AnimationSamplerData &sampler, float timeSeconds) {
  const SampleWindow window = findSampleWindow(sampler.keyTimes, timeSeconds);
  if (window.leftIndex == window.rightIndex ||
      sampler.interpolation == AnimationInterpolation::Step ||
      sampler.interpolation == AnimationInterpolation::CubicSpline) {
    return readQuaternionValue(sampler, window.leftIndex);
  }
  const glm::quat left = readQuaternionValue(sampler, window.leftIndex);
  glm::quat right = readQuaternionValue(sampler, window.rightIndex);
  float cosTheta = glm::dot(left, right);
  if (cosTheta < 0.0f) {
    right = -right;
    cosTheta = -cosTheta;
  }
  cosTheta = std::clamp(cosTheta, -1.0f, 1.0f);
  if (cosTheta > 0.9995f) {
    return glm::normalize(left + window.t * (right - left));
  }
  const float angle = std::acos(cosTheta);
  const float sinAngle = std::sin(angle);
  if (sinAngle <= 1.0e-5f) {
    return left;
  }
  const float leftWeight = std::sin((1.0f - window.t) * angle) / sinAngle;
  const float rightWeight = std::sin(window.t * angle) / sinAngle;
  return glm::normalize(left * leftWeight + right * rightWeight);
}
[[nodiscard]] float clampPlaybackTime(const AnimationClipData *clip,
                                      float timeSeconds) noexcept {
  const float maxTime = clip != nullptr && clip->durationSeconds > 0.0f
                            ? clip->durationSeconds
                            : 0.0f;
  return std::clamp(timeSeconds, 0.0f, maxTime);
}
void advancePlaybackTime(float &timeSeconds, bool &playing,
                         AnimationPlaybackMode mode, float deltaSeconds,
                         const AnimationClipData &clip) noexcept {
  timeSeconds = std::max(0.0f, timeSeconds + deltaSeconds);
  if (clip.durationSeconds <= 0.0f) {
    timeSeconds = 0.0f;
    playing = false;
    return;
  }
  if (mode == AnimationPlaybackMode::Loop) {
    timeSeconds = std::fmod(timeSeconds, clip.durationSeconds);
    return;
  }
  if (timeSeconds >= clip.durationSeconds) {
    timeSeconds = clip.durationSeconds;
    playing = false;
  }
}
void sampleSampler(const AnimationSamplerData &sampler, float timeSeconds,
                   std::span<float> out) {
  const SampleWindow window = findSampleWindow(sampler.keyTimes, timeSeconds);
  if (window.leftIndex == window.rightIndex ||
      sampler.interpolation == AnimationInterpolation::Step) {
    std::copy_n(sampler.values.begin() +
                    sampleValueOffset(sampler, window.leftIndex),
                sampler.valueArity, out.begin());
    return;
  }
  if (sampler.interpolation == AnimationInterpolation::Linear) {
    const size_t leftBase = sampleValueOffset(sampler, window.leftIndex);
    const size_t rightBase = sampleValueOffset(sampler, window.rightIndex);
    for (uint32_t i = 0; i < sampler.valueArity; ++i) {
      out[i] = glm::mix(sampler.values[leftBase + i],
                        sampler.values[rightBase + i], window.t);
    }
    return;
  }
  const size_t leftBase =
      static_cast<size_t>(window.leftIndex) * sampler.valueArity * 3u;
  const size_t rightBase =
      static_cast<size_t>(window.rightIndex) * sampler.valueArity * 3u;
  const float deltaTime =
      sampler.keyTimes[window.rightIndex] - sampler.keyTimes[window.leftIndex];
  const float t = glm::clamp(window.t, 0.0f, 1.0f);
  const float t2 = t * t;
  const float t3 = t2 * t;
  const float h00 = 2.0f * t3 - 3.0f * t2 + 1.0f;
  const float h10 = t3 - 2.0f * t2 + t;
  const float h01 = -2.0f * t3 + 3.0f * t2;
  const float h11 = t3 - t2;
  for (uint32_t i = 0; i < sampler.valueArity; ++i) {
    const float value0 = sampler.values[leftBase + sampler.valueArity + i];
    const float outTangent0 =
        sampler.values[leftBase + sampler.valueArity * 2u + i];
    const float value1 = sampler.values[rightBase + sampler.valueArity + i];
    const float inTangent1 = sampler.values[rightBase + i];
    out[i] = h00 * value0 + h10 * deltaTime * outTangent0 + h01 * value1 +
             h11 * deltaTime * inTangent1;
  }
}
} // namespace

SceneAnimationPlayer::SceneAnimationPlayer(
    const ScenePrefab &prefab, const SceneInstantiationMap &instantiationMap,
    std::pmr::memory_resource *memory)
    : prefab_(prefab), instantiationMap_(instantiationMap),
      baseNodeStates_(memoryOrDefault(memory)),
      sampledNodeStates_(memoryOrDefault(memory)) {
  memory = memoryOrDefault(memory);
  baseNodeStates_.reserve(prefab.nodes.size());
  sampledNodeStates_.reserve(prefab.nodes.size());
  for (const ScenePrefabNode &node : prefab.nodes) {
    baseNodeStates_.emplace_back(memory);
    sampledNodeStates_.emplace_back(memory);
    NodeState &base = baseNodeStates_.back();
    decomposeTransform(node.localFromParent, base.translation, base.rotation,
                       base.scale);
    base.morphWeights.assign(node.morphWeights.begin(),
                             node.morphWeights.end());
    sampledNodeStates_.back() = base;
  }
}

const AnimationClipData *SceneAnimationPlayer::activeClip() const noexcept {
  if (clipIndex_ >= prefab_.animations.size()) {
    return nullptr;
  }
  return &prefab_.animations[clipIndex_];
}

Result<bool, std::string>
SceneAnimationPlayer::play(uint32_t clipIndex, AnimationPlaybackMode mode) {
  if (clipIndex >= prefab_.animations.size()) {
    return Result<bool, std::string>::makeError(
        "SceneAnimationPlayer::play: clip index is out of range");
  }
  clipIndex_ = clipIndex;
  timeSeconds_ = 0.0f;
  playing_ = true;
  mode_ = mode;
  sampledNodeStates_ = baseNodeStates_;
  return Result<bool, std::string>::makeResult(true);
}

void SceneAnimationPlayer::stop() { playing_ = false; }

void SceneAnimationPlayer::seek(float timeSeconds) {
  timeSeconds_ = clampPlaybackTime(activeClip(), timeSeconds);
}

void SceneAnimationPlayer::applyClip(SceneGraph &graph) const {
  const AnimationClipData *clip = activeClip();
  if (clip == nullptr) {
    return;
  }
  ScratchArena scratch;
  ScopedScratch scopedScratch(scratch);
  uint32_t maxValueArity = 0u;
  for (const AnimationSamplerData &sampler : clip->samplers) {
    maxValueArity = std::max(maxValueArity, sampler.valueArity);
  }
  std::pmr::vector<float> sampleBuffer(scopedScratch.resource());
  sampleBuffer.resize(maxValueArity);
  for (const AnimationChannelData &channel : clip->channels) {
    const AnimationSamplerData &sampler = clip->samplers[channel.samplerIndex];
    NodeState &nodeState = sampledNodeStates_[channel.targetNodeIndex];
    switch (channel.path) {
    case AnimationTargetPath::Translation:
    case AnimationTargetPath::Scale:
    case AnimationTargetPath::Weights: {
      std::span<float> sample(sampleBuffer.data(), sampler.valueArity);
      sampleSampler(sampler, timeSeconds_, sample);
      if (channel.path == AnimationTargetPath::Weights) {
        nodeState.morphWeights.assign(sample.begin(), sample.end());
      } else {
        nodeState.*(channel.path == AnimationTargetPath::Translation
                        ? &NodeState::translation
                        : &NodeState::scale) =
            glm::vec3(sample[0], sample[1], sample[2]);
      }
      break;
    }
    case AnimationTargetPath::Rotation:
      nodeState.rotation = sampleRotationSampler(sampler, timeSeconds_);
      break;
    }
  }
  for (uint32_t nodeIndex = 0; nodeIndex < sampledNodeStates_.size();
       ++nodeIndex) {
    if (!isValid(instantiationMap_.nodes[nodeIndex])) {
      continue;
    }
    const NodeState &state = sampledNodeStates_[nodeIndex];
    (void)graph.setNodeLocalTransform(
        instantiationMap_.nodes[nodeIndex],
        composeTransform(state.translation, state.rotation, state.scale));
    graph.forEachRenderableOnNode(
        instantiationMap_.nodes[nodeIndex], [&](RenderableId renderableId) {
          (void)graph.setRenderableMorphWeights(
              renderableId, std::span<const float>(state.morphWeights.data(),
                                                   state.morphWeights.size()));
        });
  }
  (void)graph.syncWorldTransforms();
  uint32_t maxJointCount = 0u;
  for (const SkinData &skin : prefab_.skins) {
    maxJointCount = std::max(
        maxJointCount, static_cast<uint32_t>(skin.jointNodeIndices.size()));
  }
  std::pmr::vector<glm::mat4> palette(scopedScratch.resource());
  palette.resize(maxJointCount, glm::mat4(1.0f));
  for (uint32_t renderableIndex = 0;
       renderableIndex < prefab_.renderables.size(); ++renderableIndex) {
    const ScenePrefabRenderable &prefabRenderable =
        prefab_.renderables[renderableIndex];
    if (!isValid(instantiationMap_.renderables[renderableIndex]) ||
        prefabRenderable.skinIndex >= prefab_.skins.size()) {
      continue;
    }
    const SkinData &skin = prefab_.skins[prefabRenderable.skinIndex];
    std::fill_n(palette.begin(), skin.jointNodeIndices.size(), glm::mat4(1.0f));
    for (uint32_t jointIndex = 0; jointIndex < skin.jointNodeIndices.size();
         ++jointIndex) {
      glm::mat4 jointWorld(1.0f);
      if (!graph.getCachedNodeWorldTransform(
              instantiationMap_.nodes[skin.jointNodeIndices[jointIndex]],
              jointWorld)) {
        continue;
      }
      const glm::mat4 inverseBind = jointIndex < skin.inverseBindMatrices.size()
                                        ? skin.inverseBindMatrices[jointIndex]
                                        : glm::mat4(1.0f);
      palette[jointIndex] = jointWorld * inverseBind;
    }
    (void)graph.setRenderableSkinPalette(
        instantiationMap_.renderables[renderableIndex],
        std::span<const glm::mat4>(palette.data(),
                                   skin.jointNodeIndices.size()));
  }
}

void SceneAnimationPlayer::update(SceneGraph &graph, float deltaSeconds) {
  const AnimationClipData *clip = activeClip();
  if (clip == nullptr) {
    return;
  }
  if (playing_) {
    advancePlaybackTime(timeSeconds_, playing_, mode_, deltaSeconds, *clip);
  }
  applyClip(graph);
}

} // namespace nuri
