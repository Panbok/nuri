#include "nuri/pch.h"

#include "nuri/scene/scene_animation_player.h"

#include "nuri/core/pmr_scratch.h"
#include "nuri/math/utils.h"

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
  if (keyTimes.empty()) {
    return {};
  }
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

void readLinearValue(std::span<const float> values, uint32_t keyIndex,
                     uint32_t valueArity, std::span<float> out) {
  const size_t base = static_cast<size_t>(keyIndex) * valueArity;
  for (uint32_t i = 0; i < valueArity; ++i) {
    out[i] = values[base + i];
  }
}

void sampleSampler(const AnimationSamplerData &sampler, float timeSeconds,
                   std::span<float> out) {
  if (sampler.keyTimes.empty() || sampler.valueArity == 0u) {
    std::fill(out.begin(), out.end(), 0.0f);
    return;
  }
  const std::span<const float> keyTimes(sampler.keyTimes.data(),
                                        sampler.keyTimes.size());
  const std::span<const float> values(sampler.values.data(),
                                      sampler.values.size());
  const SampleWindow window = findSampleWindow(keyTimes, timeSeconds);
  if (window.leftIndex == window.rightIndex) {
    if (sampler.interpolation == AnimationInterpolation::CubicSpline) {
      const size_t base =
          static_cast<size_t>(window.leftIndex) * sampler.valueArity * 3u +
          sampler.valueArity;
      for (uint32_t i = 0; i < sampler.valueArity; ++i) {
        out[i] = values[base + i];
      }
    } else {
      readLinearValue(values, window.leftIndex, sampler.valueArity, out);
    }
    return;
  }

  if (sampler.interpolation == AnimationInterpolation::Step) {
    readLinearValue(values, window.leftIndex, sampler.valueArity, out);
    return;
  }
  if (sampler.interpolation == AnimationInterpolation::Linear) {
    const size_t leftBase =
        static_cast<size_t>(window.leftIndex) * sampler.valueArity;
    const size_t rightBase =
        static_cast<size_t>(window.rightIndex) * sampler.valueArity;
    for (uint32_t i = 0; i < sampler.valueArity; ++i) {
      out[i] = glm::mix(values[leftBase + i], values[rightBase + i], window.t);
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
    const float value0 = values[leftBase + sampler.valueArity + i];
    const float outTangent0 = values[leftBase + sampler.valueArity * 2u + i];
    const float value1 = values[rightBase + sampler.valueArity + i];
    const float inTangent1 = values[rightBase + i];
    out[i] = h00 * value0 + h10 * deltaTime * outTangent0 + h01 * value1 +
             h11 * deltaTime * inTangent1;
  }
}

} // namespace

SceneAnimationPlayer::SceneAnimationPlayer(
    const ScenePrefab &prefab, const SceneInstantiationMap &instantiationMap,
    std::pmr::memory_resource *memory)
    : prefab_(&prefab), instantiationMap_(&instantiationMap),
      memory_(memory != nullptr ? memory : std::pmr::get_default_resource()),
      baseNodeStates_(memory_), sampledNodeStates_(memory_) {
  baseNodeStates_.reserve(prefab.nodes.size());
  sampledNodeStates_.reserve(prefab.nodes.size());
  for (const ScenePrefabNode &node : prefab.nodes) {
    baseNodeStates_.emplace_back(memory_);
    sampledNodeStates_.emplace_back(memory_);
    NodeState &base = baseNodeStates_.back();
    decomposeTransform(node.localFromParent, base.translation, base.rotation,
                       base.scale);
    base.morphWeights.assign(node.morphWeights.begin(),
                             node.morphWeights.end());
  }
}

const AnimationClipData *SceneAnimationPlayer::activeClip() const noexcept {
  if (prefab_ == nullptr || clipIndex_ >= prefab_->animations.size()) {
    return nullptr;
  }
  return &prefab_->animations[clipIndex_];
}

Result<bool, std::string>
SceneAnimationPlayer::play(uint32_t clipIndex, AnimationPlaybackMode mode) {
  if (prefab_ == nullptr || clipIndex >= prefab_->animations.size()) {
    return Result<bool, std::string>::makeError(
        "SceneAnimationPlayer::play: clip index is out of range");
  }
  clipIndex_ = clipIndex;
  timeSeconds_ = 0.0f;
  playing_ = true;
  mode_ = mode;
  return Result<bool, std::string>::makeResult(true);
}

void SceneAnimationPlayer::stop() { playing_ = false; }

void SceneAnimationPlayer::seek(float timeSeconds) {
  timeSeconds_ = std::max(timeSeconds, 0.0f);
}

void SceneAnimationPlayer::applyClip(SceneGraph &graph) const {
  const AnimationClipData *clip = activeClip();
  if (clip == nullptr || prefab_ == nullptr || instantiationMap_ == nullptr) {
    return;
  }
  ScratchArena scratch;
  ScopedScratch scopedScratch(scratch);

  NURI_ASSERT(sampledNodeStates_.size() == baseNodeStates_.size(),
              "SceneAnimationPlayer: node state buffers are out of sync");
  for (size_t nodeIndex = 0; nodeIndex < baseNodeStates_.size(); ++nodeIndex) {
    const NodeState &base = baseNodeStates_[nodeIndex];
    NodeState &sampled = sampledNodeStates_[nodeIndex];
    sampled.translation = base.translation;
    sampled.rotation = base.rotation;
    sampled.scale = base.scale;
    sampled.morphWeights.assign(base.morphWeights.begin(),
                                base.morphWeights.end());
  }
  uint32_t maxValueArity = 0u;
  for (const AnimationSamplerData &sampler : clip->samplers) {
    maxValueArity = std::max(maxValueArity, sampler.valueArity);
  }
  std::pmr::vector<float> sampleBuffer(scopedScratch.resource());
  sampleBuffer.resize(maxValueArity);
  for (const AnimationChannelData &channel : clip->channels) {
    if (channel.targetNodeIndex >= sampledNodeStates_.size() ||
        channel.samplerIndex >= clip->samplers.size()) {
      continue;
    }
    const AnimationSamplerData &sampler = clip->samplers[channel.samplerIndex];
    NodeState &nodeState = sampledNodeStates_[channel.targetNodeIndex];
    std::span<float> sample(sampleBuffer.data(), sampler.valueArity);
    sampleSampler(sampler, timeSeconds_, sample);
    switch (channel.path) {
    case AnimationTargetPath::Translation:
      if (sampler.valueArity >= 3u) {
        nodeState.translation = glm::vec3(sample[0], sample[1], sample[2]);
      }
      break;
    case AnimationTargetPath::Rotation:
      if (sampler.valueArity >= 4u) {
        nodeState.rotation = glm::normalize(
            glm::quat(sample[3], sample[0], sample[1], sample[2]));
      }
      break;
    case AnimationTargetPath::Scale:
      if (sampler.valueArity >= 3u) {
        nodeState.scale = glm::vec3(sample[0], sample[1], sample[2]);
      }
      break;
    case AnimationTargetPath::Weights:
      nodeState.morphWeights.assign(sample.begin(), sample.end());
      break;
    }
  }

  for (uint32_t nodeIndex = 0; nodeIndex < sampledNodeStates_.size();
       ++nodeIndex) {
    if (nodeIndex >= instantiationMap_->nodes.size() ||
        !isValid(instantiationMap_->nodes[nodeIndex])) {
      continue;
    }
    const NodeState &state = sampledNodeStates_[nodeIndex];
    (void)graph.setNodeLocalTransform(
        instantiationMap_->nodes[nodeIndex],
        composeTransform(state.translation, state.rotation, state.scale));
    graph.forEachRenderableOnNode(
        instantiationMap_->nodes[nodeIndex], [&](RenderableId renderableId) {
          (void)graph.setRenderableMorphWeights(
              renderableId, std::span<const float>(state.morphWeights.data(),
                                                   state.morphWeights.size()));
        });
  }

  (void)graph.syncWorldTransforms();
  uint32_t maxJointCount = 0u;
  for (const SkinData &skin : prefab_->skins) {
    maxJointCount = std::max(
        maxJointCount, static_cast<uint32_t>(skin.jointNodeIndices.size()));
  }
  std::pmr::vector<glm::mat4> palette(scopedScratch.resource());
  palette.resize(maxJointCount, glm::mat4(1.0f));
  for (uint32_t renderableIndex = 0;
       renderableIndex < prefab_->renderables.size(); ++renderableIndex) {
    const ScenePrefabRenderable &prefabRenderable =
        prefab_->renderables[renderableIndex];
    if (renderableIndex >= instantiationMap_->renderables.size() ||
        !isValid(instantiationMap_->renderables[renderableIndex]) ||
        prefabRenderable.skinIndex >= prefab_->skins.size()) {
      continue;
    }
    const SkinData &skin = prefab_->skins[prefabRenderable.skinIndex];
    std::fill_n(palette.begin(), skin.jointNodeIndices.size(), glm::mat4(1.0f));
    for (uint32_t jointIndex = 0; jointIndex < skin.jointNodeIndices.size();
         ++jointIndex) {
      if (skin.jointNodeIndices[jointIndex] >=
          instantiationMap_->nodes.size()) {
        continue;
      }
      glm::mat4 jointWorld(1.0f);
      if (!graph.getCachedNodeWorldTransform(
              instantiationMap_->nodes[skin.jointNodeIndices[jointIndex]],
              jointWorld)) {
        continue;
      }
      const glm::mat4 inverseBind = jointIndex < skin.inverseBindMatrices.size()
                                        ? skin.inverseBindMatrices[jointIndex]
                                        : glm::mat4(1.0f);
      palette[jointIndex] = jointWorld * inverseBind;
    }
    (void)graph.setRenderableSkinPalette(
        instantiationMap_->renderables[renderableIndex],
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
    timeSeconds_ = std::max(0.0f, timeSeconds_ + deltaSeconds);
    if (clip->durationSeconds > 0.0f) {
      if (mode_ == AnimationPlaybackMode::Loop) {
        timeSeconds_ = std::fmod(timeSeconds_, clip->durationSeconds);
      } else if (timeSeconds_ >= clip->durationSeconds) {
        timeSeconds_ = clip->durationSeconds;
        playing_ = false;
      }
    }
  }
  applyClip(graph);
}

} // namespace nuri
