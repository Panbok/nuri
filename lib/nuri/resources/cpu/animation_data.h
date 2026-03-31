#pragma once

#include "nuri/pch.h"

#include <cstdint>
#include <limits>
#include <memory_resource>

#include <glm/glm.hpp>

namespace nuri {

enum class AnimationTargetPath : uint8_t {
  Translation = 0,
  Rotation = 1,
  Scale = 2,
  Weights = 3,
};

enum class AnimationInterpolation : uint8_t {
  Step = 0,
  Linear = 1,
  CubicSpline = 2,
};

struct AnimationSamplerData {
  std::pmr::vector<float> keyTimes;
  std::pmr::vector<float> values;
  AnimationInterpolation interpolation = AnimationInterpolation::Linear;
  uint32_t valueArity = 0;

  explicit AnimationSamplerData(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : keyTimes(memory), values(memory) {}
};

struct AnimationChannelData {
  uint32_t samplerIndex = std::numeric_limits<uint32_t>::max();
  uint32_t targetNodeIndex = std::numeric_limits<uint32_t>::max();
  AnimationTargetPath path = AnimationTargetPath::Translation;
};

struct AnimationClipData {
  std::pmr::string name;
  std::pmr::vector<AnimationSamplerData> samplers;
  std::pmr::vector<AnimationChannelData> channels;
  float durationSeconds = 0.0f;

  explicit AnimationClipData(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : name(memory), samplers(memory), channels(memory) {}
};

struct SkinData {
  std::pmr::string name;
  uint32_t skeletonRootNodeIndex = std::numeric_limits<uint32_t>::max();
  std::pmr::vector<uint32_t> jointNodeIndices;
  std::pmr::vector<glm::mat4> inverseBindMatrices;

  explicit SkinData(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : name(memory), jointNodeIndices(memory), inverseBindMatrices(memory) {}
};

} // namespace nuri
