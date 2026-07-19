#pragma once
#include "nuri/pch.h"
#include <cstdint>
#include <glm/glm.hpp>
#include <limits>
#include <memory_resource>
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
  using allocator_type = std::pmr::polymorphic_allocator<std::byte>;
  std::pmr::vector<float> keyTimes;
  std::pmr::vector<float> values;
  AnimationInterpolation interpolation = AnimationInterpolation::Linear;
  uint32_t valueArity = 0;
  template <typename Alloc>
    requires std::is_constructible_v<allocator_type, const Alloc &>
  [[nodiscard]] static std::pmr::memory_resource *
  resourceFromAlloc(const Alloc &alloc) {
    return allocator_type(alloc).resource();
  }
  template <typename Alloc>
    requires std::is_constructible_v<allocator_type, const Alloc &>
  AnimationSamplerData(std::allocator_arg_t, const Alloc &alloc)
      : AnimationSamplerData(resourceFromAlloc(alloc)) {}
  template <typename Alloc>
    requires std::is_constructible_v<allocator_type, const Alloc &>
  AnimationSamplerData(std::allocator_arg_t, const Alloc &alloc,
                       std::pmr::memory_resource *memory)
      : AnimationSamplerData(memory != nullptr ? memory
                                               : resourceFromAlloc(alloc)) {}
  template <typename Alloc>
    requires std::is_constructible_v<allocator_type, const Alloc &>
  AnimationSamplerData(std::allocator_arg_t, const Alloc &alloc,
                       const AnimationSamplerData &other)
      : keyTimes(other.keyTimes, resourceFromAlloc(alloc)),
        values(other.values, resourceFromAlloc(alloc)),
        interpolation(other.interpolation), valueArity(other.valueArity) {}
  template <typename Alloc>
    requires std::is_constructible_v<allocator_type, const Alloc &>
  AnimationSamplerData(std::allocator_arg_t, const Alloc &alloc,
                       AnimationSamplerData &&other)
      : keyTimes(std::move(other.keyTimes), resourceFromAlloc(alloc)),
        values(std::move(other.values), resourceFromAlloc(alloc)),
        interpolation(other.interpolation), valueArity(other.valueArity) {}
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
  AnimationClipData(const AnimationClipData &other,
                    std::pmr::memory_resource *memory)
      : name(other.name, memory), samplers(other.samplers, memory),
        channels(other.channels, memory),
        durationSeconds(other.durationSeconds) {}
};

struct SkinData {
  std::pmr::string name;
  uint32_t skeletonRootNodeIndex = std::numeric_limits<uint32_t>::max();
  std::pmr::vector<uint32_t> jointNodeIndices;
  std::pmr::vector<glm::mat4> inverseBindMatrices;
  explicit SkinData(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : name(memory), jointNodeIndices(memory), inverseBindMatrices(memory) {}
  SkinData(const SkinData &other, std::pmr::memory_resource *memory)
      : name(other.name, memory),
        skeletonRootNodeIndex(other.skeletonRootNodeIndex),
        jointNodeIndices(other.jointNodeIndices, memory),
        inverseBindMatrices(other.inverseBindMatrices, memory) {}
};

} // namespace nuri
