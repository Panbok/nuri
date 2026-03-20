#pragma once

#include "nuri/defines.h"
#include "nuri/resources/gpu/resource_handles.h"

#include <cmath>
#include <cstdint>
#include <string>

#include "nuri/math/utils.h"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace nuri {

enum class LightType : uint8_t {
  Directional = 0,
  Point = 1,
  Spot = 2,
};

struct NURI_API LightId {
  LightType type = LightType::Directional;
  uint32_t value = 0;

  constexpr bool operator==(LightId other) const noexcept {
    return type == other.type && value == other.value;
  }
  constexpr bool operator!=(LightId other) const noexcept {
    return !(*this == other);
  }
};

inline constexpr LightId kInvalidLightId{};

[[nodiscard]] constexpr bool isValid(LightId id) noexcept {
  return id.value != 0u;
}

[[nodiscard]] constexpr uint32_t indexOf(LightId id) noexcept {
  return unpackResourceHandle(id.value).index;
}

[[nodiscard]] constexpr uint32_t generationOf(LightId id) noexcept {
  return unpackResourceHandle(id.value).generation;
}

[[nodiscard]] constexpr LightId makeLightId(LightType type, uint32_t index,
                                            uint32_t generation) noexcept {
  return LightId{type, packResourceHandle(index, generation)};
}

// LocalLightGpuType starts at zero because directional lights are packed in a
// separate GPU buffer. Do not cast directly between LocalLightGpuType and
// LightType: LightType::Point is 1 while LocalLightGpuType::Point is 0.
enum class LocalLightGpuType : uint32_t {
  Point = 0u,
  Spot = 1u,
};

struct NURI_API LightDesc {
  LightType type = LightType::Directional;
  std::string name{};
  glm::vec3 position{0.0f};
  glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
  glm::vec3 color{1.0f};
  float intensity = 1.0f;
  float range = 0.0f;
  float innerConeAngleRadians = 0.0f;
  float outerConeAngleRadians = glm::quarter_pi<float>();
  bool enabled = true;
};

struct alignas(16) DirectionalLightGpuData {
  glm::vec4 directionIlluminance{0.0f, -1.0f, 0.0f, 0.0f};
  glm::vec4 colorReserved{1.0f, 1.0f, 1.0f, 0.0f};
};
static_assert(sizeof(DirectionalLightGpuData) == 2u * sizeof(glm::vec4),
              "DirectionalLightGpuData must remain std430-friendly");

struct alignas(16) LocalLightGpuData {
  glm::vec4 positionRange{0.0f, 0.0f, 0.0f, 0.0f};
  glm::vec4 directionOuterCos{0.0f, 0.0f, -1.0f, -1.0f};
  glm::vec4 colorIntensity{1.0f, 1.0f, 1.0f, 1.0f};
  // LocalLightGpuData::innerCosTypeEnabledReserved must stay std430-friendly
  // and match the shader unpacking: .x stores inner cone cosine as float bits,
  // .y stores LocalLightGpuType, .z stores the enabled flag plus future packed
  // bits, and .w is reserved. Mirror any layout changes in shader decode code.
  glm::uvec4 innerCosTypeEnabledReserved{0u, 0u, 0u, 0u};
};
static_assert(sizeof(LocalLightGpuData) == 4u * sizeof(glm::vec4),
              "LocalLightGpuData must remain std430-friendly");

[[nodiscard]] inline glm::vec3
lightDirectionFromRotationForLocalLights(const glm::quat &rotation) {
  // Keep a local implementation here to avoid including `nuri/math/light.h`
  // (which already includes this header and would create a cycle).
  constexpr float kMinLength = 1.0e-6f;
  const glm::vec3 direction =
      sanitizeRotation(rotation) * glm::vec3(0.0f, 0.0f, -1.0f);
  const float length = glm::length(direction);
  if (!std::isfinite(length) || length <= kMinLength) {
    return glm::vec3(0.0f, 0.0f, -1.0f);
  }
  return direction / length;
}

[[nodiscard]] inline DirectionalLightGpuData
packDirectionalLight(const glm::quat &rotation, const glm::vec3 &color,
                     float intensity) {
  const glm::vec3 direction =
      lightDirectionFromRotationForLocalLights(rotation);
  return DirectionalLightGpuData{
      .directionIlluminance =
          glm::vec4(direction.x, direction.y, direction.z, intensity),
      .colorReserved = glm::vec4(color, 0.0f),
  };
}

[[nodiscard]] inline LocalLightGpuData
packPointLight(const glm::vec3 &position, const glm::quat &rotation,
               const glm::vec3 &color, float intensity, float range,
               bool enabled) {
  const glm::vec3 direction =
      lightDirectionFromRotationForLocalLights(rotation);
  return LocalLightGpuData{
      .positionRange = glm::vec4(position, range),
      .directionOuterCos = glm::vec4(direction, -1.0f),
      .colorIntensity = glm::vec4(color, intensity),
      .innerCosTypeEnabledReserved =
          glm::uvec4(floatBitsToUint(-1.0f),
                     static_cast<uint32_t>(LocalLightGpuType::Point),
                     enabled ? 1u : 0u, 0u),
  };
}

[[nodiscard]] inline LocalLightGpuData
packSpotLight(const glm::vec3 &position, const glm::quat &rotation,
              const glm::vec3 &color, float intensity, float range,
              float innerConeAngleRadians, float outerConeAngleRadians) {
  const glm::vec3 direction =
      lightDirectionFromRotationForLocalLights(rotation);
  return LocalLightGpuData{
      .positionRange = glm::vec4(position, range),
      .directionOuterCos =
          glm::vec4(direction, std::cos(outerConeAngleRadians)),
      .colorIntensity = glm::vec4(color, intensity),
      .innerCosTypeEnabledReserved =
          glm::uvec4(floatBitsToUint(std::cos(innerConeAngleRadians)),
                     static_cast<uint32_t>(LocalLightGpuType::Spot), 1u, 0u),
  };
}

template <typename Store>
[[nodiscard]] inline LightDesc
makeLocalLightDesc(const Store &store, uint32_t index, LightType type) {
  LightDesc out{};
  out.type = type;
  out.name = store.names[index];
  out.position = store.localPositions[index];
  out.rotation = store.localRotations[index];
  out.color = store.colors[index];
  out.intensity = store.intensities[index];
  if constexpr (requires { store.ranges[index]; }) {
    out.range = store.ranges[index];
  }
  if constexpr (requires { store.innerConeAngles[index]; }) {
    out.innerConeAngleRadians = store.innerConeAngles[index];
  }
  if constexpr (requires { store.outerConeAngles[index]; }) {
    out.outerConeAngleRadians = store.outerConeAngles[index];
  }
  out.enabled = store.enabled[index] != 0u;
  return out;
}

} // namespace nuri
