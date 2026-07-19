#pragma once
#include "nuri/defines.h"
#include "nuri/math/utils.h"
#include "nuri/resources/gpu/resource_handles.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <string>
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
};

inline constexpr LightId kInvalidLightId{};

[[nodiscard]] constexpr bool isValid(LightId id) noexcept {
  return id.value != 0u;
}

[[nodiscard]] constexpr uint32_t indexOf(LightId id) noexcept {
  return id.value & kResourceHandleIndexMask;
}

[[nodiscard]] constexpr uint32_t generationOf(LightId id) noexcept {
  return id.value >> kResourceHandleIndexBits;
}

[[nodiscard]] constexpr LightId makeLightId(LightType type, uint32_t index,
                                            uint32_t generation) noexcept {
  return LightId{type, packResourceHandle(index, generation)};
}

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
  float angularRadiusDegrees = 0.27f;
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
  glm::uvec4 innerCosTypeEnabledReserved{0u, 0u, 0u, 0u};
};
static_assert(sizeof(LocalLightGpuData) == 4u * sizeof(glm::vec4),
              "LocalLightGpuData must remain std430-friendly");

[[nodiscard]] inline glm::vec3
lightDirectionFromRotationForLocalLights(const glm::quat &rotation) {
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
                     float intensity, float angularRadiusDegrees = 0.27f) {
  constexpr float kMaxAngularRadiusDegrees = 180.0f;
  const glm::vec3 direction =
      lightDirectionFromRotationForLocalLights(rotation);
  const float sanitizedAngularRadiusDegrees =
      std::isfinite(angularRadiusDegrees) ? angularRadiusDegrees : 0.0f;
  const float clampedAngularRadiusDegrees =
      std::clamp(sanitizedAngularRadiusDegrees, 0.0f, kMaxAngularRadiusDegrees);
  return DirectionalLightGpuData{
      .directionIlluminance =
          glm::vec4(direction.x, direction.y, direction.z, intensity),
      .colorReserved =
          glm::vec4(color, glm::radians(clampedAngularRadiusDegrees)),
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

template <typename Record>
[[nodiscard]] inline LightDesc makeLocalLightDesc(const Record &record,
                                                  LightType type) {
  LightDesc out{};
  out.type = type;
  out.name = record.name;
  out.position = record.localPosition;
  out.rotation = record.localRotation;
  out.color = record.color;
  out.intensity = record.intensity;
  out.range = record.range;
  out.innerConeAngleRadians = record.innerConeAngle;
  out.outerConeAngleRadians = record.outerConeAngle;
  out.angularRadiusDegrees = record.angularRadiusDegrees;
  out.enabled = record.enabled;
  return out;
}

} // namespace nuri
