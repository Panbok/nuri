#pragma once
#include "nuri/defines.h"
#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>
namespace nuri {

struct InstanceData {
  glm::mat4 modelMatrix{1.0f};
  glm::vec4 normalMatCol0{1.0f, 0.0f, 0.0f, 0.0f};
  glm::vec4 normalMatCol1{0.0f, 1.0f, 0.0f, 0.0f};
  glm::vec4 normalMatCol2{0.0f, 0.0f, 1.0f, 0.0f};
};
static_assert(sizeof(InstanceData) == 112);

inline InstanceData makeInstanceData(const glm::mat4 &model) {
  const glm::mat3 nm = glm::transpose(glm::inverse(glm::mat3(model)));
  return InstanceData{
      .modelMatrix = model,
      .normalMatCol0 = glm::vec4(nm[0], 0.0f),
      .normalMatCol1 = glm::vec4(nm[1], 0.0f),
      .normalMatCol2 = glm::vec4(nm[2], 0.0f),
  };
}

struct ForwardMeshPushConstants {
  uint64_t frameDataAddress = 0;
  uint64_t vertexBufferAddress = 0;
  uint64_t vertexDecodeBufferAddress = 0;
  uint64_t instanceMatricesAddress = 0;
  uint64_t previousInstanceMatricesAddress = 0;
  uint64_t instanceRemapAddress = 0;
  uint64_t instanceCentersPhaseAddress = 0;
  uint64_t instanceBaseMatricesAddress = 0;
  uint64_t velocityInstanceFlagsAddress = 0;
  uint64_t velocityFrameDataAddress = 0;
  uint32_t instanceCount = 0;
  uint32_t materialIndex = 0;
  uint32_t vertexDecodeIndex = 0;
  uint32_t packedVertexFormat = 0;
  float timeSeconds = 0.0f;
  float tessNearDistance = 1.0f;
  float tessFarDistance = 8.0f;
  float tessMinFactor = 1.0f;
  float tessMaxFactor = 6.0f;
  uint32_t debugVisualizationMode = 0;
  uint32_t shadowCascadeIndex = 0;
};
static_assert(sizeof(ForwardMeshPushConstants) == 128);
static_assert(offsetof(ForwardMeshPushConstants, instanceRemapAddress) == 40u);
static_assert(offsetof(ForwardMeshPushConstants, instanceCount) == 80u);
static_assert(offsetof(ForwardMeshPushConstants, shadowCascadeIndex) == 120u);

} // namespace nuri
