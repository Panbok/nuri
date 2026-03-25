#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>

namespace nuri {

// Per-instance GPU data uploaded to InstanceMatricesBuffer.
// Layout must match the InstanceData struct in common.sp (std430).
struct InstanceData {
  glm::mat4 modelMatrix{1.0f};
  // Columns of transpose(inverse(mat3(modelMatrix))), each padded to vec4.
  glm::vec4 normalMatCol0{1.0f, 0.0f, 0.0f, 0.0f};
  glm::vec4 normalMatCol1{0.0f, 1.0f, 0.0f, 0.0f};
  glm::vec4 normalMatCol2{0.0f, 0.0f, 1.0f, 0.0f};
};
static_assert(sizeof(InstanceData) == 112,
              "InstanceData size must match shader InstanceData layout");

inline InstanceData makeInstanceData(const glm::mat4 &model) {
  const glm::mat3 nm = glm::transpose(glm::inverse(glm::mat3(model)));
  return InstanceData{
      .modelMatrix = model,
      .normalMatCol0 = glm::vec4(nm[0], 0.0f),
      .normalMatCol1 = glm::vec4(nm[1], 0.0f),
      .normalMatCol2 = glm::vec4(nm[2], 0.0f),
  };
}

} // namespace nuri
