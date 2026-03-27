#pragma once

#include "nuri/defines.h"

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

// `model` is expected to have an invertible upper-left 3x3. Singular
// transforms fall back to identity normal columns to avoid propagating NaN/Inf
// into InstanceData.normalMatCol0/1/2.
inline InstanceData makeInstanceData(const glm::mat4 &model) {
  glm::mat3 nm(1.0f);
  const float det = glm::determinant(glm::mat3(model));
  if (glm::abs(det) > 1.0e-8f) {
    nm = glm::transpose(glm::inverse(glm::mat3(model)));
  } else {
    NURI_ASSERT(false, "makeInstanceData requires an invertible model matrix");
  }
  return InstanceData{
      .modelMatrix = model,
      .normalMatCol0 = glm::vec4(nm[0], 0.0f),
      .normalMatCol1 = glm::vec4(nm[1], 0.0f),
      .normalMatCol2 = glm::vec4(nm[2], 0.0f),
  };
}

} // namespace nuri
