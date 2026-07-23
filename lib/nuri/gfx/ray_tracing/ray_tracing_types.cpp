#include "nuri/gfx/ray_tracing/ray_tracing_types.h"

namespace nuri {

AccelerationStructureTransform
makeAccelerationStructureTransform(const glm::mat4 &matrix) noexcept {
  return AccelerationStructureTransform{.rowMajor3x4 = {
                                            matrix[0][0],
                                            matrix[1][0],
                                            matrix[2][0],
                                            matrix[3][0],
                                            matrix[0][1],
                                            matrix[1][1],
                                            matrix[2][1],
                                            matrix[3][1],
                                            matrix[0][2],
                                            matrix[1][2],
                                            matrix[2][2],
                                            matrix[3][2],
                                        }};
}

} // namespace nuri
