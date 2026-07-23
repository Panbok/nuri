#version 460 core
#extension GL_EXT_buffer_reference : require

#define NURI_CUSTOM_PUSH_CONSTANTS 1
#include "common.sp"

struct DDGIDiagnosticRayGpuData {
  vec4 originAndDistance;
  vec4 directionAndKind;
};
layout(std430, buffer_reference) readonly buffer DDGIDiagnosticRayBuffer {
  DDGIDiagnosticRayGpuData rays[];
};
layout(push_constant) uniform PushConstants {
  mat4 viewProjection;
  DDGIDiagnosticRayBuffer diagnostic;
  uint rayCount;
  uint reserved;
}
pc;
layout(location = 0) out vec4 outColor;

void main() {
  const uint ray = uint(gl_VertexIndex) >> 1u;
  const DDGIDiagnosticRayGpuData value = pc.diagnostic.rays[ray];
  const bool endpoint = (uint(gl_VertexIndex) & 1u) != 0u;
  const vec3 position =
      value.originAndDistance.xyz +
      (endpoint ? value.directionAndKind.xyz * value.originAndDistance.w
                : vec3(0.0));
  const uint kind = floatBitsToUint(value.directionAndKind.w);
  outColor = kind == 0u   ? vec4(0.25, 0.65, 1.0, 0.9)
             : kind == 1u ? vec4(0.15, 1.0, 0.25, 0.9)
             : kind == 2u ? vec4(1.0, 0.2, 0.08, 0.9)
                          : vec4(1.0, 0.0, 1.0, 0.9);
  gl_Position = pc.viewProjection * vec4(position, 1.0);
}
