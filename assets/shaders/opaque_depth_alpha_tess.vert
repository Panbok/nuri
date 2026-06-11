#include "common.sp"

layout(location = 0) out vec2 outUv0;
layout(location = 1) out vec2 outUv1;
layout(location = 2) out vec3 outWorldPos;

void main() {
  const uint globalInstanceId = pc.instanceRemap.ids[gl_InstanceIndex];
  const vec3 pos = decodePackedPosition(gl_VertexIndex);
  const vec2 uv0 = decodePackedUv(gl_VertexIndex);
  const vec2 uv1 = decodePackedUv1(gl_VertexIndex);

  const InstanceData inst = pc.instanceMatrices.instances[globalInstanceId];
  const vec3 worldPos = (inst.modelMatrix * vec4(pos, 1.0)).xyz;

  outUv0 = uv0;
  outUv1 = uv1;
  outWorldPos = worldPos;
  gl_Position = vec4(worldPos, 1.0);
}
