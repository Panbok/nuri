#include "common.sp"

layout(location = 0) out vec3 outWorldPos;
layout(location = 5) flat out uint outInstanceId;

void main() {
  const uint globalInstanceId = pc.instanceRemap.ids[gl_InstanceIndex];
  const vec3 pos = decodePackedPosition(gl_VertexIndex);

  const InstanceData inst = pc.instanceMatrices.instances[globalInstanceId];
  const vec3 worldPos = (inst.modelMatrix * vec4(pos, 1.0)).xyz;

  outWorldPos = worldPos;
  outInstanceId = globalInstanceId;
  gl_Position = vec4(worldPos, 1.0);
}
