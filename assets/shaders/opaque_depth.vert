#include "common.sp"

void main() {
  const uint globalInstanceId = pc.instanceRemap.ids[gl_InstanceIndex];
  const vec3 pos = decodePackedPosition(gl_VertexIndex);

  const InstanceData inst = pc.instanceMatrices.instances[globalInstanceId];
  const vec4 worldPos4 = inst.modelMatrix * vec4(pos, 1.0);
  gl_Position = pc.frameData.proj * pc.frameData.view * worldPos4;
}
