#include "common.sp"

void main() {
  const uint globalInstanceId = pc.instanceRemap.ids[gl_InstanceIndex];
  const vec3 pos = decodePackedPosition(gl_VertexIndex);
  const InstanceData inst = pc.instanceMatrices.instances[globalInstanceId];
  const mat4 lightViewProj =
      pc.frameData.shadowFrameBuffer.cascades[pc.shadowCascadeIndex]
          .lightViewProj;

  gl_Position = lightViewProj * inst.modelMatrix * vec4(pos, 1.0);
}
