#define NURI_SHADOW_DEPTH 1
#include "common.sp"

void main() {
  const uint globalInstanceId = pc.instanceRemap.ids[gl_InstanceIndex];
  ShadowDrawGpuData drawData = pc.shadowDraws.values[gl_DrawID];
  const vec3 pos = decodePackedPositionFrom(
      drawData.vertexBuffer, drawData.vertexDecodeBuffer, drawData.metadata.x,
      drawData.metadata.y, gl_VertexIndex);
  const InstanceData inst = pc.instanceMatrices.instances[globalInstanceId];
  const mat4 lightViewProj =
      pc.frameData.shadowFrameBuffer.cascades[pc.shadowCascadeIndex]
          .lightViewProj;

  gl_Position = lightViewProj * inst.modelMatrix * vec4(pos, 1.0);
}
