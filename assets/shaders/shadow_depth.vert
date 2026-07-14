#define NURI_SHADOW_DEPTH 1
#include "common.sp"

layout(location = 0) out vec2 outUv0;
layout(location = 1) out vec2 outUv1;

void main() {
  const uint globalInstanceId = pc.instanceRemap.ids[gl_InstanceIndex];
  ShadowDrawGpuData drawData = pc.shadowDraws.values[gl_DrawID];
  const vec3 pos = decodePackedPositionFrom(
      drawData.vertexBuffer, drawData.vertexDecodeBuffer, drawData.metadata.x,
      drawData.metadata.y, gl_VertexIndex);
  const vec2 uv0 = decodePackedUvFrom(drawData.vertexBuffer,
                                      drawData.metadata.y, gl_VertexIndex);
  const vec2 uv1 = decodePackedUv1From(drawData.vertexBuffer,
                                       drawData.metadata.y, gl_VertexIndex);

  const InstanceData inst = pc.instanceMatrices.instances[globalInstanceId];
  const mat4 model = inst.modelMatrix;
  const mat4 lightViewProj =
      pc.frameData.shadowFrameBuffer.cascades[pc.shadowCascadeIndex]
          .lightViewProj;

  const vec4 worldPos4 = model * vec4(pos, 1.0);
  gl_Position = lightViewProj * worldPos4;
  outUv0 = uv0;
  outUv1 = uv1;
}
