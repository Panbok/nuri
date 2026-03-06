#include "common.sp"

layout(location = 0) out PerVertex vtx;
layout(location = 10) flat out uint outInstanceId;

void main() {
  const uint globalInstanceId = pc.instanceRemap.ids[gl_InstanceIndex];
  const PackedVertex packed = pc.vertexBuffer.vertices[gl_VertexIndex];
  const vec3 pos = decodePackedPosition(packed);
  const vec3 normal = decodePackedNormal(packed);
  const vec2 uv0 = decodePackedUv(packed);
  const vec2 uv1 = decodePackedUv1(packed);

  const mat4 model = pc.instanceMatrices.matrices[globalInstanceId];
  const mat4 view = pc.frameData.view;
  const mat4 proj = pc.frameData.proj;

  const vec4 worldPos4 = model * vec4(pos, 1.0);
  gl_Position = proj * view * worldPos4;

  const mat3 normalMatrix = transpose(inverse(mat3(model)));
  vtx.uv0 = uv0;
  vtx.uv1 = uv1;
  vtx.worldNormal = normalize(normalMatrix * normal);
  vtx.worldTangent = vec4(0.0, 0.0, 0.0, 1.0);
  vtx.worldPos = worldPos4.xyz;
  vtx.patchBarycentric = vec3(0.0);
  vtx.triBarycentric = vec3(0.0);
  vtx.patchOuterFactors = vec3(1.0);
  vtx.patchInnerFactor = 1.0;
  vtx.tessellatedFlag = 0.0;
  outInstanceId = globalInstanceId;
}
