#include "common.sp"

layout(location = 0) out vec2 outUv;
layout(location = 1) out vec3 outWorldNormal;
layout(location = 2) out vec3 outWorldPos;
layout(location = 3) flat out uint outInstanceId;

void main() {
  const uint globalInstanceId = pc.instanceRemap.ids[gl_InstanceIndex];
  const PackedVertex packed = pc.vertexBuffer.vertices[gl_VertexIndex];

  const vec3 pos = decodePackedPosition(packed);
  const vec3 normal = decodePackedNormal(packed);
  const vec2 uv = decodePackedUv(packed);

  const mat4 model = pc.instanceMatrices.matrices[globalInstanceId];
  const vec3 worldPos = (model * vec4(pos, 1.0)).xyz;
  const vec3 worldNormal = normalize(mat3(model) * normal);

  outUv = uv;
  outWorldNormal = worldNormal;
  outWorldPos = worldPos;
  outInstanceId = globalInstanceId;
  gl_Position = vec4(worldPos, 1.0);
}
