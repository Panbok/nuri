#include "common.sp"

layout(location = 0) out vec2 outUv0;
layout(location = 1) out vec2 outUv1;
layout(location = 2) out vec3 outWorldNormal;
layout(location = 3) out vec3 outWorldPos;
layout(location = 4) out vec4 outWorldTangent;
layout(location = 5) flat out uint outInstanceId;

void main() {
  const uint globalInstanceId = pc.instanceRemap.ids[gl_InstanceIndex];
  const PackedVertex packed = pc.vertexBuffer.vertices[gl_VertexIndex];

  const vec3 pos = decodePackedPosition(packed);
  const vec3 normal = decodePackedNormal(packed);
  const vec4 tangent = decodePackedTangent(packed);
  const vec2 uv0 = decodePackedUv(packed);
  const vec2 uv1 = decodePackedUv1(packed);

  const mat4 model = pc.instanceMatrices.matrices[globalInstanceId];
  const vec3 worldPos = (model * vec4(pos, 1.0)).xyz;
  const mat3 normalMatrix = transpose(inverse(mat3(model)));
  const vec3 worldNormal = normalize(normalMatrix * normal);
  const vec3 worldTangent = normalize(normalMatrix * tangent.xyz);

  outUv0 = uv0;
  outUv1 = uv1;
  outWorldNormal = worldNormal;
  outWorldPos = worldPos;
  outWorldTangent = vec4(worldTangent, tangent.w);
  outInstanceId = globalInstanceId;
  gl_Position = vec4(worldPos, 1.0);
}
