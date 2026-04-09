#include "common.sp"

layout(location = 0) out vec2 outUv0;
layout(location = 1) out vec2 outUv1;
layout(location = 2) out vec3 outWorldNormal;
layout(location = 3) out vec3 outWorldPos;
layout(location = 4) out vec4 outWorldTangent;
layout(location = 5) flat out uint outInstanceId;

void main() {
  const uint globalInstanceId = pc.instanceRemap.ids[gl_InstanceIndex];
  const vec3 pos = decodePackedPosition(gl_VertexIndex);
  const vec3 normal = decodePackedNormal(gl_VertexIndex);
  const vec4 tangent = decodePackedTangent(gl_VertexIndex);
  const vec2 uv0 = decodePackedUv(gl_VertexIndex);
  const vec2 uv1 = decodePackedUv1(gl_VertexIndex);

  const InstanceData inst = pc.instanceMatrices.instances[globalInstanceId];
  const mat4 model = inst.modelMatrix;
  const vec3 worldPos = (model * vec4(pos, 1.0)).xyz;
  const mat3 normalMatrix = mat3(inst.normalMatCol0.xyz, inst.normalMatCol1.xyz,
                                 inst.normalMatCol2.xyz);
  const vec3 worldNormal = normalize(normalMatrix * normal);
  vec3 transformedTangent = mat3(model) * tangent.xyz;
  transformedTangent -= worldNormal * dot(transformedTangent, worldNormal);
  const vec4 worldTangent =
      dot(transformedTangent, transformedTangent) > 1.0e-10
          ? vec4(normalize(transformedTangent), tangent.w)
          : vec4(0.0, 0.0, 0.0, 1.0);

  outUv0 = uv0;
  outUv1 = uv1;
  outWorldNormal = worldNormal;
  outWorldPos = worldPos;
  outWorldTangent = worldTangent;
  outInstanceId = globalInstanceId;
  gl_Position = vec4(worldPos, 1.0);
}
