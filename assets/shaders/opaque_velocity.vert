#include "common.sp"

layout(location = 0) out PerVertex vtx;
layout(location = 10) out vec4 outCurrentClipNoJitter;
layout(location = 11) out vec4 outPreviousClipNoJitter;
layout(location = 12) flat out uint outVelocityFlags;

void main() {
  const uint globalInstanceId = pc.instanceRemap.ids[gl_InstanceIndex];
  const vec3 pos = decodePackedPosition(gl_VertexIndex);
  const vec3 normal = decodePackedNormal(gl_VertexIndex);
  const vec4 tangent = decodePackedTangent(gl_VertexIndex);
  const vec2 uv0 = decodePackedUv(gl_VertexIndex);
  const vec2 uv1 = decodePackedUv1(gl_VertexIndex);

  const InstanceData inst = pc.instanceMatrices.instances[globalInstanceId];
  const InstanceData previousInst =
      pc.previousInstanceMatrices.instances[globalInstanceId];
  const mat4 model = inst.modelMatrix;

  const vec4 localPos = vec4(pos, 1.0);
  const vec4 worldPos4 = model * localPos;
  const vec4 currentClipNoJitter =
      pc.velocityFrameData.data.currentViewProjNoJitter * worldPos4;

  const uint velocityFlags = pc.velocityInstanceFlags.flags[globalInstanceId];
  const vec4 previousClipNoJitter =
      velocityFlags != 0u ? pc.velocityFrameData.data.previousViewProjNoJitter *
                                previousInst.modelMatrix * localPos
                          : currentClipNoJitter;

  gl_Position = pc.frameData.proj * pc.frameData.view * worldPos4;

  const mat3 normalMatrix = mat3(inst.normalMatCol0.xyz, inst.normalMatCol1.xyz,
                                 inst.normalMatCol2.xyz);
  vtx.uv0 = uv0;
  vtx.uv1 = uv1;
  const vec3 worldNormal = normalize(normalMatrix * normal);
  vtx.worldNormal = worldNormal;
  vec3 transformedTangent = mat3(model) * tangent.xyz;
  transformedTangent -= worldNormal * dot(transformedTangent, worldNormal);
  vtx.worldTangent = dot(transformedTangent, transformedTangent) > 1.0e-10
                         ? vec4(normalize(transformedTangent), tangent.w)
                         : vec4(0.0, 0.0, 0.0, 1.0);
  vtx.worldPos = worldPos4.xyz;
  vtx.patchBarycentric = vec3(0.0);
  vtx.triBarycentric = vec3(0.0);
  vtx.patchOuterFactors = vec3(1.0);
  vtx.patchInnerFactor = 1.0;
  vtx.tessellatedFlag = 0.0;
  outCurrentClipNoJitter = currentClipNoJitter;
  outPreviousClipNoJitter = previousClipNoJitter;
  outVelocityFlags = velocityFlags;
}
