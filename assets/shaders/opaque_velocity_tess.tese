#include "common.sp"

layout(triangles, fractional_odd_spacing, cw) in;

layout(location = 0) in vec2 inUv0[];
layout(location = 1) in vec2 inUv1[];
layout(location = 2) in vec3 inWorldNormal[];
layout(location = 3) in vec3 inWorldPos[];
layout(location = 4) in vec4 inWorldTangent[];
layout(location = 6) patch in vec3 inPatchOuterFactors;
layout(location = 7) patch in float inPatchInnerFactor;
layout(location = 10) in vec3 inCurrentWorldPos[];
layout(location = 11) in vec3 inPreviousWorldPos[];
layout(location = 12) in uint inVelocityFlags[];

layout(location = 0) out PerVertex vtx;
layout(location = 10) out vec4 outCurrentClipNoJitter;
layout(location = 11) out vec4 outPreviousClipNoJitter;
layout(location = 12) flat out uint outVelocityFlags;

void main() {
  const vec3 bary = gl_TessCoord;

  const vec3 currentWorldPos = inCurrentWorldPos[0] * bary.x +
                               inCurrentWorldPos[1] * bary.y +
                               inCurrentWorldPos[2] * bary.z;
  const vec3 previousWorldPos = inPreviousWorldPos[0] * bary.x +
                                inPreviousWorldPos[1] * bary.y +
                                inPreviousWorldPos[2] * bary.z;
  const vec3 weightedNormal = inWorldNormal[0] * bary.x +
                              inWorldNormal[1] * bary.y +
                              inWorldNormal[2] * bary.z;
  const float lenSq = dot(weightedNormal, weightedNormal);
  const float eps = 1e-6;
  const vec3 linearNormal =
      (lenSq > eps) ? normalize(weightedNormal) : inWorldNormal[0];
  vec3 weightedTangent = inWorldTangent[0].xyz * bary.x +
                         inWorldTangent[1].xyz * bary.y +
                         inWorldTangent[2].xyz * bary.z;
  weightedTangent -= linearNormal * dot(weightedTangent, linearNormal);
  const float tangentLenSq = dot(weightedTangent, weightedTangent);
  const vec3 tangentHelper =
      abs(linearNormal.x) < 0.999 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
  const vec4 linearTangent =
      tangentLenSq > eps ? vec4(weightedTangent * inversesqrt(tangentLenSq),
                                inWorldTangent[0].w)
                         : vec4(normalize(cross(tangentHelper, linearNormal)),
                                inWorldTangent[0].w);

  const vec2 uv0 = inUv0[0] * bary.x + inUv0[1] * bary.y + inUv0[2] * bary.z;
  const vec2 uv1 = inUv1[0] * bary.x + inUv1[1] * bary.y + inUv1[2] * bary.z;

  vtx.uv0 = uv0;
  vtx.uv1 = uv1;
  vtx.worldNormal = linearNormal;
  vtx.worldTangent = linearTangent;
  vtx.worldPos = currentWorldPos;
  vtx.patchBarycentric = bary;
  vtx.triBarycentric = vec3(0.0);
  vtx.patchOuterFactors = inPatchOuterFactors;
  vtx.patchInnerFactor = inPatchInnerFactor;
  vtx.tessellatedFlag = 1.0;

  outCurrentClipNoJitter =
      pc.velocityFrameData.data.currentViewProjNoJitter *
      vec4(currentWorldPos, 1.0);
  outPreviousClipNoJitter =
      pc.velocityFrameData.data.previousViewProjNoJitter *
      vec4(previousWorldPos, 1.0);
  outVelocityFlags = inVelocityFlags[0];
  gl_Position =
      pc.frameData.proj * pc.frameData.view * vec4(currentWorldPos, 1.0);
}
