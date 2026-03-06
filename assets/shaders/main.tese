#include "common.sp"

layout(triangles, fractional_odd_spacing, cw) in;

layout(location = 0) in vec2 inUv0[];
layout(location = 1) in vec2 inUv1[];
layout(location = 2) in vec3 inWorldNormal[];
layout(location = 3) in vec3 inWorldPos[];
layout(location = 4) in vec4 inWorldTangent[];
layout(location = 5) flat in uint inInstanceId[];
layout(location = 6) patch in vec3 inPatchOuterFactors;
layout(location = 7) patch in float inPatchInnerFactor;

layout(location = 0) out PerVertex vtx;
layout(location = 10) flat out uint outInstanceId;

void main() {
  const vec3 bary = gl_TessCoord;

  const vec3 p0 = inWorldPos[0];
  const vec3 p1 = inWorldPos[1];
  const vec3 p2 = inWorldPos[2];

  const vec3 linearPos = p0 * bary.x + p1 * bary.y + p2 * bary.z;
  const vec3 weightedNormal = inWorldNormal[0] * bary.x +
                              inWorldNormal[1] * bary.y +
                              inWorldNormal[2] * bary.z;
  const float lenSq = dot(weightedNormal, weightedNormal);
  const float eps = 1e-6;
  const vec3 linearNormal =
      (lenSq > eps) ? normalize(weightedNormal) : inWorldNormal[0];

  const vec2 uv0 = inUv0[0] * bary.x + inUv0[1] * bary.y + inUv0[2] * bary.z;
  const vec2 uv1 = inUv1[0] * bary.x + inUv1[1] * bary.y + inUv1[2] * bary.z;

  vtx.uv0 = uv0;
  vtx.uv1 = uv1;
  vtx.worldNormal = linearNormal;
  vtx.worldTangent = vec4(0.0, 0.0, 0.0, 1.0);
  vtx.worldPos = linearPos;
  vtx.patchBarycentric = bary;
  vtx.triBarycentric = vec3(0.0);
  vtx.patchOuterFactors = inPatchOuterFactors;
  vtx.patchInnerFactor = inPatchInnerFactor;
  vtx.tessellatedFlag = 1.0;
  outInstanceId = inInstanceId[0];
  gl_Position = pc.frameData.proj * pc.frameData.view * vec4(linearPos, 1.0);
}
