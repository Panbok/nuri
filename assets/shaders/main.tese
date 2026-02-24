#include "common.sp"

layout(triangles, fractional_odd_spacing, cw) in;

layout(location = 0) in vec2 inUv[];
layout(location = 1) in vec3 inWorldNormal[];
layout(location = 2) in vec3 inWorldPos[];
layout(location = 3) flat in uint inInstanceId[];
layout(location = 4) patch in vec3 inPatchOuterFactors;
layout(location = 5) patch in float inPatchInnerFactor;

layout(location = 0) out PerVertex vtx;
layout(location = 9) flat out uint outInstanceId;

void main() {
  const vec3 bary = gl_TessCoord;

  const vec3 p0 = inWorldPos[0];
  const vec3 p1 = inWorldPos[1];
  const vec3 p2 = inWorldPos[2];

  const vec3 linearPos = p0 * bary.x + p1 * bary.y + p2 * bary.z;
  const vec3 linearNormal = normalize(inWorldNormal[0] * bary.x +
                                      inWorldNormal[1] * bary.y +
                                      inWorldNormal[2] * bary.z);

  const vec2 uv = inUv[0] * bary.x + inUv[1] * bary.y + inUv[2] * bary.z;

  vtx.uv = uv;
  vtx.worldNormal = linearNormal;
  vtx.worldPos = linearPos;
  vtx.patchBarycentric = bary;
  vtx.triBarycentric = vec3(0.0);
  vtx.patchOuterFactors = inPatchOuterFactors;
  vtx.patchInnerFactor = inPatchInnerFactor;
  vtx.tessellatedFlag = 1.0;
  outInstanceId = inInstanceId[0];
  gl_Position = pc.frameData.proj * pc.frameData.view * vec4(linearPos, 1.0);
}
