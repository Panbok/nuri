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

vec3 evaluatePnPosition(vec3 p0, vec3 p1, vec3 p2, vec3 n0, vec3 n1, vec3 n2,
                        vec3 bary) {
  const vec3 b300 = p0;
  const vec3 b030 = p1;
  const vec3 b003 = p2;

  const vec3 b210 = (2.0 * p0 + p1 - dot(p1 - p0, n0) * n0) / 3.0;
  const vec3 b120 = (2.0 * p1 + p0 - dot(p0 - p1, n1) * n1) / 3.0;
  const vec3 b021 = (2.0 * p1 + p2 - dot(p2 - p1, n1) * n1) / 3.0;
  const vec3 b012 = (2.0 * p2 + p1 - dot(p1 - p2, n2) * n2) / 3.0;
  const vec3 b102 = (2.0 * p2 + p0 - dot(p0 - p2, n2) * n2) / 3.0;
  const vec3 b201 = (2.0 * p0 + p2 - dot(p2 - p0, n0) * n0) / 3.0;

  const vec3 edgeCenter =
      (b210 + b120 + b021 + b012 + b102 + b201) * (1.0 / 6.0);
  const vec3 triangleCenter = (p0 + p1 + p2) * (1.0 / 3.0);
  const vec3 b111 = edgeCenter + (edgeCenter - triangleCenter) * 0.5;

  const float u = bary.x;
  const float v = bary.y;
  const float w = bary.z;

  const float uu = u * u;
  const float vv = v * v;
  const float ww = w * w;

  return b300 * (uu * u) + b030 * (vv * v) + b003 * (ww * w) +
         b210 * (3.0 * uu * v) + b120 * (3.0 * u * vv) + b201 * (3.0 * uu * w) +
         b102 * (3.0 * u * ww) + b021 * (3.0 * vv * w) + b012 * (3.0 * v * ww) +
         b111 * (6.0 * u * v * w);
}

void main() {
  const vec3 bary = gl_TessCoord;

  const vec3 p0 = inWorldPos[0];
  const vec3 p1 = inWorldPos[1];
  const vec3 p2 = inWorldPos[2];

  const bool hasDegenerateNormals =
      dot(inWorldNormal[0], inWorldNormal[0]) < 1.0e-6 ||
      dot(inWorldNormal[1], inWorldNormal[1]) < 1.0e-6 ||
      dot(inWorldNormal[2], inWorldNormal[2]) < 1.0e-6;

  const vec3 fallbackNormal = vec3(0.0, 1.0, 0.0);
  const vec3 n0 =
      hasDegenerateNormals ? fallbackNormal : normalize(inWorldNormal[0]);
  const vec3 n1 =
      hasDegenerateNormals ? fallbackNormal : normalize(inWorldNormal[1]);
  const vec3 n2 =
      hasDegenerateNormals ? fallbackNormal : normalize(inWorldNormal[2]);

  const vec3 linearPos = p0 * bary.x + p1 * bary.y + p2 * bary.z;
  const vec3 linearNormal = normalize(n0 * bary.x + n1 * bary.y + n2 * bary.z);

  const float area2 = length(cross(p1 - p0, p2 - p0));
  const bool hasDegenerateTriangle = area2 < 1.0e-6;
  vec3 worldPos = linearPos;
  if (!hasDegenerateTriangle && !hasDegenerateNormals) {
    worldPos = evaluatePnPosition(p0, p1, p2, n0, n1, n2, bary);
  }

  const vec2 uv = inUv[0] * bary.x + inUv[1] * bary.y + inUv[2] * bary.z;
  const vec3 worldNormal = linearNormal;

  vtx.uv = uv;
  vtx.worldNormal = worldNormal;
  vtx.worldPos = worldPos;
  vtx.patchBarycentric = bary;
  vtx.triBarycentric = vec3(0.0);
  vtx.patchOuterFactors = inPatchOuterFactors;
  vtx.patchInnerFactor = inPatchInnerFactor;
  vtx.tessellatedFlag = 1.0;
  outInstanceId = inInstanceId[0];
  gl_Position = pc.frameData.proj * pc.frameData.view * vec4(worldPos, 1.0);
}
