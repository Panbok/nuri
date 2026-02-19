#include "common.sp"

layout(vertices = 3) out;

layout(location = 0) in vec2 inUv[];
layout(location = 1) in vec3 inWorldNormal[];
layout(location = 2) in vec3 inWorldPos[];
layout(location = 3) flat in uint inInstanceId[];

layout(location = 0) out vec2 outUv[];
layout(location = 1) out vec3 outWorldNormal[];
layout(location = 2) out vec3 outWorldPos[];
layout(location = 3) flat out uint outInstanceId[];

float distanceToTessFactor(float distanceToCamera) {
  const float nearDistance = max(pc.tessNearDistance, 0.0);
  const float farDistance = max(pc.tessFarDistance, nearDistance + 1.0e-3);
  const float minFactor = clamp(pc.tessMinFactor, 1.0, 64.0);
  const float maxFactor = clamp(pc.tessMaxFactor, minFactor, 64.0);
  const float t =
      clamp((distanceToCamera - nearDistance) / (farDistance - nearDistance),
            0.0, 1.0);
  return mix(maxFactor, minFactor, t);
}

void main() {
  outUv[gl_InvocationID] = inUv[gl_InvocationID];
  outWorldNormal[gl_InvocationID] = inWorldNormal[gl_InvocationID];
  outWorldPos[gl_InvocationID] = inWorldPos[gl_InvocationID];
  outInstanceId[gl_InvocationID] = inInstanceId[gl_InvocationID];
  gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;

  if (gl_InvocationID != 0u) {
    return;
  }

  const vec3 p0 = inWorldPos[0];
  const vec3 p1 = inWorldPos[1];
  const vec3 p2 = inWorldPos[2];
  const vec3 cameraPos = pc.frameData.cameraPos.xyz;

  const float outer0 = distanceToTessFactor(length(cameraPos - 0.5 * (p1 + p2)));
  const float outer1 = distanceToTessFactor(length(cameraPos - 0.5 * (p2 + p0)));
  const float outer2 = distanceToTessFactor(length(cameraPos - 0.5 * (p0 + p1)));
  gl_TessLevelOuter[0] = outer0;
  gl_TessLevelOuter[1] = outer1;
  gl_TessLevelOuter[2] = outer2;
  gl_TessLevelInner[0] = (outer0 + outer1 + outer2) * (1.0 / 3.0);
}
