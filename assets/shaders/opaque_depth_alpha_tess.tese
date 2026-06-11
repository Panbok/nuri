#include "common.sp"

layout(triangles, fractional_odd_spacing, cw) in;

layout(location = 0) in vec2 inUv0[];
layout(location = 1) in vec2 inUv1[];
layout(location = 2) in vec3 inWorldPos[];

layout(location = 0) out vec2 outUv0;
layout(location = 1) out vec2 outUv1;

void main() {
  const vec3 bary = gl_TessCoord;
  const vec3 linearPos =
      inWorldPos[0] * bary.x + inWorldPos[1] * bary.y + inWorldPos[2] * bary.z;

  outUv0 = inUv0[0] * bary.x + inUv0[1] * bary.y + inUv0[2] * bary.z;
  outUv1 = inUv1[0] * bary.x + inUv1[1] * bary.y + inUv1[2] * bary.z;
  gl_Position = pc.frameData.proj * pc.frameData.view * vec4(linearPos, 1.0);
}
