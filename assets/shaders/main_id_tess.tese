#include "common.sp"

layout(triangles, fractional_odd_spacing, cw) in;

layout(location = 0) in vec3 inWorldPos[];
layout(location = 5) flat in uint inInstanceId[];

layout(location = 10) flat out uint outInstanceId;

void main() {
  const vec3 bary = gl_TessCoord;

  const vec3 p0 = inWorldPos[0];
  const vec3 p1 = inWorldPos[1];
  const vec3 p2 = inWorldPos[2];

  const vec3 linearPos = p0 * bary.x + p1 * bary.y + p2 * bary.z;

  outInstanceId = inInstanceId[0];
  gl_Position = pc.frameData.proj * pc.frameData.view * vec4(linearPos, 1.0);
}
