#include "common.sp"

layout(triangles, fractional_odd_spacing, cw) in;

layout(location = 0) in vec3 inWorldPos[];

void main() {
  const vec3 bary = gl_TessCoord;
  const vec3 linearPos =
      inWorldPos[0] * bary.x + inWorldPos[1] * bary.y + inWorldPos[2] * bary.z;
  gl_Position = pc.frameData.proj * pc.frameData.view * vec4(linearPos, 1.0);
}
