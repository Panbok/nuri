#include "common.sp"

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;

layout(location = 0) out PerVertex vtx;

void main() {
  const mat4 model = pc.perFrame.model;
  const mat4 view = pc.perFrame.view;
  const mat4 proj = pc.perFrame.proj;

  gl_Position = proj * view * model * vec4(pos, 1.0);

  const mat3 normalMatrix = transpose(inverse(mat3(model)));
  vtx.uv = uv;
  vtx.worldNormal = normalMatrix * normal;
  vtx.worldPos = (model * vec4(pos, 1.0)).xyz;
}
