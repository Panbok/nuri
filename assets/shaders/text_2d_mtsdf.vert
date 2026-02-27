#version 460

layout(location = 0) in vec2 inPos;
layout(location = 1) in vec2 inUv;
layout(location = 2) in vec4 inColor;

layout(location = 0) out vec2 outUv;
layout(location = 1) out vec4 outColor;

layout(push_constant) uniform PushConstants {
  mat4 projection;
  uint atlasBindless;
  float pxRange;
  float _pad0;
  float _pad1;
} pc;

void main() {
  outUv = inUv;
  outColor = inColor;
  gl_Position = pc.projection * vec4(inPos, 0.0, 1.0);
}
