#version 460

#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_samplerless_texture_functions : require

layout(set = 0, binding = 0) uniform texture2D kTextures2D[];
layout(set = 0, binding = 1) uniform sampler kSamplers[];

layout(location = 0) in vec2 inPos;
layout(location = 1) in vec2 inUv;
layout(location = 2) in vec4 inColor;

layout(location = 0) out vec2 outUv;
layout(location = 1) out vec4 outColor;
layout(location = 2) out vec2 outUnitRange;

layout(push_constant) uniform PushConstants {
  mat4 projection;
  uint atlasBindless;
  float pxRange;
  float _pad0;
  float _pad1;
}
pc;

void main() {
  outUv = inUv;
  outColor = inColor;
  // Compute pxRange / atlasSize once per vertex so the fragment shader avoids
  // a per-fragment textureSize query.
  vec2 atlasSize =
      vec2(textureSize(kTextures2D[nonuniformEXT(pc.atlasBindless)], 0));
  outUnitRange = vec2(max(pc.pxRange, 0.001)) / max(atlasSize, vec2(1.0));
  gl_Position = pc.projection * vec4(inPos, 0.0, 1.0);
}
