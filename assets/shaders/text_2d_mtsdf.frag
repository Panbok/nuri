#version 460

#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_samplerless_texture_functions : require

layout(set = 0, binding = 0) uniform texture2D kTextures2D[];
layout(set = 0, binding = 1) uniform sampler kSamplers[];

layout(location = 0) in vec2 inUv;
layout(location = 1) in vec4 inColor;
layout(location = 0) out vec4 outFragColor;

layout(push_constant) uniform PushConstants {
  mat4 projection;
  uint atlasBindless;
  float pxRange;
  float _pad0;
  float _pad1;
} pc;

vec4 sampleAtlas(vec2 uv) {
  return texture(nonuniformEXT(sampler2D(kTextures2D[pc.atlasBindless], kSamplers[0])), uv);
}

float median3(vec3 v) { return max(min(v.x, v.y), min(max(v.x, v.y), v.z)); }

float screenPxRange(vec2 uv) {
  vec2 atlasSize = vec2(textureSize(sampler2D(kTextures2D[pc.atlasBindless], kSamplers[0]), 0));
  vec2 unitRange = vec2(max(pc.pxRange, 0.001)) / max(atlasSize, vec2(1.0));
  vec2 screenTexSize = vec2(1.0) / max(fwidth(uv), vec2(1.0e-6));
  return max(0.5 * dot(unitRange, screenTexSize), 1.0);
}

void main() {
  vec4 atlas = sampleAtlas(inUv);
  float pxRange = screenPxRange(inUv);
  float sdMsdf = median3(atlas.rgb) - 0.5;
  float sdSdf = atlas.a - 0.5;
  // Use SDF fallback for very small projected text to reduce color-channel artifacts.
  float sdfFallback = clamp(2.0 - pxRange, 0.0, 1.0);
  float sd = mix(sdMsdf, sdSdf, sdfFallback);
  float alpha = clamp(pxRange * sd + 0.5, 0.0, 1.0);
  outFragColor = vec4(inColor.rgb, inColor.a * alpha);
}
