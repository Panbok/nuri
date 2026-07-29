#version 460

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_samplerless_texture_functions : require

layout(set = 0, binding = 0) uniform texture2D kTextures2D[];
layout(set = 0, binding = 1) uniform sampler kSamplers[];

layout(location = 0) out vec2 outUv;
layout(location = 1) out vec4 outColor;
layout(location = 2) out vec2 outUnitRange;

struct GlyphInstance {
  vec4 rectMinMax;
  vec4 uvMinMax;
  uint color;
  uint transformIndex;
  uint _pad0;
  uint _pad1;
};

layout(std430, buffer_reference) readonly buffer GlyphInstanceBuffer {
  GlyphInstance instances[];
};

layout(push_constant) uniform PushConstants {
  mat4 projection;
  GlyphInstanceBuffer glyphBuffer;
  uint atlasBindless;
  float pxRange;
}
pc;

vec4 unpackColor(uint packed) {
  const float r = float((packed >> 0u) & 0xffu) / 255.0;
  const float g = float((packed >> 8u) & 0xffu) / 255.0;
  const float b = float((packed >> 16u) & 0xffu) / 255.0;
  const float a = float((packed >> 24u) & 0xffu) / 255.0;
  return vec4(r, g, b, a);
}

void main() {
  const GlyphInstance glyph = pc.glyphBuffer.instances[gl_InstanceIndex];
  const uint cornerLUT[6] = uint[6](0u, 1u, 2u, 2u, 3u, 0u);
  const uint corner = cornerLUT[gl_VertexIndex];
  vec2 pos;
  if (corner == 0u) {
    pos = glyph.rectMinMax.xy;
    outUv = glyph.uvMinMax.xy;
  } else if (corner == 1u) {
    pos = glyph.rectMinMax.zy;
    outUv = glyph.uvMinMax.zy;
  } else if (corner == 2u) {
    pos = glyph.rectMinMax.zw;
    outUv = glyph.uvMinMax.zw;
  } else {
    pos = glyph.rectMinMax.xw;
    outUv = glyph.uvMinMax.xw;
  }
  outColor = unpackColor(glyph.color);
  // Compute pxRange / atlasSize once per vertex so the fragment shader avoids
  // a per-fragment textureSize query.
  vec2 atlasSize =
      vec2(textureSize(kTextures2D[nonuniformEXT(pc.atlasBindless)], 0));
  outUnitRange = vec2(max(pc.pxRange, 0.001)) / max(atlasSize, vec2(1.0));
  gl_Position = pc.projection * vec4(pos, 0.0, 1.0);
}
