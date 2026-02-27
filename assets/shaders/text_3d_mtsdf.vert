#version 460

#extension GL_EXT_buffer_reference : require

layout(location = 0) out vec2 outUv;
layout(location = 1) out vec4 outColor;

struct GlyphInstance {
  vec4 rectMinMax;
  vec4 uvMinMax;
  uint color;
  uint transformIndex;
  uint _pad0;
  uint _pad1;
};

struct ResolvedTransform {
  vec4 basisX;
  vec4 basisY;
  vec4 translation;
};

layout(std430, buffer_reference) readonly buffer GlyphInstanceBuffer {
  GlyphInstance instances[];
};

layout(std430, buffer_reference) readonly buffer TransformBuffer {
  ResolvedTransform transforms[];
};

layout(push_constant) uniform PushConstants {
  mat4 viewProjection;
  GlyphInstanceBuffer glyphBuffer;
  TransformBuffer transformBuffer;
  uint atlasBindless;
  float pxRange;
  float _pad0;
  float _pad1;
} pc;

vec4 unpackColor(uint packed) {
  const float r = float((packed >> 0u) & 0xffu) / 255.0;
  const float g = float((packed >> 8u) & 0xffu) / 255.0;
  const float b = float((packed >> 16u) & 0xffu) / 255.0;
  const float a = float((packed >> 24u) & 0xffu) / 255.0;
  return vec4(r, g, b, a);
}

void main() {
  const GlyphInstance glyph = pc.glyphBuffer.instances[gl_InstanceIndex];
  const ResolvedTransform transform =
      pc.transformBuffer.transforms[glyph.transformIndex];

  uint corner = uint(gl_VertexIndex);
  if (corner == 3u) {
    corner = 2u;
  } else if (corner == 4u) {
    corner = 3u;
  } else if (corner == 5u) {
    corner = 0u;
  }

  vec2 local = vec2(0.0);
  vec2 uv = vec2(0.0);
  if (corner == 0u) {
    local = vec2(glyph.rectMinMax.x, glyph.rectMinMax.y);
    uv = vec2(glyph.uvMinMax.x, glyph.uvMinMax.y);
  } else if (corner == 1u) {
    local = vec2(glyph.rectMinMax.z, glyph.rectMinMax.y);
    uv = vec2(glyph.uvMinMax.z, glyph.uvMinMax.y);
  } else if (corner == 2u) {
    local = vec2(glyph.rectMinMax.z, glyph.rectMinMax.w);
    uv = vec2(glyph.uvMinMax.z, glyph.uvMinMax.w);
  } else {
    local = vec2(glyph.rectMinMax.x, glyph.rectMinMax.w);
    uv = vec2(glyph.uvMinMax.x, glyph.uvMinMax.w);
  }

  const vec3 worldPos = transform.translation.xyz +
                        transform.basisX.xyz * local.x +
                        transform.basisY.xyz * local.y;
  gl_Position = pc.viewProjection * vec4(worldPos, 1.0);
  outUv = uv;
  outColor = unpackColor(glyph.color);
}
