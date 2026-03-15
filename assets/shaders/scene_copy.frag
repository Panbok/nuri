#extension GL_EXT_buffer_reference : require

layout(location = 0) in vec2 outUv;
layout(location = 0) out vec4 out_FragColor;

const uint kCopyFlagDownsample = 1u << 0u;

layout(push_constant) uniform CopyPushConstants {
  uint sourceTexId;
  uint sourceSamplerId;
  uint flags;
  uint reserved0;
} pc;

vec4 sampleSource(vec2 uv) {
  return textureBindless2D(pc.sourceTexId, pc.sourceSamplerId, uv);
}

vec4 sampleDownsampled(vec2 uv) {
  ivec2 sourceSize = textureBindlessSize2D(pc.sourceTexId);
  vec2 texel = 1.0 / max(vec2(sourceSize), vec2(1.0));

  vec4 s00 = sampleSource(uv + texel * vec2(-0.5, -0.5));
  vec4 s10 = sampleSource(uv + texel * vec2(0.5, -0.5));
  vec4 s01 = sampleSource(uv + texel * vec2(-0.5, 0.5));
  vec4 s11 = sampleSource(uv + texel * vec2(0.5, 0.5));
  return 0.25 * (s00 + s10 + s01 + s11);
}

void main() {
  if ((pc.flags & kCopyFlagDownsample) != 0u) {
    out_FragColor = sampleDownsampled(outUv);
    return;
  }

  out_FragColor = sampleSource(outUv);
}
