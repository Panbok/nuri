#extension GL_EXT_buffer_reference : require

layout(location = 0) in vec2 uv;
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

// Karis-average weighted box filter: weights each sample by 1/(1+luma) so
// that high-luminance fireflies cannot dominate the downsampled result.
// This is appropriate for HDR scene-color mip chains used by transmission.
float karisLuminance(vec3 c) {
  return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

vec4 sampleDownsampled(vec2 uv) {
  ivec2 sourceSize = textureBindlessSize2D(pc.sourceTexId);
  vec2 texel = 1.0 / max(vec2(sourceSize), vec2(1.0));

  vec4 s00 = sampleSource(uv + texel * vec2(-0.5, -0.5));
  vec4 s10 = sampleSource(uv + texel * vec2( 0.5, -0.5));
  vec4 s01 = sampleSource(uv + texel * vec2(-0.5,  0.5));
  vec4 s11 = sampleSource(uv + texel * vec2( 0.5,  0.5));

  float w00 = 1.0 / (1.0 + karisLuminance(s00.rgb));
  float w10 = 1.0 / (1.0 + karisLuminance(s10.rgb));
  float w01 = 1.0 / (1.0 + karisLuminance(s01.rgb));
  float w11 = 1.0 / (1.0 + karisLuminance(s11.rgb));
  float totalWeight = w00 + w10 + w01 + w11;

  return (s00 * w00 + s10 * w10 + s01 * w01 + s11 * w11) / totalWeight;
}

void main() {
  if ((pc.flags & kCopyFlagDownsample) != 0u) {
    out_FragColor = sampleDownsampled(uv);
    return;
  }

  out_FragColor = sampleSource(uv);
}
