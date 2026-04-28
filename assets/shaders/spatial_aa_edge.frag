#extension GL_EXT_buffer_reference : require

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 out_FragColor;

layout(push_constant) uniform SpatialAAPushConstants {
  uint sourceTexId;
  uint edgeTexId;
  uint blendTexId;
  uint areaTexId;
  uint searchTexId;
  uint linearSamplerId;
  uint pointSamplerId;
  uint mode;
  uint inverseWidthBits;
  uint inverseHeightBits;
  uint edgeThresholdBits;
  uint maxSearchSteps;
  uint resolveStrengthBits;
}
pc;

float pushFloat(uint bits) { return uintBitsToFloat(bits); }

vec2 screenUv(vec2 fullscreenUv) { return fract(fullscreenUv); }

vec4 sampleSource(vec2 sampleUv) {
  return textureBindless2D(pc.sourceTexId, pc.pointSamplerId, sampleUv);
}

float compressedLuma(vec3 color) {
  float luma = dot(max(color, vec3(0.0)), vec3(0.2126, 0.7152, 0.0722));
  return log2(1.0 + luma);
}

void main() {
  vec2 centerUv = screenUv(uv);
  vec2 texel = vec2(max(pushFloat(pc.inverseWidthBits), 1.0e-8),
                    max(pushFloat(pc.inverseHeightBits), 1.0e-8));
  vec2 minUv = texel * 0.5;
  vec2 maxUv = vec2(1.0) - minUv;
  float threshold = max(pushFloat(pc.edgeThresholdBits), 0.0);

  float center = compressedLuma(sampleSource(centerUv).rgb);
  float left = compressedLuma(
      sampleSource(clamp(centerUv - vec2(texel.x, 0.0), minUv, maxUv)).rgb);
  float top = compressedLuma(
      sampleSource(clamp(centerUv - vec2(0.0, texel.y), minUv, maxUv)).rgb);

  vec2 delta = abs(vec2(center - left, center - top));
  vec2 edges = step(vec2(threshold), delta);
  if (dot(edges, vec2(1.0)) == 0.0) {
    out_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    return;
  }

  float right = compressedLuma(
      sampleSource(clamp(centerUv + vec2(texel.x, 0.0), minUv, maxUv)).rgb);
  float bottom = compressedLuma(
      sampleSource(clamp(centerUv + vec2(0.0, texel.y), minUv, maxUv)).rgb);
  float leftLeft = compressedLuma(
      sampleSource(clamp(centerUv - vec2(texel.x * 2.0, 0.0), minUv, maxUv))
          .rgb);
  float topTop = compressedLuma(
      sampleSource(clamp(centerUv - vec2(0.0, texel.y * 2.0), minUv, maxUv))
          .rgb);

  vec2 maxDelta = max(delta, abs(vec2(center - right, center - bottom)));
  maxDelta = max(maxDelta, abs(vec2(left - leftLeft, top - topTop)));
  float finalDelta = max(maxDelta.x, maxDelta.y);
  edges *= step(vec2(finalDelta), delta * 2.0);

  out_FragColor = vec4(edges, 0.0, 1.0);
}
