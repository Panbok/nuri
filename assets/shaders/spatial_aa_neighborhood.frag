#extension GL_EXT_buffer_reference : require

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 out_FragColor;

const uint kSpatialAAModeNeighborhood = 0u;
const uint kSpatialAAModeCopy = 1u;
const uint kSpatialAAModeCleanupMask = 2u;
const uint kSpatialAAModeSplitCompare = 3u;

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
  uint localContrastFactorBits;
  uint cornerRoundingBits;
}
pc;

float pushFloat(uint bits) { return uintBitsToFloat(bits); }

vec2 screenUv(vec2 fullscreenUv) { return fract(fullscreenUv); }

vec4 sampleSource(vec2 sampleUv) {
  return textureBindless2D(pc.sourceTexId, pc.linearSamplerId, sampleUv);
}

vec4 sampleBlend(vec2 sampleUv) {
  return textureBindless2D(pc.blendTexId, pc.pointSamplerId, sampleUv);
}

vec2 sampleEdges(vec2 sampleUv) {
  return textureBindless2D(pc.edgeTexId, pc.pointSamplerId, sampleUv).rg;
}

vec4 spatialResolve(vec2 centerUv) {
  vec2 texel = vec2(max(pushFloat(pc.inverseWidthBits), 1.0e-8),
                    max(pushFloat(pc.inverseHeightBits), 1.0e-8));
  vec2 minUv = texel * 0.5;
  vec2 maxUv = vec2(1.0) - minUv;
  vec4 center = sampleSource(centerUv);
  vec4 centerWeights = sampleBlend(centerUv);
  float leftWeight = centerWeights.r;
  float topWeight = centerWeights.g;
  float rightWeight =
      sampleBlend(clamp(centerUv + vec2(texel.x, 0.0), minUv, maxUv)).r;
  float bottomWeight =
      sampleBlend(clamp(centerUv + vec2(0.0, texel.y), minUv, maxUv)).g;
  vec4 left = sampleSource(clamp(centerUv - vec2(texel.x, 0.0), minUv, maxUv));
  vec4 top = sampleSource(clamp(centerUv - vec2(0.0, texel.y), minUv, maxUv));
  vec4 right = sampleSource(clamp(centerUv + vec2(texel.x, 0.0), minUv, maxUv));
  vec4 bottom =
      sampleSource(clamp(centerUv + vec2(0.0, texel.y), minUv, maxUv));

  float horizontalWeight = leftWeight + rightWeight;
  float verticalWeight = topWeight + bottomWeight;
  float weightSum = max(horizontalWeight, verticalWeight);
  if (weightSum <= 1.0e-5) {
    return center;
  }

  bool horizontal = horizontalWeight >= verticalWeight;
  vec4 directional = horizontal ? (left * leftWeight + right * rightWeight) /
                                      max(horizontalWeight, 1.0e-5)
                                : (top * topWeight + bottom * bottomWeight) /
                                      max(verticalWeight, 1.0e-5);
  float maxBlend = clamp(pushFloat(pc.resolveStrengthBits), 0.0, 1.0);
  vec2 centerEdges = sampleEdges(centerUv);
  vec2 rightEdges =
      sampleEdges(clamp(centerUv + vec2(texel.x, 0.0), minUv, maxUv));
  vec2 bottomEdges =
      sampleEdges(clamp(centerUv + vec2(0.0, texel.y), minUv, maxUv));
  float edgeGate =
      max(max(centerEdges.r, centerEdges.g),
          max(horizontal ? rightEdges.r : bottomEdges.g, weightSum));
  float strength =
      clamp(weightSum, 0.0, maxBlend) * smoothstep(0.03, 0.35, edgeGate);
  return mix(center, directional, strength);
}

void main() {
  vec2 centerUv = screenUv(uv);
  vec4 source = sampleSource(centerUv);
  if (pc.mode == kSpatialAAModeCopy) {
    out_FragColor = source;
    return;
  }

  vec4 resolved = spatialResolve(centerUv);
  if (pc.mode == kSpatialAAModeCleanupMask) {
    float delta = length(resolved.rgb - source.rgb);
    out_FragColor = vec4(vec3(clamp(delta * 8.0, 0.0, 1.0)), 1.0);
    return;
  }
  if (pc.mode == kSpatialAAModeSplitCompare) {
    out_FragColor = centerUv.x < 0.5 ? source : resolved;
    return;
  }

  out_FragColor = resolved;
}
