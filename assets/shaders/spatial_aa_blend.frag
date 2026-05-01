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
  uint localContrastFactorBits;
  uint cornerRoundingBits;
}
pc;

float pushFloat(uint bits) { return uintBitsToFloat(bits); }

vec2 screenUv(vec2 fullscreenUv) { return fract(fullscreenUv); }

vec4 sampleSource(vec2 sampleUv) {
  return textureBindless2D(pc.sourceTexId, pc.pointSamplerId, sampleUv);
}

vec4 sampleEdges(vec2 sampleUv) {
  return textureBindless2D(pc.edgeTexId, pc.pointSamplerId, sampleUv);
}

vec4 sampleArea(vec2 sampleUv) {
  return textureBindless2D(pc.areaTexId, pc.linearSamplerId, sampleUv);
}

float sampleSearch(vec2 sampleUv) {
  return textureBindless2D(pc.searchTexId, pc.pointSamplerId, sampleUv).r;
}

float compressedLuma(vec3 color) {
  vec3 compressed = max(color, vec3(0.0));
  compressed = compressed / (vec3(1.0) + compressed);
  return dot(compressed, vec3(0.2126, 0.7152, 0.0722));
}

float edgeResponse(float delta, float threshold) {
  return smoothstep(threshold, threshold * 2.5, delta);
}

float edgeComponent(vec2 edge, uint component) {
  return component == 0u ? edge.r : edge.g;
}

float sampleEdgeComponent(vec2 sampleUv, uint component) {
  return edgeComponent(sampleEdges(sampleUv).rg, component);
}

float searchDistance(vec2 centerUv, vec2 stepUv, uint component) {
  float distanceValue = 0.0;
  uint maxSteps = max(pc.maxSearchSteps, 1u);
  for (uint i = 1u; i <= maxSteps; ++i) {
    vec2 sampleUv = clamp(centerUv + stepUv * float(i), vec2(0.0), vec2(1.0));
    float edgeActive = edgeComponent(sampleEdges(sampleUv).rg, component);
    distanceValue = float(i);
    if (edgeActive < 0.05) {
      break;
    }
  }
  return distanceValue;
}

float cornerAttenuation(vec2 centerUv, vec2 texel, uint majorComponent) {
  float rounding = clamp(pushFloat(pc.cornerRoundingBits), 0.0, 1.0);
  uint crossComponent = 1u - majorComponent;
  vec2 perpendicular =
      majorComponent == 0u ? vec2(0.0, texel.y) : vec2(texel.x, 0.0);
  float crossing = sampleEdgeComponent(centerUv, crossComponent);
  crossing = max(crossing, sampleEdgeComponent(clamp(centerUv + perpendicular,
                                                     vec2(0.0), vec2(1.0)),
                                               crossComponent));
  crossing = max(crossing, sampleEdgeComponent(clamp(centerUv - perpendicular,
                                                     vec2(0.0), vec2(1.0)),
                                               crossComponent));
  crossing =
      max(crossing, sampleEdgeComponent(clamp(centerUv + vec2(texel.x, texel.y),
                                              vec2(0.0), vec2(1.0)),
                                        crossComponent));
  crossing =
      max(crossing, sampleEdgeComponent(clamp(centerUv - vec2(texel.x, texel.y),
                                              vec2(0.0), vec2(1.0)),
                                        crossComponent));
  return 1.0 - clamp(crossing * rounding, 0.0, 0.75);
}

float areaWeight(float negativeDistance, float positiveDistance,
                 float crossingStrength) {
  float maxDistance = float(max(pc.maxSearchSteps, 1u));
  vec2 areaUv = vec2(clamp(negativeDistance / maxDistance, 0.0, 1.0),
                     clamp(positiveDistance / maxDistance, 0.0, 1.0));
  vec2 searchUv = vec2(
      clamp((negativeDistance + positiveDistance) / max(maxDistance * 2.0, 1.0),
            0.0, 1.0),
      crossingStrength);
  float searchBias = sampleSearch(searchUv);
  vec2 area = sampleArea(areaUv).rg;
  return clamp(max(area.r, area.g) * 0.75 + searchBias * 0.25, 0.0, 1.0);
}

void main() {
  vec2 centerUv = screenUv(uv);
  vec2 texel = vec2(max(pushFloat(pc.inverseWidthBits), 1.0e-8),
                    max(pushFloat(pc.inverseHeightBits), 1.0e-8));
  vec2 minUv = texel * 0.5;
  vec2 maxUv = vec2(1.0) - minUv;
  vec2 edges = sampleEdges(centerUv).rg;
  float threshold = max(pushFloat(pc.edgeThresholdBits), 1.0e-5);
  float maxBlend = clamp(pushFloat(pc.resolveStrengthBits), 0.0, 1.0);

  float center = compressedLuma(sampleSource(centerUv).rgb);
  float left = compressedLuma(
      sampleSource(clamp(centerUv - vec2(texel.x, 0.0), minUv, maxUv)).rgb);
  float top = compressedLuma(
      sampleSource(clamp(centerUv - vec2(0.0, texel.y), minUv, maxUv)).rgb);

  float leftDelta = abs(center - left);
  float topDelta = abs(center - top);

  float leftWeight = 0.0;
  float topWeight = 0.0;

  if (edges.r > 0.05) {
    float leftDistance = searchDistance(centerUv, vec2(-texel.x, 0.0), 0u);
    float rightDistance = searchDistance(centerUv, vec2(texel.x, 0.0), 0u);
    float continuity =
        0.35 + 0.65 * areaWeight(leftDistance, rightDistance, edges.g);
    leftWeight = edges.r * 0.24 * continuity *
                 cornerAttenuation(centerUv, texel, 0u) *
                 edgeResponse(leftDelta, threshold);
  }

  if (edges.g > 0.05) {
    float topDistance = searchDistance(centerUv, vec2(0.0, -texel.y), 1u);
    float bottomDistance = searchDistance(centerUv, vec2(0.0, texel.y), 1u);
    float continuity =
        0.35 + 0.65 * areaWeight(topDistance, bottomDistance, edges.r);
    topWeight = edges.g * 0.24 * continuity *
                cornerAttenuation(centerUv, texel, 1u) *
                edgeResponse(topDelta, threshold);
  }

  vec4 weights = vec4(leftWeight, topWeight, 0.0, 0.0);
  float weightSum = weights.r + weights.g + weights.b + weights.a;
  if (weightSum > maxBlend) {
    weights *= maxBlend / weightSum;
  }

  out_FragColor = weights;
}
