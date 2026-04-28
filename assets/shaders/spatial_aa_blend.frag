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
  float luma = dot(max(color, vec3(0.0)), vec3(0.2126, 0.7152, 0.0722));
  return log2(1.0 + luma);
}

float edgeResponse(float delta, float threshold) {
  return smoothstep(threshold, threshold * 3.0, delta);
}

float searchDistance(vec2 centerUv, vec2 stepUv, int component) {
  float distanceValue = 0.0;
  uint maxSteps = max(pc.maxSearchSteps, 1u);
  for (uint i = 1u; i <= maxSteps; ++i) {
    vec2 sampleUv = clamp(centerUv + stepUv * float(i), vec2(0.0), vec2(1.0));
    vec2 edge = sampleEdges(sampleUv).rg;
    float edgeActive = component == 0 ? edge.r : edge.g;
    distanceValue = float(i);
    if (edgeActive < 0.5) {
      break;
    }
  }
  return distanceValue;
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

  if (edges.r > 0.5) {
    float leftDistance = searchDistance(centerUv, vec2(-texel.x, 0.0), 0);
    float rightDistance = searchDistance(centerUv, vec2(texel.x, 0.0), 0);
    float continuity =
        0.35 + 0.65 * areaWeight(leftDistance, rightDistance, edges.g);
    leftWeight = 0.24 * continuity * edgeResponse(leftDelta, threshold);
  }

  if (edges.g > 0.5) {
    float topDistance = searchDistance(centerUv, vec2(0.0, -texel.y), 1);
    float bottomDistance = searchDistance(centerUv, vec2(0.0, texel.y), 1);
    float continuity =
        0.35 + 0.65 * areaWeight(topDistance, bottomDistance, edges.r);
    topWeight = 0.24 * continuity * edgeResponse(topDelta, threshold);
  }

  vec4 weights = vec4(leftWeight, topWeight, 0.0, 0.0);
  float weightSum = weights.r + weights.g + weights.b + weights.a;
  if (weightSum > maxBlend) {
    weights *= maxBlend / weightSum;
  }

  out_FragColor = weights;
}
