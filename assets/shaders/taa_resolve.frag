#extension GL_EXT_buffer_reference : require

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 out_FragColor;

const uint kTaaResolveFlagHistoryValid = 1u << 0u;
const uint kTaaResolveFlagPreviousVelocityValid = 1u << 1u;
const uint kTaaResolveFlagDepthReject = 1u << 2u;
const uint kTaaResolveFlagVelocityReject = 1u << 3u;
const uint kTaaResolveFlagNeighborhoodClamp = 1u << 4u;
const uint kTaaResolveFlagAdaptiveBlend = 1u << 5u;
const uint kTaaResolveFlagClampBlendAttenuation = 1u << 6u;
const uint kTaaResolveFlagNeighborhoodFallback = 1u << 7u;
const uint kTaaResolveFlagReactiveMask = 1u << 8u;
const uint kTaaResolveFlagVelocityDilation = 1u << 9u;
const uint kTaaResolveModeResolve = 0u;
const uint kTaaResolveModeCopyCurrent = 1u;
const uint kTaaResolveModePreviousHistory = 2u;
const uint kTaaResolveModeHistoryValidity = 3u;
const uint kTaaResolveModeRejectionMask = 4u;
const uint kTaaResolveModeBlendFactor = 5u;
const uint kTaaResolveModeClampDelta = 6u;
const uint kTaaResolveModePixelInspector = 7u;
const uint kTaaResolveModeVelocityMotionVectors = 8u;
const uint kTaaResolveModeVelocityMagnitude = 9u;
const uint kTaaResolveModeReactiveMask = 10u;
const uint kTaaResolveModeDisocclusionMask = 11u;
const uint kTaaResolveModeVelocityDilation = 12u;
const uint kTaaResolveModeReprojectedHistory = 13u;
const uint kTaaResolveModeResolveConfidence = 14u;
const uint kTaaResolveModeClampDiagnostics = 15u;
const uint kTaaResolveModePreviousVelocity = 16u;
const uint kTaaResolveModeHdrWeight = 17u;
const uint kTaaResolveModeHistoryFilterDelta = 18u;
const uint kTaaResolveModeDisocclusionFallback = 19u;
const uint kTaaResolveModeSplitCompare = 20u;
const uint kTaaClampModeClamp = 0u;
const uint kTaaClampModeClip = 1u;
const uint kTaaClampModeVariance = 2u;
const uint kTaaHdrWeightingModeNone = 0u;
const uint kTaaHdrWeightingModeLuminance = 1u;
const uint kTaaHdrWeightingModeLogLuminance = 2u;
const uint kTaaVelocityDilationModeNone = 0u;
const uint kTaaVelocityDilationModeClosestDepth = 1u;
const uint kTaaVelocityDilationModeLargestMagnitude = 2u;
const uint kTaaHistoryFilterModeCatmullRom = 0u;
const uint kTaaHistoryFilterModeBilinear = 1u;
const float kVelocityDebugScale = 64.0;
const float kClampEpsilon = 0.00001;
// Shader-local TAA neighborhood thresholds. Promote these to settings only when
// runtime tuning is needed; the push constant block is kept within 128 bytes.
const float kHighFrequencyLuminanceScaleFloor = 0.25;
const float kHighFrequencySpanLow = 0.015;
const float kHighFrequencySpanHigh = 0.12;
const float kEdgeLuminanceScaleFloor = 0.25;
const float kEdgeSpanLow = 0.03;
const float kEdgeSpanHigh = 0.18;
const float kCenterDeviationLow = 0.01;
const float kCenterDeviationHigh = 0.10;
const float kAntialiasMotionLow = 0.50;
const float kAntialiasMotionHigh = 8.0;
const float kAntialiasMotionScale = 0.22;
const float kAntialiasRejectionLow = 0.10;
const float kAntialiasRejectionHigh = 0.75;
const float kAntialiasRejectionScale = 0.35;
const float kMaxAntialiasFilterStrength = 0.35;

layout(push_constant) uniform TAAResolvePushConstants {
  uint currentTexId;
  uint historyTexId;
  uint depthTexId;
  uint velocityTexId;
  uint previousVelocityTexId;
  uint reactiveMaskTexId;
  uint currentSamplerId;
  uint historySamplerId;
  uint depthSamplerId;
  uint velocitySamplerId;
  uint reactiveMaskSamplerId;
  uint flags;
  uint mode;
  uint currentWeightBits;
  uint inverseWidthBits;
  uint inverseHeightBits;
  uint depthThresholdBits;
  uint velocityThresholdBits;
  uint velocityBlendScaleBits;
  uint disocclusionWeightBits;
  uint clampAttenuationBits;
  uint varianceGammaBits;
  uint hdrWeightStrengthBits;
  uint reactiveCurrentWeightBits;
  uint reactiveStrengthBits;
  uint velocityDilationDepthThresholdBits;
  uint clampMode;
  uint hdrWeightingMode;
  uint velocityDilationMode;
  uint motionCurrentWeightBits;
  uint clampCurrentWeightBits;
  uint historyFilterMode;
}
pc;

struct Neighborhood {
  vec3 minColor;
  vec3 maxColor;
  vec3 avgColor;
  vec3 variance;
  float minLuminance;
  float maxLuminance;
  float minDepth;
  float maxDepth;
};

struct ResolveEvaluation {
  vec3 current;
  vec3 reprojectedHistory;
  vec3 history;
  vec3 resolved;
  float currentAlpha;
  float currentWeight;
  float clampDelta;
  float rawClampDelta;
  float staticClampRelax;
  float velocityDelta;
  float velocityDilationDelta;
  float historyFilterDelta;
  float hdrWeight;
  float reactiveWeight;
  float fallbackStrength;
  float rejectionStrength;
  bool validHistory;
  bool depthRejected;
  bool velocityRejected;
  bool velocityDilated;
};

bool flagEnabled(uint flag) { return (pc.flags & flag) != 0u; }

float pushFloat(uint bits) { return uintBitsToFloat(bits); }

bool uvInBounds(vec2 value) {
  return all(greaterThanEqual(value, vec2(0.0))) &&
         all(lessThan(value, vec2(1.0)));
}

vec2 taaScreenUv(vec2 fullscreenUv) {
  // fullscreen_copy.vert emits Y in the [1, 2] range and relies on repeat
  // sampling for copy passes. TAA performs explicit history bounds checks, so
  // it must evaluate reprojection in normalized screen UV space instead.
  return fract(fullscreenUv);
}

vec4 currentColor(vec2 sampleUv) {
  return textureBindless2D(pc.currentTexId, pc.currentSamplerId, sampleUv);
}

vec4 historyColor(vec2 sampleUv) {
  return textureBindless2D(pc.historyTexId, pc.historySamplerId, sampleUv);
}

// Uses the depth sampler as the shared point sampler for exact texel taps.
vec4 historyColorPoint(vec2 sampleUv) {
  return textureBindless2D(pc.historyTexId, pc.depthSamplerId, sampleUv);
}

vec3 historyColorBilinear(vec2 sampleUv) { return historyColor(sampleUv).rgb; }

float catmullRomWeight(float value) {
  const float x = abs(value);
  if (x <= 1.0) {
    return ((1.5 * x - 2.5) * x) * x + 1.0;
  }
  if (x < 2.0) {
    return (((-0.5 * x + 2.5) * x - 4.0) * x) + 2.0;
  }
  return 0.0;
}

vec3 historyColorCatmullRom(vec2 sampleUv) {
  const vec2 invExtent = vec2(max(pushFloat(pc.inverseWidthBits), 1.0e-8),
                              max(pushFloat(pc.inverseHeightBits), 1.0e-8));
  const vec2 texelPosition = sampleUv / invExtent - vec2(0.5);
  const vec2 baseTexel = floor(texelPosition);
  const vec2 texelFraction = texelPosition - baseTexel;
  const vec2 halfTexel = invExtent * 0.5;
  const vec2 minUv = halfTexel;
  const vec2 maxUv = vec2(1.0) - halfTexel;

  vec3 sum = vec3(0.0);
  float weightSum = 0.0;
  for (int y = -1; y <= 2; ++y) {
    const float weightY = catmullRomWeight(float(y) - texelFraction.y);
    for (int x = -1; x <= 2; ++x) {
      const float weight =
          weightY * catmullRomWeight(float(x) - texelFraction.x);
      const vec2 texelUv =
          clamp((baseTexel + vec2(float(x), float(y)) + vec2(0.5)) * invExtent,
                minUv, maxUv);
      sum += historyColorPoint(texelUv).rgb * weight;
      weightSum += weight;
    }
  }
  return sum / max(weightSum, 1.0e-5);
}

vec3 historyColorFiltered(vec2 sampleUv) {
  if (pc.historyFilterMode == kTaaHistoryFilterModeBilinear) {
    return historyColorBilinear(sampleUv);
  }
  return historyColorCatmullRom(sampleUv);
}

float sceneDepth(vec2 sampleUv) {
  return textureBindless2D(pc.depthTexId, pc.depthSamplerId, sampleUv).r;
}

vec2 velocity(vec2 sampleUv) {
  return textureBindless2D(pc.velocityTexId, pc.velocitySamplerId, sampleUv).rg;
}

vec2 previousVelocity(vec2 sampleUv) {
  return textureBindless2D(pc.previousVelocityTexId, pc.velocitySamplerId,
                           sampleUv)
      .rg;
}

float reactiveMask(vec2 sampleUv) {
  return textureBindless2D(pc.reactiveMaskTexId, pc.reactiveMaskSamplerId,
                           sampleUv)
      .r;
}

vec2 dilatedVelocity(vec2 centerUv, float centerDepth, out bool affected) {
  affected = false;
  const vec2 baseVelocity = velocity(centerUv);
  if (!flagEnabled(kTaaResolveFlagVelocityDilation) ||
      pc.velocityDilationMode == kTaaVelocityDilationModeNone) {
    return baseVelocity;
  }

  const vec2 invExtent = vec2(max(pushFloat(pc.inverseWidthBits), 0.0),
                              max(pushFloat(pc.inverseHeightBits), 0.0));
  const float threshold =
      max(pushFloat(pc.velocityDilationDepthThresholdBits), 0.0);
  float minDepth = centerDepth;
  float maxDepth = centerDepth;
  float closestDepth = centerDepth;
  float largestMagnitude = length(baseVelocity);
  vec2 closestVelocity = baseVelocity;
  vec2 largestVelocity = baseVelocity;

  for (int y = -1; y <= 1; ++y) {
    for (int x = -1; x <= 1; ++x) {
      if (x == 0 && y == 0) {
        continue;
      }
      const vec2 sampleUv =
          clamp(centerUv + vec2(float(x), float(y)) * invExtent, vec2(0.0),
                vec2(1.0));
      const float sampleDepth = sceneDepth(sampleUv);
      const vec2 sampleVelocity = velocity(sampleUv);
      minDepth = min(minDepth, sampleDepth);
      maxDepth = max(maxDepth, sampleDepth);
      if (sampleDepth < closestDepth) {
        closestDepth = sampleDepth;
        closestVelocity = sampleVelocity;
      }
      const float sampleMagnitude = length(sampleVelocity);
      if (sampleMagnitude > largestMagnitude) {
        largestMagnitude = sampleMagnitude;
        largestVelocity = sampleVelocity;
      }
    }
  }

  if ((maxDepth - minDepth) <= threshold) {
    return baseVelocity;
  }

  const vec2 selectedVelocity =
      pc.velocityDilationMode == kTaaVelocityDilationModeLargestMagnitude
          ? largestVelocity
          : closestVelocity;
  affected = length(selectedVelocity - baseVelocity) > 1.0e-7;
  return selectedVelocity;
}

vec3 heatmap(float value) {
  float t = clamp(value, 0.0, 1.0);
  return clamp(vec3(1.5) - abs(vec3(4.0, 4.0, 4.0) * t - vec3(3.0, 2.0, 1.0)),
               vec3(0.0), vec3(1.0));
}

vec4 signedVelocityDebugColor(vec2 sampleVelocity) {
  const float magnitude = length(sampleVelocity) * kVelocityDebugScale;
  const vec2 signedVelocity =
      clamp(sampleVelocity * kVelocityDebugScale * 0.5 + vec2(0.5), vec2(0.0),
            vec2(1.0));
  return vec4(signedVelocity.x, signedVelocity.y, clamp(magnitude, 0.0, 1.0),
              1.0);
}

vec4 velocityDebugColor(vec2 sampleUv, uint mode) {
  const vec2 sampleVelocity = velocity(sampleUv);
  const float magnitude = length(sampleVelocity) * kVelocityDebugScale;
  return mode == kTaaResolveModeVelocityMagnitude
             ? vec4(heatmap(magnitude), 1.0)
             : signedVelocityDebugColor(sampleVelocity);
}

float luminance(vec3 color) {
  return dot(max(color, vec3(0.0)), vec3(0.2126, 0.7152, 0.0722));
}

Neighborhood currentNeighborhood(vec2 centerUv) {
  const vec2 invExtent = vec2(max(pushFloat(pc.inverseWidthBits), 0.0),
                              max(pushFloat(pc.inverseHeightBits), 0.0));
  Neighborhood neighborhood;
  neighborhood.minColor = vec3(3.402823466e+38);
  neighborhood.maxColor = vec3(-3.402823466e+38);
  neighborhood.avgColor = vec3(0.0);
  vec3 colorSquareSum = vec3(0.0);
  neighborhood.minLuminance = 3.402823466e+38;
  neighborhood.maxLuminance = -3.402823466e+38;
  neighborhood.minDepth = 1.0;
  neighborhood.maxDepth = 0.0;

  for (int y = -1; y <= 1; ++y) {
    for (int x = -1; x <= 1; ++x) {
      const vec2 sampleUv =
          clamp(centerUv + vec2(float(x), float(y)) * invExtent, vec2(0.0),
                vec2(1.0));
      const vec3 color = currentColor(sampleUv).rgb;
      const float depth = sceneDepth(sampleUv);
      const float colorLuminance = luminance(color);
      neighborhood.minColor = min(neighborhood.minColor, color);
      neighborhood.maxColor = max(neighborhood.maxColor, color);
      neighborhood.avgColor += color;
      colorSquareSum += color * color;
      neighborhood.minLuminance =
          min(neighborhood.minLuminance, colorLuminance);
      neighborhood.maxLuminance =
          max(neighborhood.maxLuminance, colorLuminance);
      neighborhood.minDepth = min(neighborhood.minDepth, depth);
      neighborhood.maxDepth = max(neighborhood.maxDepth, depth);
    }
  }

  neighborhood.avgColor *= 1.0 / 9.0;
  neighborhood.variance = max(colorSquareSum * (1.0 / 9.0) -
                                  neighborhood.avgColor * neighborhood.avgColor,
                              vec3(0.0));
  return neighborhood;
}

vec3 clipToAabb(vec3 value, vec3 minValue, vec3 maxValue) {
  const vec3 center = (minValue + maxValue) * 0.5;
  const vec3 extents = max((maxValue - minValue) * 0.5, vec3(1.0e-5));
  const vec3 offset = value - center;
  const vec3 unit = abs(offset) / extents;
  const float maxUnit = max(max(unit.x, unit.y), unit.z);
  return maxUnit > 1.0 ? center + offset / maxUnit : value;
}

float hdrWeight(vec3 current, vec3 history, float strength) {
  const uint mode = pc.hdrWeightingMode;
  if (mode == kTaaHdrWeightingModeNone || strength <= 0.0) {
    return 0.0;
  }

  const float currentLuma = luminance(current);
  const float historyLuma = luminance(history);
  float delta = 0.0;
  if (mode == kTaaHdrWeightingModeLogLuminance) {
    delta = abs(log2(1.0 + currentLuma) - log2(1.0 + historyLuma));
  } else {
    delta = abs(currentLuma - historyLuma) /
            max(max(currentLuma, historyLuma), 1.0);
  }
  return clamp(delta * strength, 0.0, 1.0);
}

vec3 validateHistoryColor(vec3 history, Neighborhood neighborhood,
                          out float clampDelta) {
  const uint mode = pc.clampMode;
  vec3 validated = history;
  if (mode == kTaaClampModeVariance) {
    const float gamma = max(pushFloat(pc.varianceGammaBits), 0.0);
    const vec3 sigma = sqrt(neighborhood.variance);
    const vec3 minColor =
        max(neighborhood.minColor, neighborhood.avgColor - sigma * gamma);
    const vec3 maxColor =
        min(neighborhood.maxColor, neighborhood.avgColor + sigma * gamma);
    validated = clipToAabb(history, minColor, maxColor);
  } else if (mode == kTaaClampModeClip) {
    validated =
        clipToAabb(history, neighborhood.minColor, neighborhood.maxColor);
  } else {
    validated = clamp(history, neighborhood.minColor, neighborhood.maxColor);
  }
  clampDelta = length(history - validated);
  return validated;
}

float neighborhoodHighFrequencyStrength(Neighborhood neighborhood) {
  const float avgLuminance = luminance(neighborhood.avgColor);
  const float luminanceScale =
      max(abs(avgLuminance), kHighFrequencyLuminanceScaleFloor);
  const float neighborhoodSpan =
      (neighborhood.maxLuminance - neighborhood.minLuminance) / luminanceScale;
  return smoothstep(kHighFrequencySpanLow, kHighFrequencySpanHigh,
                    neighborhoodSpan);
}

float neighborhoodEdgeStrength(vec3 current, Neighborhood neighborhood) {
  const float currentLuminance = luminance(current);
  const float avgLuminance = luminance(neighborhood.avgColor);
  const float luminanceScale = max(
      max(abs(currentLuminance), abs(avgLuminance)), kEdgeLuminanceScaleFloor);
  const float neighborhoodSpan =
      (neighborhood.maxLuminance - neighborhood.minLuminance) / luminanceScale;
  const float centerDeviation =
      abs(currentLuminance - avgLuminance) / luminanceScale;
  return smoothstep(kEdgeSpanLow, kEdgeSpanHigh, neighborhoodSpan) *
         smoothstep(kCenterDeviationLow, kCenterDeviationHigh, centerDeviation);
}

vec3 antialiasCurrentColor(vec3 current, Neighborhood neighborhood,
                           float motionPixels, float disocclusionBlend,
                           float clampConfidence, float rejectionStrength) {
  const float edgeStrength = neighborhoodEdgeStrength(current, neighborhood);
  const float motionFilter =
      smoothstep(kAntialiasMotionLow, kAntialiasMotionHigh, motionPixels) *
      kAntialiasMotionScale;
  const float rejectionFilter =
      smoothstep(
          kAntialiasRejectionLow, kAntialiasRejectionHigh,
          max(max(disocclusionBlend, clampConfidence), rejectionStrength)) *
      kAntialiasRejectionScale;
  const float filterStrength =
      clamp(edgeStrength * max(motionFilter, rejectionFilter), 0.0,
            kMaxAntialiasFilterStrength);
  return mix(current, neighborhood.avgColor, filterStrength);
}

ResolveEvaluation evaluateResolve(vec2 currentUv) {
  const vec4 currentSample = currentColor(currentUv);
  const vec3 current = currentSample.rgb;
  const float baseCurrentWeight =
      clamp(pushFloat(pc.currentWeightBits), 0.0, 1.0);

  ResolveEvaluation result;
  result.current = current;
  result.reprojectedHistory = current;
  result.history = current;
  result.resolved = current;
  result.currentAlpha = currentSample.a;
  result.currentWeight = 1.0;
  result.clampDelta = 0.0;
  result.rawClampDelta = 0.0;
  result.staticClampRelax = 0.0;
  result.velocityDelta = 0.0;
  result.velocityDilationDelta = 0.0;
  result.hdrWeight = 0.0;
  result.reactiveWeight = 0.0;
  result.rejectionStrength = 0.0;
  result.validHistory = false;
  result.depthRejected = false;
  result.velocityRejected = false;
  result.velocityDilated = false;
  result.historyFilterDelta = 0.0;
  result.fallbackStrength = 0.0;

  if (!flagEnabled(kTaaResolveFlagHistoryValid)) {
    result.rejectionStrength = 1.0;
    return result;
  }

  const float depth = sceneDepth(currentUv);
  // Clear-depth/background pixels use current color until skybox/background
  // reprojection has an explicit policy.
  if (depth >= 0.999999) {
    result.rejectionStrength = 1.0;
    return result;
  }

  const vec2 baseVelocity = velocity(currentUv);
  bool velocityDilated = false;
  const vec2 currentVelocity =
      dilatedVelocity(currentUv, depth, velocityDilated);
  result.velocityDilated = velocityDilated;
  result.velocityDilationDelta = length(currentVelocity - baseVelocity);
  const vec2 historyUv = currentUv + currentVelocity;
  if (!uvInBounds(historyUv)) {
    result.rejectionStrength = 1.0;
    return result;
  }

  const float velocityThreshold = max(pushFloat(pc.velocityThresholdBits), 0.0);
  const float velocityBlendScale =
      max(pushFloat(pc.velocityBlendScaleBits), 0.0);
  const vec2 invExtent = vec2(max(pushFloat(pc.inverseWidthBits), 1.0e-8),
                              max(pushFloat(pc.inverseHeightBits), 1.0e-8));
  const float motionPixels = length(currentVelocity / invExtent);
  const float motionBlend = clamp(motionPixels * velocityBlendScale, 0.0, 1.0);
  bool velocityRejected = false;
  float velocityRejectionStrength = 0.0;
  if (flagEnabled(kTaaResolveFlagVelocityReject) &&
      flagEnabled(kTaaResolveFlagPreviousVelocityValid) &&
      velocityThreshold > 0.0) {
    result.velocityDelta =
        length(currentVelocity - previousVelocity(historyUv));
    velocityRejected = result.velocityDelta > velocityThreshold;
    velocityRejectionStrength =
        clamp(result.velocityDelta / max(velocityThreshold, 1.0e-5), 0.0, 1.0);
    result.velocityRejected = velocityRejected;
  }

  const Neighborhood neighborhood = currentNeighborhood(currentUv);
  const float depthThreshold = max(pushFloat(pc.depthThresholdBits), 0.0);
  const bool depthRejected =
      flagEnabled(kTaaResolveFlagDepthReject) &&
      (neighborhood.maxDepth - neighborhood.minDepth) > depthThreshold;
  result.depthRejected = depthRejected;
  // Depth and velocity discontinuities are diagnostics and blend-confidence
  // inputs. A hard current-color fallback here prevents static geometric edges
  // from ever accumulating the jitter sequence.
  const float depthRejectionStrength = depthRejected ? motionBlend * 0.5 : 0.0;
  result.rejectionStrength =
      max(velocityRejectionStrength, depthRejectionStrength);
  const float edgeStrength = neighborhoodEdgeStrength(current, neighborhood);
  const float highFrequencyStrength =
      max(edgeStrength, neighborhoodHighFrequencyStrength(neighborhood));
  const float stableHistoryBlend =
      (1.0 - smoothstep(0.02, 0.35, motionPixels)) *
      (1.0 - velocityRejectionStrength);

  result.validHistory = true;
  vec3 history = historyColorFiltered(historyUv);
  if (pc.mode == kTaaResolveModeHistoryFilterDelta) {
    const vec3 catmullRomHistory =
        pc.historyFilterMode == kTaaHistoryFilterModeCatmullRom
            ? history
            : historyColorCatmullRom(historyUv);
    const vec3 bilinearHistory =
        pc.historyFilterMode == kTaaHistoryFilterModeBilinear
            ? history
            : historyColorBilinear(historyUv);
    result.historyFilterDelta = length(catmullRomHistory - bilinearHistory);
  }
  result.reprojectedHistory = history;
  float clampDelta = 0.0;
  vec3 clampedHistory =
      flagEnabled(kTaaResolveFlagNeighborhoodClamp)
          ? validateHistoryColor(history, neighborhood, clampDelta)
          : history;
  if (flagEnabled(kTaaResolveFlagNeighborhoodClamp)) {
    result.rawClampDelta = clampDelta;
    result.clampDelta = clampDelta;
    if (result.clampDelta > kClampEpsilon) {
      // On static subpixel geometry the jittered current neighborhood can
      // over-constrain an already aligned history sample. Keep the clamp
      // strict once reprojection is moving, but let stable pixels retain more
      // of their accumulated history.
      const float staticClampRelax =
          stableHistoryBlend * mix(0.70, 0.99, highFrequencyStrength);
      result.staticClampRelax = staticClampRelax;
      clampedHistory = mix(clampedHistory, history, staticClampRelax);
      result.clampDelta = length(history - clampedHistory);
    }
  }
  result.history = clampedHistory;

  float currentWeight = baseCurrentWeight;
  const float dynamicDisocclusionBlend =
      max(velocityRejectionStrength, depthRejectionStrength);
  const float neighborhoodColorSpan =
      max(length(neighborhood.maxColor - neighborhood.minColor), 1.0e-4);
  const float stableHighFrequencyBlend =
      stableHistoryBlend * highFrequencyStrength;
  float clampConfidence = 0.0;
  if (flagEnabled(kTaaResolveFlagAdaptiveBlend)) {
    const float motionCurrentWeight =
        clamp(pushFloat(pc.motionCurrentWeightBits), baseCurrentWeight, 1.0);
    currentWeight =
        max(currentWeight, mix(baseCurrentWeight, motionCurrentWeight,
                               clamp(motionBlend, 0.0, 1.0)));

    const float disocclusionWeight =
        clamp(pushFloat(pc.disocclusionWeightBits), baseCurrentWeight, 1.0);
    const float strongMotionBlend = smoothstep(6.0, 24.0, motionPixels);
    const float coherentMotionWeight =
        mix(motionCurrentWeight, disocclusionWeight, 0.25);
    currentWeight = max(currentWeight, mix(currentWeight, coherentMotionWeight,
                                           strongMotionBlend));
    result.rejectionStrength =
        max(result.rejectionStrength, strongMotionBlend * 0.10);

    currentWeight =
        max(currentWeight, mix(baseCurrentWeight, disocclusionWeight,
                               dynamicDisocclusionBlend));

    float clampStrength = 0.0;
    float confidenceForBlend = 0.0;
    if (flagEnabled(kTaaResolveFlagNeighborhoodClamp) &&
        result.clampDelta > kClampEpsilon) {
      clampStrength =
          clamp(result.clampDelta / neighborhoodColorSpan, 0.0, 1.0);
      clampConfidence = clamp(
          clampStrength * max(motionBlend, dynamicDisocclusionBlend), 0.0, 1.0);
      const float clampCurrentWeight =
          clamp(pushFloat(pc.clampCurrentWeightBits), currentWeight, 1.0);
      currentWeight = max(currentWeight, mix(currentWeight, clampCurrentWeight,
                                             clampConfidence));
      result.rejectionStrength =
          max(result.rejectionStrength, clampConfidence * 0.25);
      confidenceForBlend = clampConfidence;
    }
    const float confidenceBlend =
        max(dynamicDisocclusionBlend, confidenceForBlend);

    if (flagEnabled(kTaaResolveFlagClampBlendAttenuation) &&
        result.clampDelta > kClampEpsilon && confidenceBlend > 0.0) {
      const float clampAttenuation =
          clamp(pushFloat(pc.clampAttenuationBits), 0.0, 1.0);
      const float attenuationBlend =
          clamp(clampAttenuation * confidenceBlend, 0.0, 1.0);
      currentWeight = max(currentWeight, mix(currentWeight, disocclusionWeight,
                                             attenuationBlend));
    }
    const float hdrWeightStrength =
        clamp(pushFloat(pc.hdrWeightStrengthBits), 0.0, 1.0);
    result.hdrWeight = hdrWeight(current, clampedHistory, hdrWeightStrength);
    if (result.hdrWeight > 0.0 && confidenceBlend > 0.0) {
      const float hdrDisocclusionBlend =
          clamp(result.hdrWeight * confidenceBlend, 0.0, 1.0);
      currentWeight = max(currentWeight, mix(currentWeight, disocclusionWeight,
                                             hdrDisocclusionBlend));
    }
  }
  if (flagEnabled(kTaaResolveFlagReactiveMask)) {
    const float reactiveStrength = max(pushFloat(pc.reactiveStrengthBits), 0.0);
    result.reactiveWeight =
        clamp(reactiveMask(currentUv) * reactiveStrength, 0.0, 1.0);
    if (result.reactiveWeight > 0.0) {
      const float reactiveCurrentWeight =
          clamp(pushFloat(pc.reactiveCurrentWeightBits), currentWeight, 1.0);
      currentWeight =
          max(currentWeight,
              mix(currentWeight, reactiveCurrentWeight, result.reactiveWeight));
    }
  }

  if (flagEnabled(kTaaResolveFlagAdaptiveBlend) &&
      stableHighFrequencyBlend > 0.0 && result.reactiveWeight <= 0.0) {
    const float stableCurrentWeight =
        min(currentWeight, baseCurrentWeight * 0.15);
    currentWeight =
        mix(currentWeight, stableCurrentWeight, stableHighFrequencyBlend);
  }
  result.currentWeight = clamp(currentWeight, 0.0, 1.0);
  vec3 currentResolveColor =
      flagEnabled(kTaaResolveFlagAdaptiveBlend)
          ? antialiasCurrentColor(current, neighborhood, motionPixels,
                                  dynamicDisocclusionBlend, clampConfidence,
                                  result.rejectionStrength)
          : current;
  if (flagEnabled(kTaaResolveFlagNeighborhoodFallback) &&
      result.rejectionStrength > 0.0) {
    const float fallbackStrength =
        smoothstep(0.75, 1.0, clamp(result.rejectionStrength, 0.0, 1.0));
    result.fallbackStrength = fallbackStrength;
    currentResolveColor = mix(current, neighborhood.avgColor, fallbackStrength);
  }
  result.resolved =
      mix(clampedHistory, currentResolveColor, result.currentWeight);
  return result;
}

void main() {
  const vec2 screenUv = taaScreenUv(uv);

  if (pc.mode == kTaaResolveModeVelocityMotionVectors ||
      pc.mode == kTaaResolveModeVelocityMagnitude) {
    out_FragColor = velocityDebugColor(screenUv, pc.mode);
    return;
  }
  if (pc.mode == kTaaResolveModePreviousVelocity) {
    if (!flagEnabled(kTaaResolveFlagPreviousVelocityValid)) {
      out_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
      return;
    }
    out_FragColor = signedVelocityDebugColor(previousVelocity(screenUv));
    return;
  }
  if (pc.mode == kTaaResolveModeCopyCurrent) {
    out_FragColor = currentColor(screenUv);
    return;
  }
  if (pc.mode == kTaaResolveModePreviousHistory) {
    out_FragColor = historyColor(screenUv);
    return;
  }
  if (pc.mode == kTaaResolveModeReactiveMask) {
    out_FragColor = vec4(vec3(reactiveMask(screenUv)), 1.0);
    return;
  }

  const ResolveEvaluation eval = evaluateResolve(screenUv);

  if (pc.mode == kTaaResolveModeHistoryValidity) {
    out_FragColor =
        eval.validHistory ? vec4(0.0, 1.0, 0.0, 1.0) : vec4(1.0, 0.0, 0.0, 1.0);
    return;
  }
  if (pc.mode == kTaaResolveModeRejectionMask) {
    out_FragColor = vec4(eval.rejectionStrength, eval.depthRejected ? 1.0 : 0.0,
                         eval.velocityRejected ? 1.0 : 0.0, 1.0);
    return;
  }
  if (pc.mode == kTaaResolveModeBlendFactor) {
    out_FragColor = vec4(vec3(eval.currentWeight), 1.0);
    return;
  }
  if (pc.mode == kTaaResolveModeClampDelta) {
    const float delta = clamp(eval.clampDelta * 8.0, 0.0, 1.0);
    out_FragColor = vec4(delta, delta, delta, 1.0);
    return;
  }
  if (pc.mode == kTaaResolveModeClampDiagnostics) {
    out_FragColor = vec4(clamp(eval.rawClampDelta * 8.0, 0.0, 1.0),
                         clamp(eval.clampDelta * 8.0, 0.0, 1.0),
                         clamp(eval.staticClampRelax, 0.0, 1.0), 1.0);
    return;
  }
  if (pc.mode == kTaaResolveModeHdrWeight) {
    out_FragColor = vec4(heatmap(eval.hdrWeight), 1.0);
    return;
  }
  if (pc.mode == kTaaResolveModeHistoryFilterDelta) {
    out_FragColor =
        vec4(heatmap(clamp(eval.historyFilterDelta * 8.0, 0.0, 1.0)), 1.0);
    return;
  }
  if (pc.mode == kTaaResolveModeDisocclusionFallback) {
    out_FragColor = vec4(heatmap(eval.fallbackStrength), 1.0);
    return;
  }
  if (pc.mode == kTaaResolveModeDisocclusionMask) {
    out_FragColor = vec4(eval.rejectionStrength, eval.depthRejected ? 1.0 : 0.0,
                         eval.velocityRejected ? 1.0 : 0.0, 1.0);
    return;
  }
  if (pc.mode == kTaaResolveModeVelocityDilation) {
    out_FragColor =
        vec4(heatmap(clamp(eval.velocityDilationDelta * kVelocityDebugScale,
                           0.0, 1.0)),
             1.0);
    return;
  }
  if (pc.mode == kTaaResolveModeReprojectedHistory) {
    out_FragColor = vec4(eval.reprojectedHistory, 1.0);
    return;
  }
  if (pc.mode == kTaaResolveModeResolveConfidence) {
    out_FragColor =
        vec4(eval.currentWeight, eval.rejectionStrength,
             clamp(eval.velocityDelta /
                       max(pushFloat(pc.velocityThresholdBits), 1.0e-5),
                   0.0, 1.0),
             1.0);
    return;
  }
  if (pc.mode == kTaaResolveModePixelInspector) {
    if (screenUv.x < 0.5 && screenUv.y < 0.5) {
      out_FragColor = vec4(eval.current, 1.0);
    } else if (screenUv.x >= 0.5 && screenUv.y < 0.5) {
      out_FragColor = vec4(eval.reprojectedHistory, 1.0);
    } else if (screenUv.x < 0.5) {
      out_FragColor = vec4(eval.resolved, 1.0);
    } else {
      out_FragColor =
          vec4(eval.rejectionStrength, eval.currentWeight,
               clamp(eval.velocityDelta /
                         max(pushFloat(pc.velocityThresholdBits), 1.0e-5),
                     0.0, 1.0),
               1.0);
    }
    return;
  }
  if (pc.mode == kTaaResolveModeSplitCompare) {
    const float invWidth = pushFloat(pc.inverseWidthBits);
    const float splitUvX =
        invWidth > 0.0 ? gl_FragCoord.x * invWidth : screenUv.x;
    if (invWidth > 0.0 && abs(splitUvX - 0.5) <= invWidth) {
      out_FragColor = vec4(vec3(1.0), eval.currentAlpha);
      return;
    }
    out_FragColor = splitUvX < 0.5 ? vec4(eval.current, eval.currentAlpha)
                                   : vec4(eval.resolved, eval.currentAlpha);
    return;
  }

  out_FragColor = vec4(eval.resolved, eval.currentAlpha);
}
