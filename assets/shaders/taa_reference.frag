#extension GL_EXT_buffer_reference : require

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 out_FragColor;

const uint kFlagHistoryValid = 1u << 0u;
const uint kFlagPreviousDepthValid = 1u << 1u;
const uint kFlagOrthographicProjection = 1u << 2u;
const uint kModeResolve = 0u;
const uint kMotionClassInvalid = 0u;
const uint kMotionClassStaticCameraOnly = 1u;
const uint kMotionClassFull = 2u;
const uint kMotionClassBackgroundRotation = 3u;

layout(push_constant) uniform ReferenceTAAPushConstants {
  uint currentTexId;
  uint opaqueSceneTexId;
  uint historyTexId;
  uint depthTexId;
  uint previousDepthTexId;
  uint motionTexId;
  uint motionClassTexId;
  uint reactiveTexId;
  uint linearSamplerId;
  uint pointSamplerId;
  uint flags;
  uint mode;
  uint inverseWidthBits;
  uint inverseHeightBits;
  uint nearPlaneBits;
  uint farPlaneBits;
  uint currentWeightBits;
  uint sharpenStrengthBits;
}
pc;

vec3 sampleCurrent(vec2 sampleUv) {
  return textureBindless2D(pc.currentTexId, pc.linearSamplerId, sampleUv).rgb;
}

float linearDepth(float deviceDepth) {
  const float nearPlane = uintBitsToFloat(pc.nearPlaneBits);
  const float farPlane = uintBitsToFloat(pc.farPlaneBits);
  if ((pc.flags & kFlagOrthographicProjection) != 0u) {
    return nearPlane + deviceDepth * (farPlane - nearPlane);
  }
  const float denominator =
      max(farPlane - deviceDepth * (farPlane - nearPlane), 1.0e-6);
  return nearPlane * farPlane / denominator;
}

void main() {
  // fullscreen_copy.vert preserves the renderer's framebuffer orientation by
  // emitting Y in [1, 2] for repeat-sampled copy passes. Temporal sampling is
  // clamped, so normalize that interpolant to canonical screen UV first.
  const vec2 screenUv = vec2(uv.x, uv.y - 1.0);
  const vec4 current =
      textureBindless2D(pc.currentTexId, pc.linearSamplerId, screenUv);
  if (pc.mode != kModeResolve || (pc.flags & kFlagHistoryValid) == 0u) {
    out_FragColor = current;
    return;
  }

  const uint motionClass = uint(round(
      textureBindless2D(pc.motionClassTexId, pc.pointSamplerId, screenUv).r *
      255.0));
  const bool backgroundRotation = motionClass == kMotionClassBackgroundRotation;
  if (motionClass == kMotionClassInvalid ||
      (backgroundRotation && (pc.flags & kFlagPreviousDepthValid) == 0u)) {
    out_FragColor = current;
    return;
  }

  const vec2 motion =
      textureBindless2D(pc.motionTexId, pc.pointSamplerId, screenUv).rg;
  const vec2 historyUv = screenUv + motion;
  if (any(lessThan(historyUv, vec2(0.0))) ||
      any(greaterThan(historyUv, vec2(1.0)))) {
    out_FragColor = current;
    return;
  }

  if ((pc.flags & kFlagPreviousDepthValid) != 0u) {
    const float currentDeviceDepth =
        textureBindless2D(pc.depthTexId, pc.pointSamplerId, screenUv).r;
    const float previousDeviceDepth =
        textureBindless2D(pc.previousDepthTexId, pc.pointSamplerId, historyUv)
            .r;
    if (backgroundRotation) {
      // Rotational sky history is valid only while both samples remain clear
      // background. Foreground coverage entering either pixel is a hard reject.
      if (currentDeviceDepth < 0.999999 || previousDeviceDepth < 0.999999) {
        out_FragColor = current;
        return;
      }
    } else {
      const float currentDepth = linearDepth(currentDeviceDepth);
      const float previousDepth = linearDepth(previousDeviceDepth);
      const float depthTolerance = max(0.015, currentDepth * 0.018);
      if (abs(currentDepth - previousDepth) > depthTolerance) {
        out_FragColor = current;
        return;
      }
    }
  }

  const vec2 texel = vec2(uintBitsToFloat(pc.inverseWidthBits),
                          uintBitsToFloat(pc.inverseHeightBits));
  vec3 neighborhoodMin = vec3(1.0e20);
  vec3 neighborhoodMax = vec3(-1.0e20);
  vec3 crossSum = vec3(0.0);
  for (int y = -1; y <= 1; ++y) {
    for (int x = -1; x <= 1; ++x) {
      const vec3 sampleColor = sampleCurrent(screenUv + vec2(x, y) * texel);
      neighborhoodMin = min(neighborhoodMin, sampleColor);
      neighborhoodMax = max(neighborhoodMax, sampleColor);
      if (abs(x) + abs(y) == 1) {
        crossSum += sampleColor;
      }
    }
  }

  const vec3 history =
      textureBindless2D(pc.historyTexId, pc.linearSamplerId, historyUv).rgb;
  const vec3 clippedHistory = clamp(history, neighborhoodMin, neighborhoodMax);
  const float reactive =
      textureBindless2D(pc.reactiveTexId, pc.pointSamplerId, screenUv).r;
  const vec3 opaqueScene =
      textureBindless2D(pc.opaqueSceneTexId, pc.linearSamplerId, screenUv).rgb;
  const float compositionDelta =
      max(max(abs(current.r - opaqueScene.r), abs(current.g - opaqueScene.g)),
          abs(current.b - opaqueScene.b));
  const float baseCurrentWeight = uintBitsToFloat(pc.currentWeightBits);
  const float motionWeight = clamp(length(motion) * 96.0, 0.0, 0.72);
  const float clampDelta = length(history - clippedHistory);
  float currentWeight = max(baseCurrentWeight, motionWeight);
  currentWeight = max(currentWeight, clamp(reactive * 1.2, 0.0, 1.0));
  currentWeight = max(currentWeight, clamp(clampDelta * 2.0, 0.0, 0.85));
  if (compositionDelta > 0.012) {
    currentWeight = 1.0;
  }

  vec3 resolved = mix(clippedHistory, current.rgb, currentWeight);
  const vec3 sharpenedCurrent = current.rgb * 2.0 - crossSum * 0.25;
  const float sharpenStrength = uintBitsToFloat(pc.sharpenStrengthBits);
  resolved += (sharpenedCurrent - current.rgb) * sharpenStrength *
              (0.35 + 0.65 * currentWeight);
  resolved = clamp(resolved, neighborhoodMin, neighborhoodMax);
  out_FragColor = vec4(max(resolved, vec3(0.0)), current.a);
}
