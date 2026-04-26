#extension GL_EXT_buffer_reference : require

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 out_FragColor;

const uint kTaaResolveFlagHistoryValid = 1u << 0u;
const uint kTaaResolveModeResolve = 0u;
const uint kTaaResolveModeCopyCurrent = 1u;
const uint kTaaResolveModePreviousHistory = 2u;
const uint kTaaResolveModeHistoryValidity = 3u;

layout(push_constant) uniform TAAResolvePushConstants {
  uint currentTexId;
  uint historyTexId;
  uint depthTexId;
  uint velocityTexId;
  uint currentSamplerId;
  uint historySamplerId;
  uint depthSamplerId;
  uint velocitySamplerId;
  uint flags;
  uint mode;
  uint currentWeightBits;
  uint reserved0;
}
pc;

bool uvInBounds(vec2 value) {
  return all(greaterThanEqual(value, vec2(0.0))) &&
         all(lessThanEqual(value, vec2(1.0)));
}

bool historyAvailable(vec2 historyUv, float depth) {
  if ((pc.flags & kTaaResolveFlagHistoryValid) == 0u) {
    return false;
  }
  if (!uvInBounds(historyUv)) {
    return false;
  }
  // Clear-depth/background pixels use current color until skybox/background
  // reprojection has an explicit policy.
  return depth < 0.999999;
}

vec4 currentColor(vec2 sampleUv) {
  return textureBindless2D(pc.currentTexId, pc.currentSamplerId, sampleUv);
}

vec4 historyColor(vec2 sampleUv) {
  return textureBindless2D(pc.historyTexId, pc.historySamplerId, sampleUv);
}

void main() {
  if (pc.mode == kTaaResolveModeCopyCurrent) {
    out_FragColor = currentColor(uv);
    return;
  }
  if (pc.mode == kTaaResolveModePreviousHistory) {
    out_FragColor = historyColor(uv);
    return;
  }

  const float depth = textureBindless2D(pc.depthTexId, pc.depthSamplerId, uv).r;
  const vec2 velocity =
      textureBindless2D(pc.velocityTexId, pc.velocitySamplerId, uv).rg;
  const vec2 historyUv = uv + velocity;
  const bool validHistory = historyAvailable(historyUv, depth);

  if (pc.mode == kTaaResolveModeHistoryValidity) {
    out_FragColor =
        validHistory ? vec4(0.0, 1.0, 0.0, 1.0) : vec4(1.0, 0.0, 0.0, 1.0);
    return;
  }

  const vec4 current = currentColor(uv);
  if (!validHistory) {
    out_FragColor = current;
    return;
  }

  const vec4 history = historyColor(historyUv);
  const float currentWeight =
      clamp(uintBitsToFloat(pc.currentWeightBits), 0.0, 1.0);
  out_FragColor = mix(history, current, currentWeight);
}
