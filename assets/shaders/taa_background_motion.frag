layout(location = 0) in vec2 uv;
layout(location = 0) out vec2 out_FragVelocity;

layout(push_constant) uniform BackgroundMotionPushConstants {
  mat4 previousFromCurrentJitteredRotationClip;
  uint currentJitterUvXBits;
  uint currentJitterUvYBits;
  uint historyValid;
}
pc;

vec2 clipNdcToTaaScreenUv(vec2 ndc) {
  return vec2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
}

void main() {
  if (pc.historyValid == 0u) {
    out_FragVelocity = vec2(0.0);
    return;
  }

  // fullscreen_copy-style UVs use Y in [1, 2]. Normalize to the canonical
  // TAA screen convention before applying previousUv = currentUv + motionUv.
  const vec2 screenUv = vec2(uv.x, uv.y - 1.0);
  const vec2 currentNdc = vec2(screenUv.x * 2.0 - 1.0, 1.0 - screenUv.y * 2.0);
  const vec4 previousClip =
      pc.previousFromCurrentJitteredRotationClip * vec4(currentNdc, 1.0, 1.0);
  if (previousClip.w == 0.0) {
    out_FragVelocity = vec2(0.0);
    return;
  }

  const vec2 previousUv =
      clipNdcToTaaScreenUv(previousClip.xy / previousClip.w);
  const vec2 currentJitterUv = vec2(uintBitsToFloat(pc.currentJitterUvXBits),
                                    uintBitsToFloat(pc.currentJitterUvYBits));
  const vec2 currentNoJitterUv = screenUv - currentJitterUv;
  out_FragVelocity = previousUv - currentNoJitterUv;
}
