#extension GL_EXT_buffer_reference : require

layout(location = 0) in vec2 uv;
layout(location = 0) out vec2 out_FragVelocity;

const float kClearDepth = 0.999999;

layout(push_constant) uniform DepthMotionVectorPushConstants {
  uint depthTexId;
  uint pointSamplerId;
  uint currentJitterUvXBits;
  uint currentJitterUvYBits;
  mat4 previousFromCurrentJitteredClip;
}
pc;

float pushFloat(uint bits) { return uintBitsToFloat(bits); }

vec2 clipNdcToTaaScreenUv(vec2 ndc) {
  return vec2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
}

vec2 currentJitterUv() {
  return vec2(pushFloat(pc.currentJitterUvXBits),
              pushFloat(pc.currentJitterUvYBits));
}

void main() {
  const float depth = textureBindless2D(pc.depthTexId, pc.pointSamplerId, uv).r;
  if (depth >= kClearDepth) {
    out_FragVelocity = vec2(0.0);
    return;
  }

  const vec2 currentNdc = vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  const vec4 currentClip = vec4(currentNdc, depth, 1.0);
  const vec4 previousClip = pc.previousFromCurrentJitteredClip * currentClip;
  if (previousClip.w == 0.0) {
    out_FragVelocity = vec2(0.0);
    return;
  }

  const vec2 previousUv =
      clipNdcToTaaScreenUv(previousClip.xy / previousClip.w);
  const vec2 currentNoJitterUv = uv - currentJitterUv();
  out_FragVelocity = previousUv - currentNoJitterUv;
}
