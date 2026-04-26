#extension GL_EXT_buffer_reference : require

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 out_FragColor;

const uint kVelocityDebugModeMotionVectors = 0u;
const uint kVelocityDebugModeMagnitude = 1u;

layout(push_constant) uniform VelocityDebugPushConstants {
  uint sourceTexId;
  uint sourceSamplerId;
  uint mode;
  uint scaleBits;
}
pc;

vec3 heatmap(float value) {
  float t = clamp(value, 0.0, 1.0);
  return clamp(vec3(1.5) - abs(vec3(4.0, 4.0, 4.0) * t - vec3(3.0, 2.0, 1.0)),
               vec3(0.0), vec3(1.0));
}

void main() {
  const vec2 velocity =
      textureBindless2D(pc.sourceTexId, pc.sourceSamplerId, uv).rg;
  const float scale = uintBitsToFloat(pc.scaleBits);
  const float magnitude = length(velocity) * scale;

  if (pc.mode == kVelocityDebugModeMagnitude) {
    out_FragColor = vec4(heatmap(magnitude), 1.0);
    return;
  }

  const vec2 signedVelocity =
      clamp(velocity * scale * 0.5 + vec2(0.5), vec2(0.0), vec2(1.0));
  out_FragColor =
      vec4(signedVelocity.x, signedVelocity.y, clamp(magnitude, 0.0, 1.0), 1.0);
}
