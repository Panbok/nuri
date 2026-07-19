#extension GL_EXT_buffer_reference : require

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 out_FragColor;

layout(push_constant) uniform HDRLuminanceReducePushConstants {
  uint sourceTexId;
  uint sourceSamplerId;
  uint mode;
  float texelSizeX;
  float texelSizeY;
}
pc;

float luminance(vec3 color) { return dot(color, vec3(0.2126, 0.7152, 0.0722)); }

vec2 screenUv() { return fract(uv); }

float sampleLogLuminance(vec2 sampleUv) {
  float value = 0.0;
  if (pc.mode == 0u) {
    vec3 color =
        max(textureBindless2D(pc.sourceTexId, pc.sourceSamplerId, sampleUv).rgb,
            vec3(0.0));
    value = luminance(color);
  } else {
    value = max(
        textureBindless2D(pc.sourceTexId, pc.sourceSamplerId, sampleUv).r, 0.0);
  }
  return log2(max(value, 1.0e-4));
}

void main() {
  vec2 sampleUv = screenUv();
  vec2 halfTexel = vec2(pc.texelSizeX, pc.texelSizeY) * 0.5;
  float logAverage =
      (sampleLogLuminance(sampleUv + halfTexel * vec2(-1.0, -1.0)) +
       sampleLogLuminance(sampleUv + halfTexel * vec2(1.0, -1.0)) +
       sampleLogLuminance(sampleUv + halfTexel * vec2(-1.0, 1.0)) +
       sampleLogLuminance(sampleUv + halfTexel * vec2(1.0, 1.0))) *
      0.25;
  float reducedLuminance = exp2(logAverage);
  out_FragColor =
      vec4(reducedLuminance, reducedLuminance, reducedLuminance, 1.0);
}
