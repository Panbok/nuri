#version 460
#extension GL_EXT_buffer_reference : require

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 out_FragColor;

layout(push_constant) uniform HDRLuminanceReducePushConstants {
  uint sourceTexId;
  uint sourceSamplerId;
  vec2 texelSize;
}
pc;

float luminance(vec3 color) { return dot(color, vec3(0.2126, 0.7152, 0.0722)); }

float sampleLogLuminance(vec2 sampleUv) {
  vec3 color =
      max(textureBindless2D(pc.sourceTexId, pc.sourceSamplerId, sampleUv).rgb,
          vec3(0.0));
  return log(max(luminance(color), 1.0e-4));
}

void main() {
  vec2 halfTexel = pc.texelSize * 0.5;
  float logAverage = (sampleLogLuminance(uv + halfTexel * vec2(-1.0, -1.0)) +
                      sampleLogLuminance(uv + halfTexel * vec2(1.0, -1.0)) +
                      sampleLogLuminance(uv + halfTexel * vec2(-1.0, 1.0)) +
                      sampleLogLuminance(uv + halfTexel * vec2(1.0, 1.0))) *
                     0.25;
  float reducedLuminance = exp(logAverage);
  out_FragColor =
      vec4(reducedLuminance, reducedLuminance, reducedLuminance, 1.0);
}
