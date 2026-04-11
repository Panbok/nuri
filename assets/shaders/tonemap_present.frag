layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 out_FragColor;

layout(push_constant) uniform PresentPushConstants {
  uint sourceTexId;
  uint sourceSamplerId;
  float exposure;
  uint reserved0;
}
pcPresent;

// IEC 61966-2-1 sRGB piecewise OETF for swapchain presentation.
vec3 linearToSrgb(vec3 c) {
  const bvec3 useLinear = lessThanEqual(c, vec3(0.0031308));
  const vec3 linear = c * 12.92;
  const vec3 nonlinear =
      1.055 * pow(max(c, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
  return mix(nonlinear, linear, useLinear);
}

vec3 acesFitted(vec3 x) {
  const float a = 2.51;
  const float b = 0.03;
  const float c = 2.43;
  const float d = 0.59;
  const float e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
  vec4 hdr =
      textureBindless2D(pcPresent.sourceTexId, pcPresent.sourceSamplerId, uv);
  vec3 mapped = acesFitted(max(hdr.rgb * pcPresent.exposure, vec3(0.0)));
  out_FragColor = vec4(linearToSrgb(mapped), hdr.a);
}
