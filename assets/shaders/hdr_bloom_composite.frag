#extension GL_EXT_buffer_reference : require

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 out_FragColor;

layout(push_constant) uniform HDRBloomCompositePushConstants {
  uint sourceTexId;
  uint bloomTexId;
  uint exposureTexId;
  uint sourceSamplerId;
  uint flags;
  uint debugView;
  uint reserved0;
  uint reserved1;
  float bloomStrength;
  float fallbackExposureEv;
  float adaptationTargetGray;
  float adaptationMinEv;
  float adaptationMaxEv;
}
pc;

const uint kHDRPostFlagBloomEnabled = 1u << 0u;
const uint kHDRPostFlagAdaptationEnabled = 1u << 1u;
const uint kHDRDebugBloomPrefilter = 1u;
const uint kHDRDebugBloomFinal = 2u;
const uint kHDRDebugLogAverageLuminance = 3u;
const uint kHDRDebugAdaptedExposure = 4u;
const uint kInvalidTextureBindlessIndex = 0xffffffffu;

float luminance(vec3 color) { return dot(color, vec3(0.2126, 0.7152, 0.0722)); }

vec2 screenUv() { return fract(uv); }

vec3 sampleTexture(uint texId, vec2 sampleUv) {
  return textureBindless2D(texId, pc.sourceSamplerId, sampleUv).rgb;
}

float exposureEv() {
  if ((pc.flags & kHDRPostFlagAdaptationEnabled) == 0u) {
    return pc.fallbackExposureEv;
  }
  float adaptedLuminance =
      max(textureBindless2D(pc.exposureTexId, pc.sourceSamplerId, vec2(0.5)).r,
          1.0e-4);
  float ev = log2(max(pc.adaptationTargetGray, 1.0e-4) / adaptedLuminance);
  return clamp(ev, pc.adaptationMinEv, pc.adaptationMaxEv);
}

void main() {
  vec2 sampleUv = screenUv();
  vec3 source = sampleTexture(pc.sourceTexId, sampleUv);
  bool hasBloom = pc.bloomTexId != kInvalidTextureBindlessIndex;
  vec3 bloom = hasBloom ? sampleTexture(pc.bloomTexId, sampleUv) : vec3(0.0);

  if (pc.debugView == kHDRDebugBloomPrefilter ||
      pc.debugView == kHDRDebugBloomFinal) {
    out_FragColor = vec4(bloom * pc.bloomStrength, 1.0);
    return;
  }

  float ev = exposureEv();
  vec3 color = source * exp2(ev);
  if ((pc.flags & kHDRPostFlagBloomEnabled) != 0u && hasBloom) {
    color += bloom * pc.bloomStrength * exp2(ev);
  }

  if (pc.debugView == kHDRDebugLogAverageLuminance) {
    float value = log2(max(luminance(source), 1.0e-4)) * 0.08 + 0.5;
    out_FragColor = vec4(vec3(value), 1.0);
    return;
  }
  if (pc.debugView == kHDRDebugAdaptedExposure) {
    out_FragColor = vec4(vec3(clamp(ev * 0.08 + 0.5, 0.0, 1.0)), 1.0);
    return;
  }

  out_FragColor = vec4(color, 1.0);
}
