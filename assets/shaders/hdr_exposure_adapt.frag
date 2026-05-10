#extension GL_EXT_buffer_reference : require

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 out_FragColor;

layout(push_constant) uniform HDRExposurePushConstants {
  uint sourceTexId;
  uint previousExposureTexId;
  uint sourceSamplerId;
  uint flags;
  float targetGray;
  float speed;
  float minEv;
  float maxEv;
  float deltaSeconds;
  float reserved0;
  float reserved1;
  float reserved2;
}
pc;

const uint kHDRPostFlagExposureHistoryValid = 1u << 2u;
const float kHDRAdaptationMeterMinLuminance = 0.03;
const float kHDRAdaptationMeterMaxLuminance = 64.0;

float luminance(vec3 color) { return dot(color, vec3(0.2126, 0.7152, 0.0722)); }

float meterWeight(vec2 sampleUv) {
  vec2 centered = sampleUv * 2.0 - 1.0;
  float centerWeight = exp(-dot(centered, centered) * 1.8);
  return mix(0.25, 1.0, centerWeight);
}

float sampleAverageLuminance() {
  const int gridSize = 32;
  float weightedLogSum = 0.0;
  float weightSum = 0.0;
  for (int y = 0; y < gridSize; ++y) {
    for (int x = 0; x < gridSize; ++x) {
      vec2 sampleUv = (vec2(x, y) + vec2(0.5)) / float(gridSize);
      vec3 color = max(
          textureBindless2D(pc.sourceTexId, pc.sourceSamplerId, sampleUv).rgb,
          vec3(0.0));
      float meteredLuminance =
          clamp(luminance(color), kHDRAdaptationMeterMinLuminance,
                kHDRAdaptationMeterMaxLuminance);
      float weight = meterWeight(sampleUv);
      weightedLogSum += log2(meteredLuminance) * weight;
      weightSum += weight;
    }
  }
  return exp2(weightedLogSum / max(weightSum, 1.0e-4));
}

void main() {
  float currentLuminance = sampleAverageLuminance();
  float previousLuminance = max(pc.targetGray, 1.0e-4);
  if ((pc.flags & kHDRPostFlagExposureHistoryValid) != 0u) {
    previousLuminance =
        max(textureBindless2D(pc.previousExposureTexId, pc.sourceSamplerId,
                              vec2(0.5, 0.5))
                .r,
            1.0e-4);
  }
  float currentLogLuminance = log2(currentLuminance);
  float previousLogLuminance = log2(previousLuminance);
  float blend = 1.0 - exp(-max(pc.speed, 0.0) * max(pc.deltaSeconds, 0.0));
  float adaptedLogLuminance =
      mix(previousLogLuminance, currentLogLuminance, blend);
  float adaptedLuminance =
      clamp(exp2(adaptedLogLuminance), kHDRAdaptationMeterMinLuminance,
            kHDRAdaptationMeterMaxLuminance);

  out_FragColor = vec4(adaptedLuminance, currentLuminance, 0.0, 1.0);
}
