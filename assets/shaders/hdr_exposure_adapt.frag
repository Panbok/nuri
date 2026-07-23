#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 out_FragColor;

layout(std430, buffer_reference) buffer HDRExposureTelemetryBuffer {
  vec4 values;
  uvec4 metadata;
};

layout(push_constant) uniform HDRExposurePushConstants {
  uint sourceTexId;
  uint previousExposureTexId;
  uint sourceSamplerId;
  uint flags;
  uint64_t telemetryAddress;
  float targetGray;
  float brightenSpeed;
  float darkenSpeed;
  float minEv;
  float maxEv;
  float deltaSeconds;
  float maxEvChange;
  float lowPercentile;
  float highPercentile;
  float minLogLuminance;
  float maxLogLuminance;
  uint meteringMode;
  uint frameIndex;
}
pc;

const uint kHDRPostFlagExposureHistoryValid = 1u << 2u;
const uint kHDRExposureMeteringCenterWeighted = 1u;
const uint kHDRExposureTelemetrySchemaVersion = 2u;
const int kHistogramBinCount = 64;
const int kHistogramGridSize = 32;

float luminance(vec3 color) { return dot(color, vec3(0.2126, 0.7152, 0.0722)); }

float meterWeight(vec2 sampleUv) {
  if (pc.meteringMode != kHDRExposureMeteringCenterWeighted) {
    return 1.0;
  }
  vec2 centered = sampleUv * 2.0 - 1.0;
  float centerWeight = exp(-dot(centered, centered) * 1.8);
  return mix(0.25, 1.0, centerWeight);
}

float sampleHistogramLuminance(out float invalidFraction) {
  float bins[kHistogramBinCount];
  for (int bin = 0; bin < kHistogramBinCount; ++bin) {
    bins[bin] = 0.0;
  }

  float totalWeight = 0.0;
  float invalidSamples = 0.0;
  float logRange = max(pc.maxLogLuminance - pc.minLogLuminance, 1.0e-4);
  for (int y = 0; y < kHistogramGridSize; ++y) {
    for (int x = 0; x < kHistogramGridSize; ++x) {
      vec2 sampleUv = (vec2(x, y) + vec2(0.5)) / float(kHistogramGridSize);
      vec3 color =
          textureBindless2D(pc.sourceTexId, pc.sourceSamplerId, sampleUv).rgb;
      float sampleLuminance = luminance(color);
      if (isnan(sampleLuminance) || isinf(sampleLuminance) ||
          sampleLuminance < 0.0) {
        invalidSamples += 1.0;
        continue;
      }
      float logLuminance =
          clamp(log2(max(sampleLuminance, exp2(pc.minLogLuminance))),
                pc.minLogLuminance, pc.maxLogLuminance);
      int bin = clamp(int(floor((logLuminance - pc.minLogLuminance) / logRange *
                                float(kHistogramBinCount))),
                      0, kHistogramBinCount - 1);
      float weight = meterWeight(sampleUv);
      bins[bin] += weight;
      totalWeight += weight;
    }
  }

  invalidFraction =
      invalidSamples / float(kHistogramGridSize * kHistogramGridSize);
  if (totalWeight <= 0.0) {
    return max(pc.targetGray, exp2(pc.minLogLuminance));
  }

  float lowThreshold = clamp(pc.lowPercentile, 0.0, 1.0) * totalWeight;
  float highThreshold = clamp(pc.highPercentile, 0.0, 1.0) * totalWeight;
  float cumulative = 0.0;
  float clippedLogSum = 0.0;
  float clippedWeight = 0.0;
  for (int bin = 0; bin < kHistogramBinCount; ++bin) {
    float next = cumulative + bins[bin];
    float accepted =
        max(min(next, highThreshold) - max(cumulative, lowThreshold), 0.0);
    float binCenter = pc.minLogLuminance +
                      (float(bin) + 0.5) / float(kHistogramBinCount) * logRange;
    clippedLogSum += accepted * binCenter;
    clippedWeight += accepted;
    cumulative = next;
  }
  return exp2(clippedLogSum / max(clippedWeight, 1.0e-4));
}

void main() {
  float invalidFraction = 0.0;
  float currentLuminance = sampleHistogramLuminance(invalidFraction);
  float previousLuminance = max(pc.targetGray, 1.0e-4);
  if ((pc.flags & kHDRPostFlagExposureHistoryValid) != 0u) {
    const float sampledPrevious =
        textureBindless2D(pc.previousExposureTexId, pc.sourceSamplerId,
                          vec2(0.5, 0.5))
            .r;
    if (!isnan(sampledPrevious) && !isinf(sampledPrevious) &&
        sampledPrevious > 0.0) {
      previousLuminance = max(sampledPrevious, 1.0e-4);
    }
  }
  float targetEv =
      clamp(log2(max(pc.targetGray, 1.0e-4) / max(currentLuminance, 1.0e-4)),
            pc.minEv, pc.maxEv);
  float previousEv = clamp(log2(max(pc.targetGray, 1.0e-4) / previousLuminance),
                           pc.minEv, pc.maxEv);
  float speed = targetEv > previousEv ? pc.brightenSpeed : pc.darkenSpeed;
  float blend = 1.0 - exp(-max(speed, 0.0) * max(pc.deltaSeconds, 0.0));
  float evDelta = (targetEv - previousEv) * blend;
  float maxDelta = max(pc.maxEvChange, 0.0);
  float adaptedEv = clamp(previousEv + clamp(evDelta, -maxDelta, maxDelta),
                          pc.minEv, pc.maxEv);
  float adaptedLuminance = max(pc.targetGray, 1.0e-4) / exp2(adaptedEv);

  out_FragColor =
      vec4(adaptedLuminance, currentLuminance, adaptedEv, invalidFraction);
  if (pc.telemetryAddress != 0ul) {
    HDRExposureTelemetryBuffer telemetry =
        HDRExposureTelemetryBuffer(pc.telemetryAddress);
    telemetry.values =
        vec4(adaptedEv, targetEv, currentLuminance, invalidFraction);
    telemetry.metadata =
        uvec4(pc.frameIndex, kHDRExposureTelemetrySchemaVersion, 0u, 0u);
  }
}
