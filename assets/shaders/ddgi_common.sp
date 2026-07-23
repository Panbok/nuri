const float kDDGIIrradianceGamma = 5.0;
const float kDDGIVisibilityFloor = 0.05;
const float kDDGIChebyshevExponent = 3.0;
const float kDDGIProbeWeightCrushThreshold = 0.2;
const float kDDGIIrradianceChangeThreshold = 0.25;
const float kDDGIIrradianceResetThreshold = 0.80;
const float kDDGIIrradianceHysteresisReduction = 0.15;
struct DDGIVolumeSample {
  vec3 irradiance;
  float confidence;
  float coverage;
  float visibility;
  float distanceMean;
  float distanceVariance;
  float classification;
  float relocation;
  float updateAge;
  float leakRisk;
};

struct DDGIVolumeCandidate {
  float coverage;
  float historyConfidence;
  float confidence;
};

struct DDGIQueryResult {
  vec3 irradiance;
  float historyConfidence;
  float skyWeight;
  float visibility;
  uint firstVolume;
  uint secondVolume;
  float firstWeight;
  float secondWeight;
  float distanceMean;
  float distanceVariance;
  float classification;
  float relocation;
  float updateAge;
  float leakRisk;
};

vec2 ddgiOctEncode(vec3 direction) {
  direction /= max(abs(direction.x) + abs(direction.y) + abs(direction.z),
                   1.0e-6);
  vec2 encoded = direction.xy;
  if (direction.z < 0.0) {
    encoded = (vec2(1.0) - abs(encoded.yx)) *
              vec2(encoded.x >= 0.0 ? 1.0 : -1.0,
                   encoded.y >= 0.0 ? 1.0 : -1.0);
  }
  return encoded;
}

vec3 ddgiOctDecode(vec2 encoded) {
  vec3 direction = vec3(encoded, 1.0 - abs(encoded.x) - abs(encoded.y));
  if (direction.z < 0.0) {
    direction.xy = (vec2(1.0) - abs(direction.yx)) *
                   vec2(direction.x >= 0.0 ? 1.0 : -1.0,
                        direction.y >= 0.0 ? 1.0 : -1.0);
  }
  return normalize(direction);
}

uvec2 ddgiBorderCopyCoordinate(uvec2 tilePixel, uint interiorExtent) {
  const uint tileExtent = interiorExtent + 2u;
  const bool corner =
      (tilePixel.x == 0u || tilePixel.x == tileExtent - 1u) &&
      (tilePixel.y == 0u || tilePixel.y == tileExtent - 1u);
  const bool row = tilePixel.x > 0u && tilePixel.x < tileExtent - 1u;
  if (corner) {
    return uvec2(tilePixel.x > 0u ? 1u : interiorExtent,
                 tilePixel.y > 0u ? 1u : interiorExtent);
  }
  if (row) {
    return uvec2(tileExtent - 1u - tilePixel.x,
                 tilePixel.y > 0u ? tilePixel.y - 1u : tilePixel.y + 1u);
  }
  return uvec2(tilePixel.x > 0u ? tilePixel.x - 1u : tilePixel.x + 1u,
               tileExtent - 1u - tilePixel.y);
}

vec3 ddgiRayDirection(uint ray, uint count, uint submittedSequence) {
  const float z = 1.0 - 2.0 * (float(ray) + 0.5) /
                               float(max(count, 1u));
  const float radius = sqrt(max(1.0 - z * z, 0.0));
  const float angle = 2.39996322972865332 * float(ray) +
                      0.7548776662466927 * float(submittedSequence & 1023u);
  return vec3(cos(angle) * radius, sin(angle) * radius, z);
}

uint ddgiProbeIndex(uvec3 coordinate, uvec3 counts) {
  return coordinate.x + counts.x * (coordinate.y + counts.y * coordinate.z);
}

uvec3 ddgiPhysicalCoordinate(uvec3 logical, uvec3 origin, uvec3 counts) {
  return (logical + origin) % counts;
}

vec3 ddgiNominalLocalPosition(DDGIVolumeGpuData volume,
                              uvec3 coordinate) {
  const ivec3 cameraCell = ivec3(volume.generations.x,
                                 volume.generations.y,
                                 volume.generations.w);
  return (vec3(coordinate) - 0.5 * vec3(volume.probeCountsAndCount.xyz - 1u)) *
             volume.probeSpacingAndBias.xyz +
         vec3(cameraCell) * volume.probeSpacingAndBias.xyz;
}

bool ddgiProbeShades(uint state) {
  return state == kDDGIProbeStateAwake || state == kDDGIProbeStateVigilant;
}

float ddgiProbeRayTMin(vec3 probeSpacing) {
  return max(1.0e-4 * min(min(probeSpacing.x, probeSpacing.y), probeSpacing.z),
             1.0e-4);
}

vec2 ddgiAtlasUv(uvec4 atlas, uint probeIndex, vec3 direction,
                 float interiorExtent) {
  const uint columns = atlas.z;
  const uint rows = atlas.w;
  const float tileExtent = interiorExtent + 2.0;
  const uvec2 tile = uvec2(probeIndex % columns, probeIndex / columns);
  const vec2 octUv = ddgiOctEncode(normalize(direction)) * 0.5 + 0.5;
  const vec2 texel = vec2(tile) * tileExtent + vec2(1.0) +
                     octUv * max(interiorExtent - 1.0, 1.0) + vec2(0.5);
  return texel / (vec2(columns, rows) * tileExtent);
}

float ddgiStaticVolumeCoverage(DDGIVolumeGpuData volume, vec3 localPoint) {
  const vec3 halfExtents = volume.centerHalfExtentsAndMaxDistance.xyz;
  const float blend = uintBitsToFloat(volume.ringOriginAndFlags.w);
  const vec3 distanceToFace = halfExtents - abs(localPoint);
  if (any(lessThan(distanceToFace, vec3(0.0)))) {
    return 0.0;
  }
  if (blend <= 1.0e-6) {
    return 1.0;
  }
  return clamp(min(min(distanceToFace.x, distanceToFace.y), distanceToFace.z) /
                   blend,
               0.0, 1.0);
}

float ddgiVolumeCoverage(DDGIVolumeGpuData volume, vec3 localPoint,
                         vec3 trackedCenter) {
  if (volume.effectiveIdentity.x != kDDGIEffectiveKindClipmapCascade) {
    return ddgiStaticVolumeCoverage(volume, localPoint - trackedCenter);
  }

  const vec3 distanceFromCamera =
      abs(localPoint - volume.continuousCameraLocal.xyz);
  const vec3 fadeStart = volume.fadeStartHalfExtents.xyz;
  const vec3 fadeEnd = volume.fadeEndHalfExtents.xyz;
  const vec3 fadeWidth = max(fadeEnd - fadeStart, vec3(1.0e-6));
  const vec3 axisCoverage =
      vec3(1.0) - clamp((distanceFromCamera - fadeStart) / fadeWidth,
                        vec3(0.0), vec3(1.0));
  return min(min(axisCoverage.x, axisCoverage.y), axisCoverage.z);
}

DDGIVolumeCandidate ddgiEvaluateVolumeCandidate(
    DDGIVolumeGpuData volume, vec3 worldPoint, vec3 surfaceNormal,
    vec3 viewDirection) {
  DDGIVolumeCandidate result;
  result.coverage = 0.0;
  result.historyConfidence = 0.0;
  result.confidence = 0.0;
  if ((volume.resourceFlagsReserved.x & 1u) == 0u ||
      volume.irradianceAtlas.x == 0xffffffffu ||
      volume.distanceAtlas.x == 0xffffffffu) {
    return result;
  }

  const vec3 localPoint =
      (volume.localFromWorld * vec4(worldPoint, 1.0)).xyz;
  const ivec3 cameraCell = ivec3(volume.generations.x,
                                 volume.generations.y,
                                 volume.generations.w);
  const vec3 trackedCenter =
      vec3(cameraCell) * volume.probeSpacingAndBias.xyz;
  result.coverage = ddgiVolumeCoverage(volume, localPoint, trackedCenter);
  if (result.coverage <= 0.0) {
    return result;
  }

  const float minSpacing = min(min(volume.probeSpacingAndBias.x,
                                   volume.probeSpacingAndBias.y),
                               volume.probeSpacingAndBias.z);
  const vec3 safeNormal = normalize(surfaceNormal);
  const vec3 safeView = normalize(viewDirection);
  const vec3 biasedWorldPoint =
      worldPoint + (0.2 * safeNormal + 0.8 * safeView) *
                       (0.75 * minSpacing) * volume.probeSpacingAndBias.w;
  const vec3 biasedLocalPoint =
      (volume.localFromWorld * vec4(biasedWorldPoint, 1.0)).xyz;
  const uvec3 counts = volume.probeCountsAndCount.xyz;
  const vec3 grid = clamp((biasedLocalPoint - trackedCenter) /
                                  volume.probeSpacingAndBias.xyz +
                              0.5 * vec3(counts - 1u),
                          vec3(0.0), vec3(counts - 1u));
  const uvec3 base = uvec3(min(floor(grid), vec3(counts - 2u)));
  const vec3 fraction = grid - vec3(base);
  for (uint neighbor = 0u; neighbor < 8u; ++neighbor) {
    const uvec3 offset = uvec3(neighbor & 1u, (neighbor >> 1u) & 1u,
                               (neighbor >> 2u) & 1u);
    const vec3 trilinearAxis =
        mix(vec3(1.0) - fraction, fraction, vec3(offset));
    const float trilinear =
        trilinearAxis.x * trilinearAxis.y * trilinearAxis.z;
    const uvec3 physical = ddgiPhysicalCoordinate(
        base + offset, volume.ringOriginAndFlags.xyz, counts);
    const uint probe = ddgiProbeIndex(physical, counts);
    if (ddgiProbeShades(
            volume.probeStates.values[probe].stateAgeFlags.x)) {
      result.historyConfidence += trilinear;
    }
  }
  result.historyConfidence = clamp(result.historyConfidence, 0.0, 1.0);
  result.confidence =
      clamp(result.coverage * result.historyConfidence, 0.0, 1.0);
  return result;
}

DDGIVolumeSample ddgiSampleVolume(DDGIVolumeGpuData volume,
                                  vec3 worldPoint, vec3 surfaceNormal,
                                  vec3 viewDirection,
                                  uint samplerId,
                                  DDGIVolumeCandidate candidate) {
  DDGIVolumeSample result;
  result.irradiance = vec3(0.0);
  result.confidence = 0.0;
  result.coverage = 0.0;
  result.visibility = 0.0;
  result.distanceMean = 0.0;
  result.distanceVariance = 0.0;
  result.classification = 0.0;
  result.relocation = 0.0;
  result.updateAge = 0.0;
  result.leakRisk = 0.0;
  const ivec3 cameraCell = ivec3(volume.generations.x,
                                 volume.generations.y,
                                 volume.generations.w);
  const vec3 trackedCenter = vec3(cameraCell) *
                             volume.probeSpacingAndBias.xyz;
  result.coverage = candidate.coverage;
  result.confidence = candidate.historyConfidence;
  if (candidate.confidence <= 0.0) {
    return result;
  }
  const float minSpacing = min(min(volume.probeSpacingAndBias.x,
                                   volume.probeSpacingAndBias.y),
                               volume.probeSpacingAndBias.z);
  const vec3 safeNormal = normalize(surfaceNormal);
  const vec3 safeView = normalize(viewDirection);
  const vec3 biasedWorldPoint = worldPoint +
      (0.2 * safeNormal + 0.8 * safeView) *
          (0.75 * minSpacing) * volume.probeSpacingAndBias.w;
  const vec3 biasedLocalPoint =
      (volume.localFromWorld * vec4(biasedWorldPoint, 1.0)).xyz;
  const uvec3 counts = volume.probeCountsAndCount.xyz;
  const vec3 grid = clamp((biasedLocalPoint - trackedCenter) /
                                  volume.probeSpacingAndBias.xyz +
                              0.5 * vec3(counts - 1u),
                          vec3(0.0), vec3(counts - 1u));
  const uvec3 base = uvec3(min(floor(grid), vec3(counts - 2u)));
  const vec3 fraction = grid - vec3(base);
  vec3 encodedAccumulation = vec3(0.0);
  float weightSum = 0.0;
  float confidence = 0.0;
  float visibilitySum = 0.0;
  float distanceMeanSum = 0.0;
  float distanceVarianceSum = 0.0;
  float classificationSum = 0.0;
  float relocationSum = 0.0;
  float updateAgeSum = 0.0;
  float leakRiskSum = 0.0;
  for (uint neighbor = 0u; neighbor < 8u; ++neighbor) {
    const uvec3 offset = uvec3(neighbor & 1u, (neighbor >> 1u) & 1u,
                               (neighbor >> 2u) & 1u);
    const uvec3 logical = base + offset;
    const vec3 trilinearAxis = mix(vec3(1.0) - fraction, fraction, vec3(offset));
    const float trilinear = trilinearAxis.x * trilinearAxis.y * trilinearAxis.z;
    const uvec3 physical = ddgiPhysicalCoordinate(
        logical, volume.ringOriginAndFlags.xyz, counts);
    const uint probe = ddgiProbeIndex(physical, counts);
    const DDGIProbeStateGpuData probeState = volume.probeStates.values[probe];
    if (!ddgiProbeShades(probeState.stateAgeFlags.x)) {
      continue;
    }
    confidence += trilinear;
    const vec3 nominal = ddgiNominalLocalPosition(volume, logical);
    const vec3 localProbe = nominal + probeState.relocation.xyz;
    const vec3 worldProbe =
        (volume.worldFromLocal * vec4(localProbe, 1.0)).xyz;
    const vec3 pointToProbe = worldProbe - biasedWorldPoint;
    const float receiverDistance = length(pointToProbe);
    const vec3 directionToProbe =
        receiverDistance > 1.0e-6 ? pointToProbe / receiverDistance
                                  : surfaceNormal;
    float weight = trilinear *
                   (pow(0.5 * (dot(directionToProbe, surfaceNormal) + 1.0),
                        2.0) +
                    0.2);
    const vec2 distanceUv = ddgiAtlasUv(volume.distanceAtlas, probe,
                                        -directionToProbe, 16.0);
    const vec2 moments = textureBindless2DLod(
        volume.distanceAtlas.x, samplerId, distanceUv, 0.0).rg;
    const float normalizedReceiver = receiverDistance /
        max(volume.centerHalfExtentsAndMaxDistance.w, 1.0e-6);
    float visibility = 1.0;
    if (normalizedReceiver > moments.x) {
      const float variance = max(moments.y - moments.x * moments.x, 0.0);
      const float delta = normalizedReceiver - moments.x;
      const float chebyshev = variance / max(variance + delta * delta, 1.0e-8);
      visibility = max(kDDGIVisibilityFloor,
                       pow(clamp(chebyshev, 0.0, 1.0),
                           kDDGIChebyshevExponent));
      weight *= visibility;
    }
    if (weight < kDDGIProbeWeightCrushThreshold) {
      weight *= (weight * weight) /
                (kDDGIProbeWeightCrushThreshold *
                 kDDGIProbeWeightCrushThreshold);
    }
    const vec2 irradianceUv = ddgiAtlasUv(volume.irradianceAtlas, probe,
                                          surfaceNormal, 8.0);
    const vec3 encoded = textureBindless2DLod(
        volume.irradianceAtlas.x, samplerId, irradianceUv, 0.0).rgb;
    encodedAccumulation +=
        pow(max(encoded, vec3(0.0)), vec3(kDDGIIrradianceGamma * 0.5)) *
        weight;
    visibilitySum += visibility * weight;
    distanceMeanSum += moments.x * weight;
    distanceVarianceSum +=
        max(moments.y - moments.x * moments.x, 0.0) * weight;
    classificationSum +=
        float(probeState.stateAgeFlags.x) / float(kDDGIProbeStateVigilant) *
        weight;
    relocationSum += clamp(length(probeState.relocation.xyz) /
                               max(0.5 * minSpacing, 1.0e-6),
                           0.0, 1.0) *
                     weight;
    const uint updateAge = volume.generations.z - probeState.stateAgeFlags.y;
    updateAgeSum += min(float(updateAge) / 60.0, 1.0) * weight;
    leakRiskSum += (1.0 - visibility) * weight;
    weightSum += weight;
  }
  if (weightSum > 1.0e-6) {
    const vec3 filtered = encodedAccumulation / weightSum;
    result.irradiance = filtered * filtered;
    result.visibility = visibilitySum / weightSum;
    result.distanceMean = distanceMeanSum / weightSum;
    result.distanceVariance = distanceVarianceSum / weightSum;
    result.classification = classificationSum / weightSum;
    result.relocation = relocationSum / weightSum;
    result.updateAge = updateAgeSum / weightSum;
    result.leakRisk = leakRiskSum / weightSum;
  }
  result.confidence = clamp(confidence, 0.0, 1.0);
  return result;
}

DDGIQueryResult queryDDGI(DDGIFrameBuffer frame, vec3 worldPoint,
                          vec3 surfaceNormal, vec3 viewDirection) {
  DDGIQueryResult result;
  result.irradiance = vec3(0.0);
  result.historyConfidence = 0.0;
  result.skyWeight = 1.0;
  result.visibility = 0.0;
  result.firstVolume = 0xffffffffu;
  result.secondVolume = 0xffffffffu;
  result.firstWeight = 0.0;
  result.secondWeight = 0.0;
  result.distanceMean = 0.0;
  result.distanceVariance = 0.0;
  result.classification = 0.0;
  result.relocation = 0.0;
  result.updateAge = 0.0;
  result.leakRisk = 0.0;
  if (frame.activeCountDebugFlagsSampler.z != kDDGIFrameGpuDataVersion) {
    return result;
  }

  float remaining = 1.0;
  const uint count = min(frame.activeCountDebugFlagsSampler.x,
                         kDDGIMaxVolumes);
  const uint samplerId = frame.activeCountDebugFlagsSampler.w;
  uint first = 0xffffffffu;
  uint second = 0xffffffffu;
  DDGIVolumeCandidate firstCandidate;
  firstCandidate.coverage = 0.0;
  firstCandidate.historyConfidence = 0.0;
  firstCandidate.confidence = 0.0;
  DDGIVolumeCandidate secondCandidate = firstCandidate;
  for (uint volumeIndex = 0u; volumeIndex < count; ++volumeIndex) {
    const DDGIVolumeCandidate candidate = ddgiEvaluateVolumeCandidate(
        frame.volumes[volumeIndex], worldPoint, surfaceNormal, viewDirection);
    if (candidate.confidence <= 0.0) {
      continue;
    }
    if (first == 0xffffffffu) {
      first = volumeIndex;
      firstCandidate = candidate;
      if (candidate.confidence >= 1.0) {
        break;
      }
    } else if (candidate.confidence > secondCandidate.confidence) {
      second = volumeIndex;
      secondCandidate = candidate;
    }
  }

  const uint selectedVolumes[kDDGIMaxSampledVolumes] = uint[2](first, second);
  const DDGIVolumeCandidate selectedCandidates[kDDGIMaxSampledVolumes] =
      DDGIVolumeCandidate[2](firstCandidate, secondCandidate);
  for (uint selected = 0u; selected < kDDGIMaxSampledVolumes; ++selected) {
    const uint volumeIndex = selectedVolumes[selected];
    if (volumeIndex == 0xffffffffu || remaining <= 0.0) {
      continue;
    }
    const DDGIVolumeSample volumeSample = ddgiSampleVolume(
        frame.volumes[volumeIndex], worldPoint, surfaceNormal, viewDirection,
        samplerId, selectedCandidates[selected]);
    const float candidateCoverage = clamp(
        volumeSample.coverage * volumeSample.confidence, 0.0, 1.0);
    const float blendWeight = remaining * candidateCoverage;
    result.irradiance += volumeSample.irradiance * blendWeight;
    result.historyConfidence += volumeSample.confidence * blendWeight;
    result.visibility += volumeSample.visibility * blendWeight;
    result.distanceMean += volumeSample.distanceMean * blendWeight;
    result.distanceVariance += volumeSample.distanceVariance * blendWeight;
    result.classification += volumeSample.classification * blendWeight;
    result.relocation += volumeSample.relocation * blendWeight;
    result.updateAge += volumeSample.updateAge * blendWeight;
    result.leakRisk += volumeSample.leakRisk * blendWeight;
    if (selected == 0u) {
      result.firstVolume = volumeIndex;
      result.firstWeight = blendWeight;
    } else {
      result.secondVolume = volumeIndex;
      result.secondWeight = blendWeight;
    }
    remaining *= 1.0 - candidateCoverage;
  }
  result.skyWeight = remaining;
  return result;
}
