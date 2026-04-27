#include "BRDF.sp"
#include "common.sp"
#include "material_inputs.sp"

#include "material_lighting.sp"

layout(location = 0) in PerVertex vtx;

layout(location = 0) out vec4 out_FragColor;

vec2 currentScreenUv();

vec3 transmissionModelScale() {
  return max(
      abs(vec3(pc.tessNearDistance, pc.tessFarDistance, pc.tessMinFactor)),
      vec3(1.0e-4));
}

float applyIorToRoughness(float roughness, float ior) {
  if (isIorCompatMode(ior)) {
    return roughness;
  }
  return roughness * clamp(ior * 2.0 - 2.0, 0.0, 1.0);
}

vec3 getVolumeTransmissionRay(vec3 n, vec3 v, float thickness, float ior,
                              vec3 modelScale) {
  vec3 refractionVector = refract(-v, n, 1.0 / max(ior, 1.0));
  if (dot(refractionVector, refractionVector) <= kEpsilon) {
    refractionVector = -v;
  }
  return refractionVector * thickness * modelScale;
}

vec3 applyVolumeAttenuation(vec3 radiance, float transmissionDistance,
                            vec3 attenuationColor, float attenuationDistance) {
  if (attenuationDistance == 0.0) {
    return radiance;
  }
  vec3 safeColor = max(attenuationColor, vec3(1.0e-4));
  vec3 attenuationCoefficient = -log(safeColor) / attenuationDistance;
  vec3 transmittance = exp(-attenuationCoefficient * transmissionDistance);
  return transmittance * radiance;
}

float transmissionFramebufferLod(float roughness, float ior) {
  return applyIorToRoughness(roughness, ior) * 2.0;
}

vec3 sampleTransmissionColor(vec2 uv, float roughness, float ior) {
  vec2 clampedUv = clamp(uv, vec2(0.0), vec2(1.0));
  vec3 level0 = textureBindless2D(getSceneColorPyramidTexId(pc.frameData, 0u),
                                  pc.frameData.sceneColorSamplerId, clampedUv)
                    .rgb;
  vec3 level1 = level0;
  vec3 level2 = level0;

  uint level1TexId = getSceneColorPyramidTexId(pc.frameData, 1u);
  if (level1TexId != kInvalidTextureBindlessIndex) {
    level1 = textureBindless2D(level1TexId, pc.frameData.sceneColorSamplerId,
                               clampedUv)
                 .rgb;
  }
  uint level2TexId = getSceneColorPyramidTexId(pc.frameData, 2u);
  if (level2TexId != kInvalidTextureBindlessIndex) {
    level2 = textureBindless2D(level2TexId, pc.frameData.sceneColorSamplerId,
                               clampedUv)
                 .rgb;
  } else {
    level2 = level1;
  }

  float lod = clamp(transmissionFramebufferLod(roughness, ior), 0.0, 2.0);
  if ((pc.frameData.flags & kFrameDataFlagTransmissionMipDebug) != 0u) {
    const vec3 fullRes = vec3(1.0, 0.1, 0.1);
    const vec3 halfRes = vec3(0.1, 1.0, 0.1);
    const vec3 quarterRes = vec3(0.1, 0.35, 1.0);
    if (lod <= 1.0) {
      return mix(fullRes, halfRes, lod);
    }
    return mix(halfRes, quarterRes, lod - 1.0);
  }
  if (lod <= 1.0) {
    return mix(level0, level1, lod);
  }
  return mix(level1, level2, lod - 1.0);
}

vec2 currentScreenUv() {
  ivec2 sceneSize = textureBindlessSize2D(pc.frameData.sceneColorTexId);
  vec2 safeSize = max(vec2(sceneSize), vec2(1.0));
  return clamp(gl_FragCoord.xy / safeSize, vec2(0.0), vec2(1.0));
}

vec2 resolveTransmissionUv(vec3 refractedRayExit) {
  vec4 clipPos =
      pc.frameData.proj * pc.frameData.view * vec4(refractedRayExit, 1.0);
  if (clipPos.w <= kEpsilon) {
    return currentScreenUv();
  }
  vec2 ndc = clipPos.xy / clipPos.w;
  vec2 uv = ndc * 0.5 + 0.5;
  uv.y = 1.0 - uv.y;
  return clamp(uv, vec2(0.0), vec2(1.0));
}

vec3 getIndirectTransmission(vec3 n, vec3 v, float roughness, vec3 baseColor,
                             vec3 f0, vec3 f90, vec3 worldPos, float ior,
                             float thickness, vec3 attenuationColor,
                             float attenuationDistance) {
  vec3 transmissionRay =
      getVolumeTransmissionRay(n, v, thickness, ior, transmissionModelScale());
  vec3 refractedRayExit = worldPos + transmissionRay;

  vec2 refractionCoords = resolveTransmissionUv(refractedRayExit);
  vec3 transmittedLight =
      sampleTransmissionColor(refractionCoords, roughness, ior);
  vec3 attenuatedColor =
      applyVolumeAttenuation(transmittedLight, length(transmissionRay),
                             attenuationColor, attenuationDistance);

  float ndotv = max(dot(n, v), 0.001);
  vec3 specularColor = vec3(0.0);
  if ((pc.frameData.flags & kFrameDataFlagHasBrdfLut) != 0u &&
      pc.frameData.brdfLutTexId != kInvalidTextureBindlessIndex) {
    // The transmission BRDF term follows the glTF reference path and samples
    // the LUT in perceptual-roughness space. Using the engine's generic
    // environment-space remap here can collapse smooth transmissive surfaces
    // toward zero contribution.
    vec2 brdfUv = clamp(vec2(ndotv, roughness), vec2(0.0), vec2(1.0));
    vec2 brdf = textureBindless2D(pc.frameData.brdfLutTexId, 0u, brdfUv).rg;
    specularColor = f0 * brdf.x + f90 * brdf.y;
  } else {
    specularColor = fresnelSchlick(ndotv, f0);
  }
  return (vec3(1.0) - specularColor) * attenuatedColor * baseColor;
}

vec3 getDirectTransmission(vec3 n, vec3 v, vec3 pointToLight,
                           float alphaRoughness, vec3 f0, vec3 f90,
                           vec3 baseColor, float ior) {
  // alphaRoughness = roughness² is passed through, scaled by the IOR factor.
  // distributionGGX and geometryOcclusion both consume alpha-space roughness²,
  // matching the convention used in accumulateSurfaceLightContribution.
  float transmissionRoughness =
      max(applyIorToRoughness(alphaRoughness, ior), kBrdfMinRoughness);

  float pointToLightLenSq = dot(pointToLight, pointToLight);
  if (pointToLightLenSq <= kEpsilon) {
    return vec3(0.0);
  }

  vec3 l = pointToLight * inversesqrt(pointToLightLenSq);
  vec3 lMirror = reflect(-l, n);
  float lMirrorLenSq = dot(lMirror, lMirror);
  if (lMirrorLenSq <= kEpsilon) {
    return vec3(0.0);
  }
  lMirror *= inversesqrt(lMirrorLenSq);

  vec3 halfVector = lMirror + v;
  float halfLenSq = dot(halfVector, halfVector);
  if (halfLenSq <= kEpsilon) {
    return vec3(0.0);
  }
  vec3 h = halfVector * inversesqrt(halfLenSq);

  float ndoth = clamp(dot(n, h), 0.0, 1.0);
  float ndotlMirror = clamp(dot(n, lMirror), 0.0, 1.0);
  float ndotv = clamp(dot(n, v), 0.0, 1.0);
  float vdoth = clamp(dot(v, h), 0.0, 1.0);
  if (ndotlMirror <= 0.0 || ndotv <= 0.0) {
    return vec3(0.0);
  }

  float d = distributionGGX(ndoth, transmissionRoughness);
  vec3 f = specularReflection(vdoth, f0, f90);
  float g = geometryOcclusion(ndotlMirror, ndotv, transmissionRoughness);
  return (vec3(1.0) - f) * baseColor * d * g;
}

vec3 transmissionSafeSpecular(vec3 n, vec3 v, vec3 l, vec3 f0, vec3 f90,
                              float roughness) {
  float ndotl = max(dot(n, l), 0.0);
  float ndotv = max(dot(n, v), 0.0);
  if (ndotl <= 0.0 || ndotv <= 0.0) {
    return vec3(0.0);
  }

  vec3 reflected = reflect(-l, n);
  float rv = max(dot(reflected, v), 0.0);
  float exponent = mix(48.0, 4.0, clamp(roughness, 0.0, 1.0));
  float lobe = pow(rv, exponent);
  vec3 fresnel = specularReflection(ndotv, f0, f90);
  return ndotl * fresnel * lobe * ((exponent + 2.0) * 0.125);
}

vec3 transmissionSafeDirectTransmission(vec3 n, vec3 v, vec3 pointToLight,
                                        vec3 baseColor, float roughness,
                                        float ior) {
  float pointToLightLenSq = dot(pointToLight, pointToLight);
  if (pointToLightLenSq <= kEpsilon) {
    return vec3(0.0);
  }

  vec3 lightDir = pointToLight * inversesqrt(pointToLightLenSq);
  vec3 refracted = refract(-v, n, 1.0 / max(ior, 1.0));
  float refractedLenSq = dot(refracted, refracted);
  if (refractedLenSq <= kEpsilon) {
    refracted = -v;
  } else {
    refracted *= inversesqrt(refractedLenSq);
  }

  float facing = max(dot(refracted, lightDir), 0.0);
  float exponent = mix(24.0, 3.0, clamp(roughness, 0.0, 1.0));
  return baseColor * pow(facing, exponent);
}

void accumulateTransmissionSurfaceLight(vec3 lightRadiance, vec3 l,
                                        ShadedMaterial sm,
                                        inout DirectLightingResult direct) {
  vec3 viewFresnel =
      specularReflection(max(dot(sm.nBase, sm.v), 0.0), sm.f0, sm.f90);
  float diffuseWeight = 1.0 - max3(viewFresnel);
  direct.directDiffuse += max(dot(sm.nBase, l), 0.0) * lightRadiance *
                          sm.diffuseColor * diffuseWeight * (1.0 / kPi);
  direct.directSpecular +=
      lightRadiance *
      transmissionSafeSpecular(sm.nBase, sm.v, l, sm.f0, sm.f90, sm.roughness);

  if (sm.hasClearcoat && sm.clearcoat > 0.0) {
    direct.clearcoatDirectLighting +=
        lightRadiance * sm.clearcoat *
        transmissionSafeSpecular(sm.nClearcoat, sm.v, l, sm.clearcoatF0,
                                 sm.clearcoatReflectance90,
                                 sm.clearcoatRoughness);
  }
}

DirectLightingResult evaluateTransmissionDirectLighting(ShadedMaterial sm,
                                                        vec3 worldPos) {
  DirectLightingResult direct;
  direct.directDiffuse = vec3(0.0);
  direct.directSpecular = vec3(0.0);
  direct.directSheen = vec3(0.0);
  direct.clearcoatDirectLighting = vec3(0.0);
  direct.shadowFactorDebug = 1.0;
  direct.shadowCascadeIndexDebug = 0.0;
  direct.shadowCascadeBlendDebug = 0.0;
  direct.shadowPcfFactorDebug = 1.0;
  direct.shadowReceiverDepthDebug = 0.0;
  direct.shadowMapDepthDebug = 0.0;
  direct.shadowPcssBlockerRatioDebug = 0.0;
  direct.shadowPcssAverageBlockerDepthDebug = 0.0;
  direct.shadowPcssFilterRadiusDebug = 0.0;

  for (uint i = 0u; i < pc.frameData.directionalLightCount; ++i) {
    DirectionalLightGpuData light =
        pc.frameData.directionalLightBuffer.lights[i];
    vec3 l = normalize(-directionalLightDirection(light));
    vec3 lightRadiance =
        directionalLightColor(light) * directionalLightIlluminance(light);
    accumulateTransmissionSurfaceLight(lightRadiance, l, sm, direct);
  }

  for (uint i = 0u; i < pc.frameData.localLightCount; ++i) {
    LocalLightGpuData light = pc.frameData.localLightBuffer.lights[i];
    vec3 ptl = localLightPosition(light) - worldPos;
    float dsq = dot(ptl, ptl);
    if (dsq <= kEpsilon) {
      continue;
    }
    vec3 l = ptl * inversesqrt(dsq);
    float att = punctualRangeAttenuation(dsq, localLightRange(light));
    if (localLightType(light) == kLocalLightTypeSpot) {
      att *= spotAngularAttenuation(localLightDirection(light), ptl,
                                    localLightInnerCos(light),
                                    localLightOuterCos(light));
    }
    if (att <= 0.0) {
      continue;
    }
    vec3 lightRadiance =
        localLightColor(light) * localLightIntensity(light) * att;
    accumulateTransmissionSurfaceLight(lightRadiance, l, sm, direct);
  }

  return direct;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

void main() {
  const MaterialData material = loadMaterialData(pc.materialIndex);
  const uint alphaMode = materialAlphaMode(material);
  const uint featureMask = materialFeatureMask(material);

  ShadedMaterial sm = evaluateMaterial(material, vtx);

  const float alphaCutoff = materialAlphaCutoff(material);
  if (alphaMode == kAlphaModeMask && sm.baseColor.a < alphaCutoff) {
    discard;
  }

  // Direct lighting ---------------------------------------------------
  // Keep transmission on the reduced direct-light model.
  DirectLightingResult direct =
      evaluateTransmissionDirectLighting(sm, vtx.worldPos);

  // Transmission-specific material fields ----------------------------
  const uint matSampler = pc.frameData.materialSamplerId;
  const uint transmissionTexId =
      getMaterialTextureIndex(material, kMaterialTextureSlotTransmission);
  const uint thicknessTexId =
      getMaterialTextureIndex(material, kMaterialTextureSlotThickness);
  const vec2 uvTransmission =
      transformedUv(material, vtx, kMaterialTextureSlotTransmission);
  const vec2 uvThickness =
      transformedUv(material, vtx, kMaterialTextureSlotThickness);

  float transmissionFactor =
      material.transmission.transmissionThicknessDistance.x;
  if (transmissionTexId != kInvalidTextureBindlessIndex) {
    transmissionFactor *=
        textureBindless2D(transmissionTexId, matSampler, uvTransmission).r;
  }
  transmissionFactor = saturate(transmissionFactor);

  float thickness =
      max(material.transmission.transmissionThicknessDistance.y, 0.0);
  if (thicknessTexId != kInvalidTextureBindlessIndex) {
    thickness *= textureBindless2D(thicknessTexId, matSampler, uvThickness).g;
  }
  thickness = max(thickness, 0.0);

  vec3 attenuationColor = clamp(
      material.transmission.attenuationColorReserved.rgb, vec3(0.0), vec3(1.0));
  float attenuationDistance =
      max(material.transmission.transmissionThicknessDistance.z, 0.0);

  // glTF: packed ior==0 means KHR_materials_ior was omitted (compat mode for
  // base-layer F0).  For transmission we still follow the extension default
  // IOR of 1.5.  If we gated transmission on !iorCompatMode, materials with
  // omitted IOR would keep transmissionFactor mixing (below) but never
  // accumulate indirectTransmission/directTransmission → diffuse terms go to
  // zero and glass reads black.
  const float transmissionIor = sm.iorCompatMode ? 1.5 : sm.ior;
  const bool hasTransmission = transmissionFactor > 0.0;

  vec3 transmissionRay = vec3(0.0);
  float transmissionRayLength = 0.0;
  if (hasTransmission) {
    transmissionRay = getVolumeTransmissionRay(
        sm.nBase, sm.v, thickness, transmissionIor, transmissionModelScale());
    transmissionRayLength = length(transmissionRay);
  }

  vec3 directTransmission = vec3(0.0);
  if (hasTransmission) {
    for (uint i = 0u; i < pc.frameData.directionalLightCount; ++i) {
      DirectionalLightGpuData light =
          pc.frameData.directionalLightBuffer.lights[i];
      vec3 l = normalize(-directionalLightDirection(light));
      vec3 lr =
          directionalLightColor(light) * directionalLightIlluminance(light);
      float sheenScale =
          sm.sheenWeight > 0.0
              ? computeSheenAlbedoScalingDirect(sm.sheenColor, sm.ndotv,
                                                max(dot(sm.nBase, l), 0.0),
                                                sm.sheenRoughness)
              : 1.0;
      vec3 tc = lr * transmissionSafeDirectTransmission(
                         sm.nBase, sm.v, l, sm.diffuseColor, sm.roughness,
                         transmissionIor);
      if ((featureMask & kMaterialFeatureVolume) != 0u) {
        tc = applyVolumeAttenuation(tc, transmissionRayLength, attenuationColor,
                                    attenuationDistance);
      }
      directTransmission += sheenScale * tc;
    }

    for (uint i = 0u; i < pc.frameData.localLightCount; ++i) {
      LocalLightGpuData light = pc.frameData.localLightBuffer.lights[i];
      vec3 ptl = localLightPosition(light) - vtx.worldPos;
      float dsq = dot(ptl, ptl);
      if (dsq <= kEpsilon) {
        continue;
      }
      vec3 l = ptl * inversesqrt(dsq);
      float att = punctualRangeAttenuation(dsq, localLightRange(light));
      if (localLightType(light) == kLocalLightTypeSpot) {
        att *= spotAngularAttenuation(localLightDirection(light), ptl,
                                      localLightInnerCos(light),
                                      localLightOuterCos(light));
      }
      if (att <= 0.0) {
        continue;
      }
      vec3 lr = localLightColor(light) * localLightIntensity(light) * att;
      float sheenScale =
          sm.sheenWeight > 0.0
              ? computeSheenAlbedoScalingDirect(sm.sheenColor, sm.ndotv,
                                                max(dot(sm.nBase, l), 0.0),
                                                sm.sheenRoughness)
              : 1.0;
      vec3 transmissionPtl = ptl - transmissionRay;
      vec3 tc = lr * transmissionSafeDirectTransmission(
                         sm.nBase, sm.v, transmissionPtl, sm.diffuseColor,
                         sm.roughness, transmissionIor);
      if ((featureMask & kMaterialFeatureVolume) != 0u) {
        tc = applyVolumeAttenuation(tc, transmissionRayLength, attenuationColor,
                                    attenuationDistance);
      }
      directTransmission += sheenScale * tc;
    }
  }

  // IBL ---------------------------------------------------------------
  IblResult ibl = evaluateIbl(sm);

  vec3 indirectTransmission = vec3(0.0);
  if (hasTransmission &&
      (pc.frameData.flags & kFrameDataFlagHasSceneColor) != 0u &&
      pc.frameData.sceneColorTexId != kInvalidTextureBindlessIndex) {
    indirectTransmission = getIndirectTransmission(
        sm.nBase, sm.v, sm.roughness, sm.diffuseColor, sm.f0, sm.f90,
        vtx.worldPos, transmissionIor, thickness, attenuationColor,
        attenuationDistance);
  }

  // Mix transmission into the diffuse terms ---------------------------
  vec3 indirectDiffuseTerm =
      mix(ibl.iblDiffuse, indirectTransmission, transmissionFactor);
  vec3 directDiffuseTerm =
      mix(direct.directDiffuse, directTransmission, transmissionFactor);

  // Composition -------------------------------------------------------
  vec3 indirectLighting =
      sm.clearcoatAttenuation *
          (ibl.iblSheen +
           ibl.indirectScale * (indirectDiffuseTerm + ibl.iblSpecular)) +
      ibl.clearcoatIblSpecular;
  if (ibl.hasIndirectLighting) {
    indirectLighting *= sm.ao;
  }

  vec3 directLighting =
      sm.clearcoatAttenuation *
          (direct.directSheen + directDiffuseTerm + direct.directSpecular) +
      direct.clearcoatDirectLighting;
  vec3 color =
      directLighting + indirectLighting + sm.clearcoatAttenuation * sm.emissive;
  color = max(color, vec3(0.0));

  float outAlpha = (alphaMode == kAlphaModeOpaque) ? 1.0 : sm.baseColor.a;
  out_FragColor = vec4(color, outAlpha);
}
