#include "common.sp"
#include "BRDF.sp"
#include "material_inputs.sp"

layout(location = 0) in PerVertex vtx;

layout(location = 0) out vec4 out_FragColor;

const uint kAlphaModeOpaque = 0u;
const uint kAlphaModeMask = 1u;

void accumulateDirectLightContribution(
    vec3 lightRadiance, vec3 l, vec3 v, vec3 nBase, vec3 nClearcoat,
    vec3 diffuseColor, vec3 f0, vec3 f90, float ndotv, float clearcoatNdotV,
    float alphaRoughness, vec3 sheenColor, float sheenWeight,
    float sheenRoughness, bool hasClearcoat, float clearcoat,
    float clearcoatRoughness, vec3 clearcoatF0, vec3 clearcoatReflectance90,
    inout vec3 baseDirectLighting, inout vec3 directSheen,
    inout vec3 clearcoatDirectLighting) {
  float ndotl = max(dot(nBase, l), 0.0);
  float clearcoatNdotL = max(dot(nClearcoat, l), 0.0);
  vec3 halfVector = v + l;
  float halfLenSq = dot(halfVector, halfVector);
  if (ndotl > 0.0 && halfLenSq > kBrdfEpsilon) {
    vec3 h = halfVector * inversesqrt(halfLenSq);
    float ndoth = max(dot(nBase, h), 0.0);
    float ldoth = max(dot(l, h), 0.0);
    float vdoth = max(dot(v, h), 0.0);

    vec3 f = specularReflection(vdoth, f0, f90);
    float g = geometryOcclusion(ndotl, ndotv, alphaRoughness);
    float d = microfacetDistribution(ndoth, alphaRoughness);
    vec3 diffuse = (1.0 - max3(f)) *
                   diffuseBurley(diffuseColor, ndotl, ndotv, ldoth,
                                 alphaRoughness);
    vec3 specular = f * g * d / max(4.0 * ndotl * ndotv, kBrdfEpsilon);
    float sheenDirectScale =
        sheenWeight > 0.0
            ? computeSheenAlbedoScalingDirect(sheenColor, ndotv, ndotl,
                                              sheenRoughness)
            : 1.0;
    baseDirectLighting +=
        sheenDirectScale * (ndotl * lightRadiance * (diffuse + specular));
    directSheen += computeDirectSheen(sheenColor, sheenWeight, sheenRoughness,
                                      ndotl, ndotv, ndoth, lightRadiance);
  }
  if (hasClearcoat && clearcoat > 0.0 && clearcoatNdotL > 0.0 &&
      halfLenSq > kBrdfEpsilon) {
    vec3 h = halfVector * inversesqrt(halfLenSq);
    float clearcoatNdotH = max(dot(nClearcoat, h), 0.0);
    float clearcoatVdotH = max(dot(v, h), 0.0);
    float clearcoatAlphaRoughness = clearcoatRoughness * clearcoatRoughness;
    vec3 clearcoatF = specularReflection(clearcoatVdotH, clearcoatF0,
                                         clearcoatReflectance90);
    float clearcoatG =
        geometryOcclusion(clearcoatNdotL, clearcoatNdotV, clearcoatAlphaRoughness);
    float clearcoatD =
        microfacetDistribution(clearcoatNdotH, clearcoatAlphaRoughness);
    vec3 clearcoatSpecular =
        clearcoatF * clearcoatG * clearcoatD /
        max(4.0 * clearcoatNdotL * clearcoatNdotV, kBrdfEpsilon);
    clearcoatDirectLighting +=
        clearcoat * clearcoatNdotL * lightRadiance * clearcoatSpecular;
  }
}

void main() {
  const MaterialGpuData material = pc.materialBuffer.materials[pc.materialIndex];

  const uint baseColorTexId =
      GET_TEXTURE_INDEX(material, kMaterialTextureSlotBaseColor);
  const uint metallicRoughnessTexId =
      GET_TEXTURE_INDEX(material, kMaterialTextureSlotMetallicRoughness);
  const uint normalTexId =
      GET_TEXTURE_INDEX(material, kMaterialTextureSlotNormal);
  const uint occlusionTexId =
      GET_TEXTURE_INDEX(material, kMaterialTextureSlotOcclusion);
  const uint emissiveTexId =
      GET_TEXTURE_INDEX(material, kMaterialTextureSlotEmissive);
  const uint clearcoatTexId =
      GET_TEXTURE_INDEX(material, kMaterialTextureSlotClearcoat);
  const uint clearcoatRoughnessTexId =
      GET_TEXTURE_INDEX(material, kMaterialTextureSlotClearcoatRoughness);
  const uint clearcoatNormalTexId =
      GET_TEXTURE_INDEX(material, kMaterialTextureSlotClearcoatNormal);
  const uint specularTexId =
      GET_TEXTURE_INDEX(material, kMaterialTextureSlotSpecular);
  const uint specularColorTexId =
      GET_TEXTURE_INDEX(material, kMaterialTextureSlotSpecularColor);
  const uint sheenColorTexId =
      GET_TEXTURE_INDEX(material, kMaterialTextureSlotSheenColor);
  const uint sheenRoughnessTexId =
      GET_TEXTURE_INDEX(material, kMaterialTextureSlotSheenRoughness);
  const uint alphaMode = material.materialFlags.x;
  const uint featureMask = material.materialFlags.z;
  const uint workflow = material.materialFlags.w;

  const uint baseColorSampler =
      GET_SAMPLER_INDEX(material, kMaterialTextureSlotBaseColor);
  const uint metallicRoughnessSampler =
      GET_SAMPLER_INDEX(material, kMaterialTextureSlotMetallicRoughness);
  const uint normalSampler =
      GET_SAMPLER_INDEX(material, kMaterialTextureSlotNormal);
  const uint occlusionSampler =
      GET_SAMPLER_INDEX(material, kMaterialTextureSlotOcclusion);
  const uint emissiveSampler =
      GET_SAMPLER_INDEX(material, kMaterialTextureSlotEmissive);
  const uint clearcoatSampler =
      GET_SAMPLER_INDEX(material, kMaterialTextureSlotClearcoat);
  const uint clearcoatRoughnessSampler =
      GET_SAMPLER_INDEX(material, kMaterialTextureSlotClearcoatRoughness);
  const uint clearcoatNormalSampler =
      GET_SAMPLER_INDEX(material, kMaterialTextureSlotClearcoatNormal);
  const uint specularSampler =
      GET_SAMPLER_INDEX(material, kMaterialTextureSlotSpecular);
  const uint specularColorSampler =
      GET_SAMPLER_INDEX(material, kMaterialTextureSlotSpecularColor);
  const uint sheenColorSampler =
      GET_SAMPLER_INDEX(material, kMaterialTextureSlotSheenColor);
  const uint sheenRoughnessSampler =
      GET_SAMPLER_INDEX(material, kMaterialTextureSlotSheenRoughness);

  const vec2 uvBaseColor =
      transformedUv(material, vtx, kMaterialTextureSlotBaseColor);
  const vec2 uvMetallicRoughness =
      transformedUv(material, vtx, kMaterialTextureSlotMetallicRoughness);
  const vec2 uvNormal =
      transformedUv(material, vtx, kMaterialTextureSlotNormal);
  const vec2 uvOcclusion =
      transformedUv(material, vtx, kMaterialTextureSlotOcclusion);
  const vec2 uvEmissive =
      transformedUv(material, vtx, kMaterialTextureSlotEmissive);
  const vec2 uvClearcoat =
      transformedUv(material, vtx, kMaterialTextureSlotClearcoat);
  const vec2 uvClearcoatRoughness =
      transformedUv(material, vtx, kMaterialTextureSlotClearcoatRoughness);
  const vec2 uvClearcoatNormal =
      transformedUv(material, vtx, kMaterialTextureSlotClearcoatNormal);
  const vec2 uvSpecular =
      transformedUv(material, vtx, kMaterialTextureSlotSpecular);
  const vec2 uvSpecularColor =
      transformedUv(material, vtx, kMaterialTextureSlotSpecularColor);
  const vec2 uvSheenColor =
      transformedUv(material, vtx, kMaterialTextureSlotSheenColor);
  const vec2 uvSheenRoughness =
      transformedUv(material, vtx, kMaterialTextureSlotSheenRoughness);

  vec4 baseColor = material.baseColorFactor;
  if (baseColorTexId != kInvalidTextureBindlessIndex) {
    baseColor *=
        textureBindless2D(baseColorTexId, baseColorSampler, uvBaseColor);
  }

  const float alphaCutoff = material.metallicRoughnessOcclusionAlphaCutoff.w;
  if (alphaMode == kAlphaModeMask && baseColor.a < alphaCutoff) {
    discard;
  }

  vec4 mrSample = vec4(1.0);
  if (metallicRoughnessTexId != kInvalidTextureBindlessIndex) {
    mrSample = textureBindless2D(metallicRoughnessTexId,
                                 metallicRoughnessSampler, uvMetallicRoughness);
  }

  float metallic = 0.0;
  float roughness = kBrdfMinRoughness;
  vec3 f0 = vec3(0.0);
  vec3 f90 = vec3(0.0);
  vec3 diffuseColor = vec3(0.0);
  const float ior = material.transmissionThicknessIorPadding.z;
  decodeMaterialBaseWorkflow(
      material, workflow, baseColor, mrSample, uvSpecular, specularTexId,
      specularSampler, uvSpecularColor, specularColorTexId,
      specularColorSampler, ior, metallic, roughness, f0, f90, diffuseColor);

  float occlusion = sampleMaterialOcclusion(
      material, workflow, mrSample, metallicRoughnessTexId, uvOcclusion,
      occlusionTexId, occlusionSampler);
  float ao = mix(1.0, occlusion,
                 saturate(material.metallicRoughnessOcclusionAlphaCutoff.z));

  vec3 nGeom = normalize(vtx.worldNormal);
  if (!gl_FrontFacing) {
    nGeom *= -1.0;
  }

  bool hasClearcoat = (featureMask & kMaterialFeatureClearcoat) != 0u;
  float clearcoat = 0.0;
  float clearcoatRoughness = kBrdfMinRoughness;
  vec3 clearcoatF0 = vec3(0.04);
  vec3 clearcoatReflectance90 = vec3(1.0);
  vec3 clearcoatAttenuation = vec3(1.0);
  if (hasClearcoat) {
    clearcoat = saturate(material.sheenRoughnessClearcoatFactors.y);
    if (clearcoatTexId != kInvalidTextureBindlessIndex) {
      clearcoat *= textureBindless2D(clearcoatTexId, clearcoatSampler, uvClearcoat).r;
    }
    clearcoatRoughness =
        clamp(material.sheenRoughnessClearcoatFactors.z, kBrdfMinRoughness, 1.0);
    if (clearcoatRoughnessTexId != kInvalidTextureBindlessIndex) {
      clearcoatRoughness =
          clamp(clearcoatRoughness *
                    textureBindless2D(clearcoatRoughnessTexId,
                                      clearcoatRoughnessSampler,
                                      uvClearcoatRoughness)
                        .g,
                kBrdfMinRoughness, 1.0);
    }
  }

  vec3 nBase = nGeom;
  if (normalTexId != kInvalidTextureBindlessIndex) {
    vec3 normalTexel =
        textureBindless2D(normalTexId, normalSampler, uvNormal).xyz * 2.0 - 1.0;
    normalTexel.xy *= materialNormalScale(material);
    float normalTexelLen = length(normalTexel);
    if (normalTexelLen > kBrdfEpsilon) {
      normalTexel /= normalTexelLen;
    } else {
      normalTexel = vec3(0.0, 0.0, 1.0);
    }
    // Derivative-based TBN avoids reliance on imported tangent sign
    // conventions.
    nBase = applyNormalMap(nBase, vtx.worldPos, uvNormal, normalTexel);
  }

  vec3 nClearcoat = nGeom;
  if (hasClearcoat && clearcoatNormalTexId != kInvalidTextureBindlessIndex) {
    vec3 clearcoatNormalTexel =
        textureBindless2D(clearcoatNormalTexId, clearcoatNormalSampler,
                          uvClearcoatNormal)
            .xyz *
        2.0 - 1.0;
    clearcoatNormalTexel.xy *= material.sheenRoughnessClearcoatFactors.w;
    float clearcoatNormalTexelLen = length(clearcoatNormalTexel);
    if (clearcoatNormalTexelLen > kBrdfEpsilon) {
      clearcoatNormalTexel /= clearcoatNormalTexelLen;
    } else {
      clearcoatNormalTexel = vec3(0.0, 0.0, 1.0);
    }
    vec3 perturbedClearcoatNormal =
        applyNormalMap(nClearcoat, vtx.worldPos, uvClearcoatNormal,
                       clearcoatNormalTexel);
    // Very glossy clearcoat turns harsh normal-map distortion into a plastic
    // shell without specular AA. Bias back toward the geometric normal so the
    // reflection remains coherent.
    float clearcoatNormalBlend =
        clamp(sqrt(clearcoatRoughness), kBrdfMinRoughness, 1.0);
    nClearcoat =
        normalize(mix(nClearcoat, perturbedClearcoatNormal, clearcoatNormalBlend));
  }

  vec3 emissive = material.emissiveFactorStrength.xyz;
  if (emissiveTexId != kInvalidTextureBindlessIndex) {
    emissive *=
        textureBindless2D(emissiveTexId, emissiveSampler, uvEmissive).rgb;
  }
  emissive *= materialEmissiveStrength(material);

  bool iorCompatMode = isIorCompatMode(ior);
  vec3 v = normalize(pc.frameData.cameraPos.xyz - vtx.worldPos);
  float ndotv = max(dot(nBase, v), 0.001);
  float clearcoatNdotV = max(dot(nClearcoat, v), 0.001);

  float alphaRoughness = roughness * roughness;
  float sheenWeight =
      ((featureMask & kMaterialFeatureSheen) != 0u)
          ? saturate(material.sheenColorFactorWeight.w)
          : 0.0;
  float sheenRoughness = clamp(material.sheenRoughnessClearcoatFactors.x,
                               kBrdfMinRoughness, 1.0);
  vec3 sheenColor = material.sheenColorFactorWeight.xyz;
  if (sheenColorTexId != kInvalidTextureBindlessIndex) {
    sheenColor *=
        textureBindless2D(sheenColorTexId, sheenColorSampler, uvSheenColor).rgb;
  }
  if (sheenRoughnessTexId != kInvalidTextureBindlessIndex) {
    sheenRoughness =
        clamp(sheenRoughness *
                  textureBindless2D(sheenRoughnessTexId, sheenRoughnessSampler,
                                    uvSheenRoughness)
                      .a,
              kBrdfMinRoughness, 1.0);
  }
  if (hasClearcoat) {
    vec3 clearcoatLayerF = fresnelSchlick(clearcoatNdotV, clearcoatF0);
    clearcoatAttenuation = vec3(1.0) - clearcoat * clearcoatLayerF;
  }

  vec3 baseDirectLighting = vec3(0.0);
  vec3 directSheen = vec3(0.0);
  vec3 clearcoatDirectLighting = vec3(0.0);
  for (uint lightIndex = 0u; lightIndex < pc.frameData.directionalLightCount;
       ++lightIndex) {
    DirectionalLightGpuData light =
        pc.frameData.directionalLightBuffer.lights[lightIndex];
    vec3 l = normalize(-directionalLightDirection(light));
    vec3 lightRadiance =
        directionalLightColor(light) * directionalLightIlluminance(light);
    accumulateDirectLightContribution(
        lightRadiance, l, v, nBase, nClearcoat, diffuseColor, f0, f90, ndotv,
        clearcoatNdotV, alphaRoughness, sheenColor, sheenWeight,
        sheenRoughness, hasClearcoat, clearcoat, clearcoatRoughness,
        clearcoatF0, clearcoatReflectance90, baseDirectLighting, directSheen,
        clearcoatDirectLighting);
  }
  for (uint lightIndex = 0u; lightIndex < pc.frameData.localLightCount;
       ++lightIndex) {
    LocalLightGpuData light = pc.frameData.localLightBuffer.lights[lightIndex];
    vec3 pointToLight = localLightPosition(light) - vtx.worldPos;
    float distanceSq = dot(pointToLight, pointToLight);
    if (distanceSq <= kBrdfEpsilon) {
      continue;
    }

    vec3 l = pointToLight * inversesqrt(distanceSq);
    float attenuation =
        punctualRangeAttenuation(distanceSq, localLightRange(light));
    if (localLightType(light) == kLocalLightTypeSpot) {
      attenuation *= spotAngularAttenuation(
          localLightDirection(light), pointToLight, localLightInnerCos(light),
          localLightOuterCos(light));
    }
    if (attenuation <= 0.0) {
      continue;
    }

    vec3 lightRadiance =
        localLightColor(light) * localLightIntensity(light) * attenuation;
    accumulateDirectLightContribution(
        lightRadiance, l, v, nBase, nClearcoat, diffuseColor, f0, f90, ndotv,
        clearcoatNdotV, alphaRoughness, sheenColor, sheenWeight,
        sheenRoughness, hasClearcoat, clearcoat, clearcoatRoughness,
        clearcoatF0, clearcoatReflectance90, baseDirectLighting, directSheen,
        clearcoatDirectLighting);
  }

  vec3 baseBrdfLutSample = vec3(0.0);
  vec3 sheenBrdfLutSample = vec3(0.0);
  bool hasBrdfLut = (pc.frameData.flags & kFrameDataFlagHasBrdfLut) != 0u &&
                    pc.frameData.brdfLutTexId != kInvalidTextureBindlessIndex;
  if (hasBrdfLut) {
    vec2 baseBrdfUv = clamp(vec2(ndotv, 1.0 - roughness * roughness),
                            vec2(0.0), vec2(1.0));
    baseBrdfLutSample =
        textureBindless2D(pc.frameData.brdfLutTexId, 0, baseBrdfUv).rgb;
    if (sheenWeight > 0.0) {
      vec2 sheenBrdfUv = clamp(vec2(ndotv, 1.0 - sheenRoughness * sheenRoughness),
                               vec2(0.0), vec2(1.0));
      sheenBrdfLutSample =
          textureBindless2D(pc.frameData.brdfLutTexId, 0, sheenBrdfUv).rgb;
    }
  }
  float indirectScale = 1.0;
  if (sheenWeight > 0.0 && hasBrdfLut) {
    indirectScale = computeSheenAlbedoScalingIndirect(
        sheenColor, sheenWeight, sheenBrdfLutSample);
  }

  vec3 iblDiffuse = vec3(0.0);
  vec3 iblSpecular = vec3(0.0);
  vec3 iblSheen = vec3(0.0);
  vec3 clearcoatIblSpecular = vec3(0.0);
  bool hasIndirectLighting = false;
  if ((pc.frameData.flags & kFrameDataFlagHasIblDiffuse) != 0u &&
      pc.frameData.irradianceTexId != kInvalidTextureBindlessIndex) {
    vec3 irradiance =
        textureBindlessCube(pc.frameData.irradianceTexId,
                            pc.frameData.cubemapSamplerId, nBase)
            .rgb;
    if (iorCompatMode) {
      iblDiffuse = vec3(0.0);
    } else if (hasBrdfLut) {
      iblDiffuse =
          computeIblDiffuse(diffuseColor, f0, f90, irradiance, baseBrdfLutSample);
    } else {
      iblDiffuse =
          diffuseColor * (1.0 - max3(fresnelSchlick(ndotv, f0, f90))) *
          irradiance;
    }
    hasIndirectLighting = true;
  }

  if ((pc.frameData.flags & kFrameDataFlagHasIblSpecular) != 0u &&
      pc.frameData.prefilteredGgxTexId != kInvalidTextureBindlessIndex) {
    vec3 r = reflect(-v, nBase);
    if (hasBrdfLut) {
      float mipCount =
          float(textureBindlessQueryLevelsCube(pc.frameData.prefilteredGgxTexId));
      float lod = roughness * max(mipCount - 1.0, 0.0);
      vec3 prefiltered =
          textureBindlessCubeLod(pc.frameData.prefilteredGgxTexId,
                                 pc.frameData.cubemapSamplerId, r, lod)
              .rgb;
      iblSpecular =
          computeIblSpecular(f0, f90, prefiltered, baseBrdfLutSample);
    } else {
      iblSpecular =
          textureBindlessCube(pc.frameData.prefilteredGgxTexId,
                              pc.frameData.cubemapSamplerId, r)
              .rgb *
          fresnelSchlick(ndotv, f0);
    }
    hasIndirectLighting = true;
  }

  if ((pc.frameData.flags & kFrameDataFlagHasIblSheen) != 0u &&
      pc.frameData.prefilteredCharlieTexId != kInvalidTextureBindlessIndex &&
      hasBrdfLut && sheenWeight > 0.0) {
    vec3 r = reflect(-v, nBase);
    float mipCount = float(
        textureBindlessQueryLevelsCube(pc.frameData.prefilteredCharlieTexId));
    float lod = sheenRoughness * max(mipCount - 1.0, 0.0);
    vec3 sheenEnv =
        textureBindlessCubeLod(pc.frameData.prefilteredCharlieTexId,
                               pc.frameData.cubemapSamplerId, r, lod)
            .rgb;
    iblSheen = computeIblSheen(sheenColor, sheenWeight, sheenEnv,
                               sheenBrdfLutSample);
    hasIndirectLighting = true;
  }

  if (hasClearcoat && clearcoat > 0.0 &&
      (pc.frameData.flags & kFrameDataFlagHasIblSpecular) != 0u &&
      pc.frameData.prefilteredGgxTexId != kInvalidTextureBindlessIndex) {
    vec3 clearcoatR = reflect(-v, nClearcoat);
    if (hasBrdfLut) {
      vec2 clearcoatBrdfUv =
          clamp(vec2(clearcoatNdotV, 1.0 - clearcoatRoughness * clearcoatRoughness),
                vec2(0.0), vec2(1.0));
      vec3 clearcoatBrdfLutSample =
          textureBindless2D(pc.frameData.brdfLutTexId, 0, clearcoatBrdfUv).rgb;
      float mipCount =
          float(textureBindlessQueryLevelsCube(pc.frameData.prefilteredGgxTexId));
      float lod = clearcoatRoughness * max(mipCount - 1.0, 0.0);
      vec3 prefiltered =
          textureBindlessCubeLod(pc.frameData.prefilteredGgxTexId,
                                 pc.frameData.cubemapSamplerId, clearcoatR, lod)
              .rgb;
      clearcoatIblSpecular =
          clearcoat *
          computeIblSpecular(clearcoatF0, clearcoatReflectance90, prefiltered,
                             clearcoatBrdfLutSample);
    } else {
      clearcoatIblSpecular =
          clearcoat *
          textureBindlessCube(pc.frameData.prefilteredGgxTexId,
                              pc.frameData.cubemapSamplerId, clearcoatR)
              .rgb *
          fresnelSchlick(clearcoatNdotV, clearcoatF0);
    }
    hasIndirectLighting = true;
  }

  vec3 indirectLighting =
      clearcoatAttenuation *
          (iblSheen + indirectScale * (iblDiffuse + iblSpecular)) +
      clearcoatIblSpecular;
  if (hasIndirectLighting) {
    indirectLighting *= ao;
  }
  vec3 directLighting =
      clearcoatAttenuation * (directSheen + baseDirectLighting) +
      clearcoatDirectLighting;
  vec3 color =
      directLighting + indirectLighting + clearcoatAttenuation * emissive;
  color = max(color, vec3(0.0));
  if ((pc.frameData.flags & kFrameDataFlagOutputLinearToSrgb) != 0u) {
    color = pow(color, vec3(1.0 / 2.2));
  }

  float outAlpha = (alphaMode == kAlphaModeOpaque) ? 1.0 : baseColor.a;
  out_FragColor = vec4(color, outAlpha);
}
