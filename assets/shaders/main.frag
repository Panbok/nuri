#include "common.sp"
#include "BRDF.sp"

layout(location = 0) in PerVertex vtx;

layout(location = 0) out vec4 out_FragColor;

const uint kAlphaModeOpaque = 0u;
const uint kAlphaModeMask = 1u;

vec2 selectUv(vec2 uv0, vec2 uv1, uint uvSet) {
  return (uvSet == 1u) ? uv1 : uv0;
}

void main() {
  const MaterialGpuData material = pc.materialBuffer.materials[pc.materialIndex];

  const uint baseColorTexId = material.textureIndices0.x;
  const uint metallicRoughnessTexId = material.textureIndices0.y;
  const uint normalTexId = material.textureIndices0.z;
  const uint occlusionTexId = material.textureIndices0.w;
  const uint emissiveTexId = material.textureIndices1.x;
  const uint clearcoatTexId = material.textureIndices1.y;
  const uint clearcoatRoughnessTexId = material.textureIndices1.z;
  const uint clearcoatNormalTexId = material.textureIndices1.w;
  const uint sheenColorTexId = material.textureIndices2.x;
  const uint sheenRoughnessTexId = material.textureIndices2.y;
  const uint alphaMode = material.materialFlags.x;
  const uint featureMask = material.materialFlags.z;

  const uint baseColorUvSet = material.textureUvSets0.x;
  const uint metallicRoughnessUvSet = material.textureUvSets0.y;
  const uint normalUvSet = material.textureUvSets0.z;
  const uint occlusionUvSet = material.textureUvSets0.w;
  const uint emissiveUvSet = material.textureUvSets1.x;
  const uint clearcoatUvSet = material.textureUvSets1.y;
  const uint clearcoatRoughnessUvSet = material.textureUvSets1.z;
  const uint clearcoatNormalUvSet = material.textureUvSets1.w;
  const uint sheenColorUvSet = material.textureUvSets2.x;
  const uint sheenRoughnessUvSet = material.textureUvSets2.y;

  const uint baseColorSampler = material.textureSamplerIndices0.x;
  const uint metallicRoughnessSampler = material.textureSamplerIndices0.y;
  const uint normalSampler = material.textureSamplerIndices0.z;
  const uint occlusionSampler = material.textureSamplerIndices0.w;
  const uint emissiveSampler = material.textureSamplerIndices1.x;
  const uint clearcoatSampler = material.textureSamplerIndices1.y;
  const uint clearcoatRoughnessSampler = material.textureSamplerIndices1.z;
  const uint clearcoatNormalSampler = material.textureSamplerIndices1.w;
  const uint sheenColorSampler = material.textureSamplerIndices2.x;
  const uint sheenRoughnessSampler = material.textureSamplerIndices2.y;

  const vec2 uvBaseColor = applyTextureTransform(
      selectUv(vtx.uv0, vtx.uv1, baseColorUvSet),
      material.textureTransformOffsetScale[kMaterialTextureSlotBaseColor],
      material.textureTransformRotation[kMaterialTextureSlotBaseColor]);
  const vec2 uvMetallicRoughness = applyTextureTransform(
      selectUv(vtx.uv0, vtx.uv1, metallicRoughnessUvSet),
      material.textureTransformOffsetScale[kMaterialTextureSlotMetallicRoughness],
      material.textureTransformRotation[kMaterialTextureSlotMetallicRoughness]);
  const vec2 uvNormal = applyTextureTransform(
      selectUv(vtx.uv0, vtx.uv1, normalUvSet),
      material.textureTransformOffsetScale[kMaterialTextureSlotNormal],
      material.textureTransformRotation[kMaterialTextureSlotNormal]);
  const vec2 uvOcclusion = applyTextureTransform(
      selectUv(vtx.uv0, vtx.uv1, occlusionUvSet),
      material.textureTransformOffsetScale[kMaterialTextureSlotOcclusion],
      material.textureTransformRotation[kMaterialTextureSlotOcclusion]);
  const vec2 uvEmissive = applyTextureTransform(
      selectUv(vtx.uv0, vtx.uv1, emissiveUvSet),
      material.textureTransformOffsetScale[kMaterialTextureSlotEmissive],
      material.textureTransformRotation[kMaterialTextureSlotEmissive]);
  const vec2 uvClearcoat = applyTextureTransform(
      selectUv(vtx.uv0, vtx.uv1, clearcoatUvSet),
      material.textureTransformOffsetScale[kMaterialTextureSlotClearcoat],
      material.textureTransformRotation[kMaterialTextureSlotClearcoat]);
  const vec2 uvClearcoatRoughness = applyTextureTransform(
      selectUv(vtx.uv0, vtx.uv1, clearcoatRoughnessUvSet),
      material.textureTransformOffsetScale[kMaterialTextureSlotClearcoatRoughness],
      material.textureTransformRotation[kMaterialTextureSlotClearcoatRoughness]);
  const vec2 uvClearcoatNormal = applyTextureTransform(
      selectUv(vtx.uv0, vtx.uv1, clearcoatNormalUvSet),
      material.textureTransformOffsetScale[kMaterialTextureSlotClearcoatNormal],
      material.textureTransformRotation[kMaterialTextureSlotClearcoatNormal]);
  const vec2 uvSheenColor = applyTextureTransform(
      selectUv(vtx.uv0, vtx.uv1, sheenColorUvSet),
      material.textureTransformOffsetScale[kMaterialTextureSlotSheenColor],
      material.textureTransformRotation[kMaterialTextureSlotSheenColor]);
  const vec2 uvSheenRoughness = applyTextureTransform(
      selectUv(vtx.uv0, vtx.uv1, sheenRoughnessUvSet),
      material.textureTransformOffsetScale[kMaterialTextureSlotSheenRoughness],
      material.textureTransformRotation[kMaterialTextureSlotSheenRoughness]);

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

  float metallic = saturate(material.metallicRoughnessOcclusionAlphaCutoff.x *
                            mrSample.b);
  float roughness = clamp(material.metallicRoughnessOcclusionAlphaCutoff.y *
                              mrSample.g,
                          kBrdfMinRoughness, 1.0);

  float occlusion = 1.0;
  if (occlusionTexId != kInvalidTextureBindlessIndex) {
    occlusion =
        textureBindless2D(occlusionTexId, occlusionSampler, uvOcclusion).r;
  } else if (metallicRoughnessTexId != kInvalidTextureBindlessIndex) {
    occlusion = mrSample.r;
  }
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
    normalTexel.xy *= material.emissiveFactorNormalScale.w;
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

  vec3 emissive = material.emissiveFactorNormalScale.xyz;
  if (emissiveTexId != kInvalidTextureBindlessIndex) {
    emissive *=
        textureBindless2D(emissiveTexId, emissiveSampler, uvEmissive).rgb;
  }

  vec3 v = normalize(pc.frameData.cameraPos.xyz - vtx.worldPos);
  float ndotv = max(dot(nBase, v), 0.001);
  float clearcoatNdotV = max(dot(nClearcoat, v), 0.001);

  vec3 f0 = mix(vec3(0.04), baseColor.rgb, metallic);
  vec3 diffuseColor = mix(baseColor.rgb, vec3(0.0), metallic);
  float alphaRoughness = roughness * roughness;
  float reflectance = max(max(f0.r, f0.g), f0.b);
  vec3 reflectance90 = vec3(clamp(reflectance * 25.0, 0.0, 1.0));
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

  const vec3 lightPos = vec3(0.0, 0.0, -5.0);
  const vec3 lightColor = vec3(1.0);
  vec3 l = normalize(lightPos - vtx.worldPos);
  float ndotl = max(dot(nBase, l), 0.0);
  float clearcoatNdotL = max(dot(nClearcoat, l), 0.0);
  vec3 baseDirectLighting = vec3(0.0);
  vec3 directSheen = vec3(0.0);
  vec3 clearcoatDirectLighting = vec3(0.0);
  vec3 halfVector = v + l;
  float halfLenSq = dot(halfVector, halfVector);
  if (ndotl > 0.0 && halfLenSq > kBrdfEpsilon) {
    vec3 h = halfVector * inversesqrt(halfLenSq);
    float ndoth = max(dot(nBase, h), 0.0);
    float ldoth = max(dot(l, h), 0.0);
    float vdoth = max(dot(v, h), 0.0);

    vec3 f = specularReflection(vdoth, f0, reflectance90);
    float g = geometryOcclusion(ndotl, ndotv, alphaRoughness);
    float d = microfacetDistribution(ndoth, alphaRoughness);
    vec3 diffuse = (1.0 - f) *
                   diffuseBurley(diffuseColor, ndotl, ndotv, ldoth,
                                 alphaRoughness);
    vec3 specular = f * g * d / max(4.0 * ndotl * ndotv, kBrdfEpsilon);
    baseDirectLighting = ndotl * lightColor * (diffuse + specular);
    directSheen =
        computeDirectSheen(sheenColor, sheenWeight, sheenRoughness, ndotl,
                           ndotv, ndoth, lightColor);
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
    clearcoatDirectLighting =
        clearcoat * clearcoatNdotL * lightColor * clearcoatSpecular;
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
  float directScale = 1.0;
  float indirectScale = 1.0;
  if (sheenWeight > 0.0) {
    directScale = computeSheenAlbedoScalingDirect(
        sheenColor, ndotv, ndotl, sheenRoughness);
    if (hasBrdfLut) {
      indirectScale =
          computeSheenAlbedoScalingIndirect(sheenColor, sheenBrdfLutSample);
    }
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
    if (hasBrdfLut) {
      iblDiffuse = computeIblDiffuse(diffuseColor, f0, roughness, ndotv,
                                     irradiance, baseBrdfLutSample);
    } else {
      iblDiffuse = diffuseColor * irradiance;
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
          computeIblSpecular(f0, roughness, ndotv, prefiltered,
                             baseBrdfLutSample);
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
          computeIblSpecular(clearcoatF0, clearcoatRoughness, clearcoatNdotV,
                             prefiltered, clearcoatBrdfLutSample);
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
      clearcoatAttenuation * (directSheen + directScale * baseDirectLighting) +
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
