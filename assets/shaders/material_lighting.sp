// Shared material evaluation and lighting for the opaque and transmissive
// fragment shader passes. Include after common.sp, BRDF.sp, and
// material_inputs.sp.

const uint kAlphaModeOpaque = 0u;
const uint kAlphaModeMask = 1u;

float applySpecularAARoughnessBias(float roughness, vec3 shadingNormal) {
  vec3 dndx = dFdx(shadingNormal);
  vec3 dndy = dFdy(shadingNormal);
  float normalVariance =
      clamp(0.5 * (dot(dndx, dndx) + dot(dndy, dndy)), 0.0, 1.0);
  float alpha = max(roughness * roughness, kBrdfMinRoughness);
  alpha = clamp(alpha + 0.35 * normalVariance, kBrdfMinRoughness, 1.0);
  return clamp(sqrt(alpha), kBrdfMinRoughness, 1.0);
}

struct ShadedMaterial {
  vec4 baseColor;
  vec3 diffuseColor;
  vec3 f0;
  vec3 f90;
  float metallic;
  float roughness;
  float alphaRoughness;
  float ao;
  float screenAo;
  float ior;
  bool iorCompatMode;
  vec3 nGeom;
  vec3 nBase;
  vec3 ambientBentNormal;
  vec3 nClearcoat;
  vec3 v;
  float ndotv;
  float clearcoatNdotV;
  bool hasClearcoat;
  float clearcoat;
  float clearcoatRoughness;
  vec3 clearcoatF0;
  vec3 clearcoatReflectance90;
  vec3 clearcoatAttenuation;
  float sheenWeight;
  float sheenRoughness;
  vec3 sheenColor;
  vec3 emissive;
};

ShadedMaterial evaluateMaterial(MaterialData material, PerVertex vtx) {
  ShadedMaterial sm;

  const uint featureMask = materialFeatureMask(material);
  const uint workflow = materialWorkflow(material);
  const uint matSampler = pc.frameData.materialSamplerId;

  const uint baseColorTexId =
      getMaterialTextureIndex(material, kMaterialTextureSlotBaseColor);
  const uint metallicRoughnessTexId =
      getMaterialTextureIndex(material, kMaterialTextureSlotMetallicRoughness);
  const uint normalTexId =
      getMaterialTextureIndex(material, kMaterialTextureSlotNormal);
  const uint occlusionTexId =
      getMaterialTextureIndex(material, kMaterialTextureSlotOcclusion);
  const uint emissiveTexId =
      getMaterialTextureIndex(material, kMaterialTextureSlotEmissive);
  const uint clearcoatTexId =
      getMaterialTextureIndex(material, kMaterialTextureSlotClearcoat);
  const uint clearcoatRoughnessTexId =
      getMaterialTextureIndex(material, kMaterialTextureSlotClearcoatRoughness);
  const uint clearcoatNormalTexId =
      getMaterialTextureIndex(material, kMaterialTextureSlotClearcoatNormal);
  const uint specularTexId =
      getMaterialTextureIndex(material, kMaterialTextureSlotSpecular);
  const uint specularColorTexId =
      getMaterialTextureIndex(material, kMaterialTextureSlotSpecularColor);
  const uint sheenColorTexId =
      getMaterialTextureIndex(material, kMaterialTextureSlotSheenColor);
  const uint sheenRoughnessTexId =
      getMaterialTextureIndex(material, kMaterialTextureSlotSheenRoughness);

  const vec2 uvBaseColor =
      transformedUv(material, vtx, kMaterialTextureSlotBaseColor);
  const vec2 uvMetallicRoughness =
      transformedUv(material, vtx, kMaterialTextureSlotMetallicRoughness);
  const vec2 uvNormal = transformedUv(material, vtx, kMaterialTextureSlotNormal);
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

  sm.baseColor = material.header.baseColorFactor;
  if (baseColorTexId != kInvalidTextureBindlessIndex) {
    sm.baseColor *= textureBindless2D(baseColorTexId, matSampler, uvBaseColor);
  }

  vec4 mrSample = vec4(1.0);
  if (metallicRoughnessTexId != kInvalidTextureBindlessIndex) {
    mrSample =
        textureBindless2D(metallicRoughnessTexId, matSampler, uvMetallicRoughness);
  }
  sm.metallic = 0.0;
  sm.roughness = kBrdfMinRoughness;
  sm.f0 = vec3(0.0);
  sm.f90 = vec3(0.0);
  sm.diffuseColor = vec3(0.0);
  sm.ior = materialIor(material);
  decodeMaterialBaseWorkflow(
      material, workflow, sm.baseColor, mrSample, uvSpecular, specularTexId,
      matSampler, uvSpecularColor, specularColorTexId, matSampler, sm.ior,
      sm.metallic, sm.roughness, sm.f0, sm.f90, sm.diffuseColor);

  float occlusion = sampleMaterialOcclusion(
      material, workflow, mrSample, metallicRoughnessTexId, uvOcclusion,
      occlusionTexId, matSampler);
  sm.ao = mix(1.0, occlusion,
              saturate(material.header.metallicRoughnessOcclusionAlphaCutoff.z));
  sm.screenAo = 1.0;

  sm.nGeom = normalize(vtx.worldNormal);
  if (!gl_FrontFacing) {
    sm.nGeom *= -1.0;
  }

  sm.hasClearcoat = material.hasClearcoat &&
                    (featureMask & kMaterialFeatureClearcoat) != 0u;
  sm.clearcoat = 0.0;
  sm.clearcoatRoughness = kBrdfMinRoughness;
  sm.clearcoatF0 = vec3(0.04);
  sm.clearcoatReflectance90 = vec3(1.0);
  sm.clearcoatAttenuation = vec3(1.0);
  if (sm.hasClearcoat) {
    sm.clearcoat = saturate(material.clearcoat.clearcoatFactors.x);
    if (clearcoatTexId != kInvalidTextureBindlessIndex) {
      sm.clearcoat *=
          textureBindless2D(clearcoatTexId, matSampler, uvClearcoat).r;
    }
    sm.clearcoatRoughness =
        clamp(material.clearcoat.clearcoatFactors.y, kBrdfMinRoughness, 1.0);
    if (clearcoatRoughnessTexId != kInvalidTextureBindlessIndex) {
      sm.clearcoatRoughness = clamp(
          sm.clearcoatRoughness *
              textureBindless2D(clearcoatRoughnessTexId, matSampler,
                                uvClearcoatRoughness).g,
          kBrdfMinRoughness, 1.0);
    }
  }

  sm.nBase = sm.nGeom;
  if (normalTexId != kInvalidTextureBindlessIndex) {
    vec3 n = textureBindless2D(normalTexId, matSampler, uvNormal).xyz * 2.0 - 1.0;
    n.xy *= materialNormalScale(material);
    float nLen = length(n);
    if (nLen > kEpsilon) {
      n /= nLen;
    } else {
      n = vec3(0.0, 0.0, 1.0);
    }
    sm.nBase = applyNormalMap(sm.nBase, vtx.worldTangent, vtx.worldPos,
                              uvNormal, n);
  }
  sm.ambientBentNormal = sm.nBase;
  if ((pc.frameData.flags & kFrameDataFlagHasAmbientOcclusion) != 0u &&
      pc.frameData.ambientOcclusionTexId != kInvalidTextureBindlessIndex &&
      pc.frameData.ambientOcclusionSamplerId != kInvalidSamplerBindlessIndex) {
    ivec2 aoSize = max(textureBindlessSize2D(pc.frameData.ambientOcclusionTexId),
                       ivec2(1));
    vec2 aoUv = gl_FragCoord.xy / vec2(aoSize);
    vec4 aoSample = textureBindless2D(pc.frameData.ambientOcclusionTexId,
                                      pc.frameData.ambientOcclusionSamplerId,
                                      aoUv);
    if ((pc.frameData.ambientOcclusionFlags &
         kAmbientOcclusionFlagScalarAo) != 0u) {
      sm.screenAo = saturate(aoSample.r);
    }
    if ((pc.frameData.flags & kFrameDataFlagHasAmbientBentNormal) != 0u &&
        (pc.frameData.ambientOcclusionFlags &
         kAmbientOcclusionFlagBentNormal) != 0u) {
      vec3 viewBent = aoSample.rgb * 2.0 - 1.0;
      float bentLenSq = dot(viewBent, viewBent);
      if (bentLenSq > 1.0e-6) {
        vec3 worldBent = transpose(mat3(pc.frameData.view)) *
                         (viewBent * inversesqrt(bentLenSq));
        sm.ambientBentNormal = normalize(mix(sm.nBase, worldBent, 0.75));
      }
    }
  }
  sm.roughness = applySpecularAARoughnessBias(sm.roughness, sm.nBase);

  sm.nClearcoat = sm.nGeom;
  if (sm.hasClearcoat && clearcoatNormalTexId != kInvalidTextureBindlessIndex) {
    vec3 n = textureBindless2D(clearcoatNormalTexId, matSampler, uvClearcoatNormal)
                 .xyz * 2.0 - 1.0;
    n.xy *= material.clearcoat.clearcoatFactors.z;
    float nLen = length(n);
    if (nLen > kEpsilon) {
      n /= nLen;
    } else {
      n = vec3(0.0, 0.0, 1.0);
    }
    vec3 perturbed =
        applyNormalMap(sm.nClearcoat, vtx.worldTangent, vtx.worldPos,
                       uvClearcoatNormal, n);
    float blend = clamp(sqrt(sm.clearcoatRoughness), kBrdfMinRoughness, 1.0);
    sm.nClearcoat = normalize(mix(sm.nClearcoat, perturbed, blend));
  }
  sm.clearcoatRoughness =
      applySpecularAARoughnessBias(sm.clearcoatRoughness, sm.nClearcoat);

  sm.emissive = material.header.emissiveFactorStrength.xyz;
  if (emissiveTexId != kInvalidTextureBindlessIndex) {
    sm.emissive *= textureBindless2D(emissiveTexId, matSampler, uvEmissive).rgb;
  }
  sm.emissive *= materialEmissiveStrength(material);

  sm.iorCompatMode = isIorCompatMode(sm.ior);
  sm.v = normalize(pc.frameData.cameraPos.xyz - vtx.worldPos);
  sm.ndotv = max(dot(sm.nBase, sm.v), 0.001);
  sm.clearcoatNdotV = max(dot(sm.nClearcoat, sm.v), 0.001);
  sm.alphaRoughness = sm.roughness * sm.roughness;

  sm.sheenWeight =
      (material.hasSheen && (featureMask & kMaterialFeatureSheen) != 0u)
          ? saturate(material.sheen.sheenColorFactorWeight.w)
          : 0.0;
  sm.sheenRoughness = clamp(material.sheen.sheenRoughnessReserved.x,
                            kBrdfMinRoughness, 1.0);
  sm.sheenColor = material.sheen.sheenColorFactorWeight.xyz;
  if (sheenColorTexId != kInvalidTextureBindlessIndex) {
    sm.sheenColor *=
        textureBindless2D(sheenColorTexId, matSampler, uvSheenColor).rgb;
  }
  if (sheenRoughnessTexId != kInvalidTextureBindlessIndex) {
    sm.sheenRoughness = clamp(
        sm.sheenRoughness *
            textureBindless2D(sheenRoughnessTexId, matSampler, uvSheenRoughness)
                .a,
        kBrdfMinRoughness, 1.0);
  }

  if (sm.hasClearcoat) {
    vec3 ccF = fresnelSchlick(sm.clearcoatNdotV, sm.clearcoatF0);
    sm.clearcoatAttenuation = vec3(1.0) - sm.clearcoat * ccF;
  }

  return sm;
}

void accumulateSurfaceLightContribution(
    vec3 lightRadiance, vec3 l, ShadedMaterial sm,
    inout vec3 directDiffuse, inout vec3 directSpecular,
    inout vec3 directSheen, inout vec3 clearcoatDirectLighting) {
  float ndotl = max(dot(sm.nBase, l), 0.0);
  float clearcoatNdotL = max(dot(sm.nClearcoat, l), 0.0);
  vec3 halfVector = sm.v + l;
  float halfLenSq = dot(halfVector, halfVector);

  if (ndotl > 0.0 && halfLenSq > kEpsilon) {
    vec3 h = halfVector * inversesqrt(halfLenSq);
    float ndoth = max(dot(sm.nBase, h), 0.0);
    float ldoth = max(dot(l, h), 0.0);
    float vdoth = max(dot(sm.v, h), 0.0);

    vec3 f = specularReflection(vdoth, sm.f0, sm.f90);
    float g = geometryOcclusion(ndotl, sm.ndotv, sm.alphaRoughness);
    float d = distributionGGX(ndoth, sm.alphaRoughness);

    float sheenScale = sm.sheenWeight > 0.0
        ? computeSheenAlbedoScalingDirect(sm.sheenColor, sm.ndotv, ndotl,
                                          sm.sheenRoughness)
        : 1.0;
    directDiffuse +=
        sheenScale * ndotl * lightRadiance *
        ((1.0 - max3(f)) *
         diffuseBurley(sm.diffuseColor, ndotl, sm.ndotv, ldoth,
                       sm.alphaRoughness));
    directSpecular +=
        sheenScale * ndotl * lightRadiance *
        (f * g * d / max(4.0 * ndotl * sm.ndotv, kEpsilon));
    directSheen += computeDirectSheen(sm.sheenColor, sm.sheenWeight,
                                      sm.sheenRoughness, ndotl, sm.ndotv,
                                      ndoth, lightRadiance);
  }

  if (sm.hasClearcoat && sm.clearcoat > 0.0 && clearcoatNdotL > 0.0 &&
      halfLenSq > kEpsilon) {
    vec3 h = halfVector * inversesqrt(halfLenSq);
    float ccNdotH = max(dot(sm.nClearcoat, h), 0.0);
    float ccVdotH = max(dot(sm.v, h), 0.0);
    float ccAlpha = sm.clearcoatRoughness * sm.clearcoatRoughness;
    vec3 ccF = specularReflection(ccVdotH, sm.clearcoatF0,
                                  sm.clearcoatReflectance90);
    float ccG = geometryOcclusion(clearcoatNdotL, sm.clearcoatNdotV, ccAlpha);
    float ccD = distributionGGX(ccNdotH, ccAlpha);
    clearcoatDirectLighting +=
        sm.clearcoat * clearcoatNdotL * lightRadiance *
        (ccF * ccG * ccD /
         max(4.0 * clearcoatNdotL * sm.clearcoatNdotV, kEpsilon));
  }
}

struct IblResult {
  vec3 iblDiffuse;
  vec3 iblSpecular;
  vec3 iblSheen;
  vec3 clearcoatIblSpecular;
  float indirectScale;
  bool hasIndirectLighting;
};

struct DirectLightingResult {
  vec3 directDiffuse;
  vec3 directSpecular;
  vec3 directSheen;
  vec3 clearcoatDirectLighting;
  float shadowFactorDebug;
  float shadowCascadeIndexDebug;
  float shadowCascadeBlendDebug;
  float shadowPcfFactorDebug;
  float shadowReceiverDepthDebug;
  float shadowMapDepthDebug;
  float shadowPcssBlockerRatioDebug;
  float shadowPcssAverageBlockerDepthDebug;
  float shadowPcssFilterRadiusDebug;
};

struct HardShadowInspectResult {
  float receiverDepth;
  float receiverCompareDepth;
  float sampledDepth;
  float cascadeIndexDebug;
  float cascadeBlendDebug;
  float valid;
};

struct DirectionalShadowSampleContext {
  ShadowCascadeGpuData cascade;
  vec2 shadowUv;
  float receiverDepth;
  float receiverCompareDepth;
  uint cascadeIndex;
  bool valid;
};

DirectionalShadowSampleContext makeDirectionalShadowSampleContextForCascade(
    ShadowCascadeGpuData cascade, vec3 worldPos, vec3 surfaceNormal,
    vec3 lightDir) {
  DirectionalShadowSampleContext ctx;
  ctx.cascade = cascade;
  ctx.shadowUv = vec2(0.0);
  ctx.receiverDepth = 0.0;
  ctx.receiverCompareDepth = 0.0;
  ctx.cascadeIndex = 0u;
  ctx.valid = false;

  vec3 biasedWorldPos = worldPos;
  const float normalBias = ctx.cascade.biasParams.z * ctx.cascade.splitDepthTexelSize.z;
  const float normalLength = length(surfaceNormal);
  if (normalBias > 0.0 && normalLength > kEpsilon) {
    biasedWorldPos += (surfaceNormal / normalLength) * normalBias;
  }

  vec4 lightClip = ctx.cascade.lightViewProj * vec4(biasedWorldPos, 1.0);
  if (abs(lightClip.w) <= kEpsilon) {
    return ctx;
  }

  vec3 ndc = lightClip.xyz / lightClip.w;
  vec2 shadowUv = ndc.xy * 0.5 + 0.5;
  shadowUv.y = 1.0 - shadowUv.y;
  if (any(lessThan(shadowUv, vec2(0.0))) ||
      any(greaterThan(shadowUv, vec2(1.0))) || ndc.z < 0.0 ||
      ndc.z > 1.0) {
    return ctx;
  }

  const float constantBias = ctx.cascade.biasParams.x;
  const float slopeScale = ctx.cascade.biasParams.y;
  const float ndotl =
      clamp(dot(normalize(surfaceNormal), normalize(lightDir)), 0.0, 1.0);
  const float slopeBias = abs(constantBias) * slopeScale * (1.0 - ndotl);
  ctx.shadowUv = shadowUv;
  ctx.receiverDepth = ndc.z;
  ctx.receiverCompareDepth = ndc.z - (constantBias + slopeBias);
  ctx.valid = true;
  return ctx;
}

uint selectDirectionalShadowCascade(readonly ShadowFrameBuffer shadow,
                                    float viewDepth) {
  const uint cascadeCount =
      clamp(shadow.flagsCascadeCountLightIndex.y, 1u, kMaxShadowCascades);
  uint cascadeIndex = 0u;
  for (uint i = 0u; i + 1u < cascadeCount; ++i) {
    if (viewDepth > shadow.cascades[i].splitDepthTexelSize.y) {
      cascadeIndex = i + 1u;
    }
  }
  return cascadeIndex;
}

float directionalShadowViewDepth(vec3 worldPos) {
  return max(0.0, -(pc.frameData.view * vec4(worldPos, 1.0)).z);
}

float hardDirectionalShadowFactor(DirectionalShadowSampleContext ctx) {
  const uint shadowTexId = ctx.cascade.textureSampler.x;
  const uint shadowRawSamplerId = ctx.cascade.textureSampler.z;
  if (!ctx.valid || shadowTexId == kInvalidTextureBindlessIndex ||
      shadowRawSamplerId == kInvalidSamplerBindlessIndex) {
    return 1.0;
  }

  float sampleDepth =
      textureBindless2D(shadowTexId, shadowRawSamplerId, ctx.shadowUv).r;
  return ctx.receiverCompareDepth <= sampleDepth ? 1.0 : 0.0;
}

struct DirectionalShadowFilterResult {
  float factor;
  float pcssBlockerRatio;
  float pcssAverageBlockerDepth;
  float pcssFilterRadiusRatio;
};

DirectionalShadowFilterResult makeDirectionalShadowFilterResult(float factor,
                                                                float blockers,
                                                                float blockerDepth,
                                                                float filterRadius) {
  DirectionalShadowFilterResult r;
  r.factor = factor;
  r.pcssBlockerRatio = blockers;
  r.pcssAverageBlockerDepth = blockerDepth;
  r.pcssFilterRadiusRatio = filterRadius;
  return r;
}

int directionalShadowPcfSampleCount(readonly ShadowFrameBuffer shadow) {
  return int(clamp(shadow.filterParams.x, 1u, kMaxShadowPcfSamples));
}

int directionalShadowPcssBlockerSampleCount(readonly ShadowFrameBuffer shadow) {
  return int(clamp(shadow.filterParams.y, 1u, kMaxShadowPcfSamples));
}

int directionalShadowPcssFilterSampleCount(readonly ShadowFrameBuffer shadow) {
  return int(clamp(shadow.filterParams.z, 1u, kMaxShadowPcfSamples));
}

bool directionalShadowFrameFlag(readonly ShadowFrameBuffer shadow, uint flag) {
  return (shadow.flagsCascadeCountLightIndex.x & flag) != 0u;
}

ivec2 directionalShadowTextureSize(DirectionalShadowSampleContext ctx,
                                   uint shadowTexId) {
  const uint storedShadowSize = ctx.cascade.textureSampler.w;
  if (storedShadowSize > 0u) {
    return ivec2(int(storedShadowSize));
  }
  return max(textureBindlessSize2D(shadowTexId), ivec2(1));
}

vec2 directionalShadowTexelSize(DirectionalShadowSampleContext ctx,
                                uint shadowTexId) {
  const uint storedShadowSize = ctx.cascade.textureSampler.w;
  if (storedShadowSize > 0u) {
    return vec2(1.0 / float(storedShadowSize));
  }
  return 1.0 / vec2(directionalShadowTextureSize(ctx, shadowTexId));
}

float sampleDirectionalShadowVisibility(DirectionalShadowSampleContext ctx,
                                        uint shadowTexId,
                                        uint shadowRawSamplerId,
                                        vec2 sampleUv) {
  if (any(lessThan(sampleUv, vec2(0.0))) ||
      any(greaterThan(sampleUv, vec2(1.0)))) {
    return 1.0;
  }
  float sampleDepth =
      textureBindless2D(shadowTexId, shadowRawSamplerId, sampleUv).r;
  return ctx.receiverCompareDepth <= sampleDepth ? 1.0 : 0.0;
}

float sampleDirectionalShadowCompare(DirectionalShadowSampleContext ctx,
                                     uint shadowTexId,
                                     uint shadowCompareSamplerId,
                                     vec2 sampleUv) {
  if (any(lessThan(sampleUv, vec2(0.0))) ||
      any(greaterThan(sampleUv, vec2(1.0)))) {
    return 1.0;
  }
  return textureBindless2DShadow(
      shadowTexId, shadowCompareSamplerId,
      vec3(sampleUv, ctx.receiverCompareDepth));
}

float sampleDirectionalShadowDepth(DirectionalShadowSampleContext ctx) {
  const uint shadowTexId = ctx.cascade.textureSampler.x;
  const uint shadowRawSamplerId = ctx.cascade.textureSampler.z;
  if (!ctx.valid || shadowTexId == kInvalidTextureBindlessIndex ||
      shadowRawSamplerId == kInvalidSamplerBindlessIndex) {
    return 0.0;
  }
  return textureBindless2D(shadowTexId, shadowRawSamplerId, ctx.shadowUv).r;
}

float pcfGridDirectionalShadowFactor(readonly ShadowFrameBuffer shadow,
                                     DirectionalShadowSampleContext ctx) {
  const uint shadowTexId = ctx.cascade.textureSampler.x;
  const uint shadowCompareSamplerId = ctx.cascade.textureSampler.y;
  if (!ctx.valid || shadowTexId == kInvalidTextureBindlessIndex ||
      shadowCompareSamplerId == kInvalidSamplerBindlessIndex) {
    return 1.0;
  }

  int sampleCount = directionalShadowPcfSampleCount(shadow);
  if (sampleCount <= 1) {
    return hardDirectionalShadowFactor(ctx);
  }

  vec2 texelSize = directionalShadowTexelSize(ctx, shadowTexId);
  int gridSide = int(ceil(sqrt(float(sampleCount))));
  float halfGrid = (float(gridSide) - 1.0) * 0.5;

  float visibility = 0.0;
  int usedSamples = 0;
  for (int sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex) {
    int x = sampleIndex % gridSide;
    int y = sampleIndex / gridSide;
    vec2 gridOffset = vec2(float(x), float(y)) - vec2(halfGrid);
    visibility += sampleDirectionalShadowCompare(
        ctx, shadowTexId, shadowCompareSamplerId,
        ctx.shadowUv + gridOffset * texelSize);
    ++usedSamples;
  }
  return visibility / float(max(usedSamples, 1));
}

const vec2 kShadowPoissonDisk[64] = vec2[64](
    vec2(-0.613392, 0.617481), vec2(0.170019, -0.040254),
    vec2(-0.299417, 0.791925), vec2(0.645680, 0.493210),
    vec2(-0.651784, 0.717887), vec2(0.421003, 0.027070),
    vec2(-0.817194, -0.271096), vec2(-0.705374, -0.668203),
    vec2(0.977050, -0.108615), vec2(0.063326, 0.142369),
    vec2(0.203528, 0.214331), vec2(-0.667531, 0.326090),
    vec2(-0.098422, -0.295755), vec2(-0.885922, 0.215369),
    vec2(0.566637, 0.605213), vec2(0.039766, -0.396100),
    vec2(0.751946, 0.453352), vec2(0.078707, -0.715323),
    vec2(-0.075838, -0.529344), vec2(0.724479, -0.580798),
    vec2(0.222999, -0.215125), vec2(-0.467574, -0.405438),
    vec2(-0.248268, -0.814753), vec2(0.354411, -0.887570),
    vec2(0.175817, 0.382366), vec2(0.487472, -0.063082),
    vec2(-0.084078, 0.898312), vec2(0.488876, -0.783441),
    vec2(0.470016, 0.217933), vec2(-0.696890, -0.549791),
    vec2(-0.149693, 0.605762), vec2(0.034211, 0.979980),
    vec2(0.503098, -0.308878), vec2(-0.016205, -0.872921),
    vec2(0.385784, -0.393902), vec2(-0.146886, -0.859249),
    vec2(0.643361, 0.164098), vec2(0.634388, -0.049471),
    vec2(-0.688894, 0.007843), vec2(0.464034, -0.188818),
    vec2(-0.440840, 0.137486), vec2(0.364483, 0.511704),
    vec2(0.034028, 0.325968), vec2(0.099094, -0.308023),
    vec2(0.693960, -0.366253), vec2(0.678884, -0.204688),
    vec2(0.001801, 0.780328), vec2(0.145177, -0.898984),
    vec2(0.062655, -0.611866), vec2(0.315226, -0.604297),
    vec2(-0.780145, 0.486251), vec2(-0.371868, 0.882138),
    vec2(0.200476, 0.494430), vec2(-0.494552, -0.711051),
    vec2(0.612476, 0.705252), vec2(-0.578845, -0.768792),
    vec2(-0.772454, -0.090976), vec2(0.504440, 0.372295),
    vec2(0.155736, 0.065157), vec2(0.391522, 0.849605),
    vec2(-0.620106, -0.328104), vec2(0.789239, -0.419965),
    vec2(-0.545396, 0.538133), vec2(-0.178564, -0.596057));

// If kMaxShadowPcfSamples is increased above 64, samples will repeat unless
// this table is expanded or the indexing strategy is changed.
vec2 shadowPoissonDiskSample(int sampleIndex) {
  return kShadowPoissonDisk[sampleIndex & 63];
}

uint shadowHash(uint value) {
  value ^= value >> 16u;
  value *= 0x7feb352du;
  value ^= value >> 15u;
  value *= 0x846ca68bu;
  value ^= value >> 16u;
  return value;
}

mat2 shadowPoissonRotation(readonly ShadowFrameBuffer shadow,
                           DirectionalShadowSampleContext ctx,
                           uint shadowTexId) {
  uint seed = shadow.filterParams.w ^ (ctx.cascadeIndex * 0x9e3779b9u);
  if ((shadow.flagsCascadeCountLightIndex.x &
       kShadowFrameFlagFixedPoissonRotation) != 0u) {
    uint orientation = seed & 3u;
    if (orientation == 1u) {
      return mat2(0.0, 1.0, -1.0, 0.0);
    }
    if (orientation == 2u) {
      return mat2(-1.0, 0.0, 0.0, -1.0);
    }
    if (orientation == 3u) {
      return mat2(0.0, -1.0, 1.0, 0.0);
    }
    return mat2(1.0);
  }
  ivec2 shadowSize = directionalShadowTextureSize(ctx, shadowTexId);
  vec2 clampedUv = clamp(ctx.shadowUv, vec2(0.0), vec2(0.99999994));
  ivec2 texelCoord =
      clamp(ivec2(clampedUv * vec2(shadowSize)), ivec2(0),
            shadowSize - ivec2(1));
  seed ^= uint(texelCoord.x) * 0x85ebca6bu;
  seed ^= uint(texelCoord.y) * 0xc2b2ae35u;
  float angle = float(shadowHash(seed) & 0xffffu) *
                (6.28318530718 / 65536.0);
  float s = sin(angle);
  float c = cos(angle);
  return mat2(c, -s, s, c);
}

bool tryResolveUniformPoissonPcf(DirectionalShadowSampleContext ctx,
                                 uint shadowTexId, uint shadowRawSamplerId,
                                 vec2 texelSize, mat2 rotation,
                                 float radius, out float factor) {
  const vec2 probes[8] = vec2[8](
      vec2(1.0, 0.0), vec2(-1.0, 0.0), vec2(0.0, 1.0), vec2(0.0, -1.0),
      vec2(0.707107, 0.707107), vec2(-0.707107, 0.707107),
      vec2(0.707107, -0.707107), vec2(-0.707107, -0.707107));
  float sum = sampleDirectionalShadowVisibility(
      ctx, shadowTexId, shadowRawSamplerId, ctx.shadowUv);
  for (int probeIndex = 0; probeIndex < 8; ++probeIndex) {
    vec2 offset = rotation * probes[probeIndex];
    sum += sampleDirectionalShadowVisibility(
        ctx, shadowTexId, shadowRawSamplerId,
        ctx.shadowUv + offset * radius * texelSize);
  }
  if (sum <= 0.0) {
    factor = 0.0;
    return true;
  }
  if (sum >= 9.0) {
    factor = 1.0;
    return true;
  }
  factor = 1.0;
  return false;
}

float poissonPcfDirectionalShadowFactor(readonly ShadowFrameBuffer shadow,
                                        DirectionalShadowSampleContext ctx,
                                        int sampleCount,
                                        float radiusTexels) {
  const uint shadowTexId = ctx.cascade.textureSampler.x;
  const uint shadowCompareSamplerId = ctx.cascade.textureSampler.y;
  const uint shadowRawSamplerId = ctx.cascade.textureSampler.z;
  if (!ctx.valid || shadowTexId == kInvalidTextureBindlessIndex ||
      shadowCompareSamplerId == kInvalidSamplerBindlessIndex ||
      shadowRawSamplerId == kInvalidSamplerBindlessIndex) {
    return 1.0;
  }

  if (sampleCount <= 1) {
    return hardDirectionalShadowFactor(ctx);
  }

  vec2 texelSize = directionalShadowTexelSize(ctx, shadowTexId);
  mat2 rotation = shadowPoissonRotation(shadow, ctx, shadowTexId);
  float radius = max(radiusTexels, 0.0);
  if (sampleCount >= 16) {
    float uniformFactor = 1.0;
    if (tryResolveUniformPoissonPcf(ctx, shadowTexId, shadowRawSamplerId,
                                    texelSize, rotation, radius,
                                    uniformFactor)) {
      return uniformFactor;
    }
  }
  float visibility = 0.0;
  int usedSamples = 0;
  for (int sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex) {
    vec2 offset = rotation * shadowPoissonDiskSample(sampleIndex);
    visibility += sampleDirectionalShadowCompare(
        ctx, shadowTexId, shadowCompareSamplerId,
        ctx.shadowUv + offset * radius * texelSize);
    ++usedSamples;
  }
  return visibility / float(max(usedSamples, 1));
}

DirectionalShadowFilterResult poissonPcfDirectionalShadowFactor(
    readonly ShadowFrameBuffer shadow, DirectionalShadowSampleContext ctx) {
  return makeDirectionalShadowFilterResult(
      poissonPcfDirectionalShadowFactor(
          shadow, ctx, directionalShadowPcfSampleCount(shadow), 1.0),
      0.0, 0.0, 0.0);
}

DirectionalShadowFilterResult pcssDirectionalShadowFactor(
    readonly ShadowFrameBuffer shadow, DirectionalShadowSampleContext ctx,
    float lightAngularRadiusRadians, int blockerSampleCount,
    int filterSampleCount) {
  const uint shadowTexId = ctx.cascade.textureSampler.x;
  const uint shadowRawSamplerId = ctx.cascade.textureSampler.z;
  if (!ctx.valid || shadowTexId == kInvalidTextureBindlessIndex ||
      shadowRawSamplerId == kInvalidSamplerBindlessIndex) {
    return makeDirectionalShadowFilterResult(1.0, 0.0, 0.0, 0.0);
  }

  vec2 texelSize = directionalShadowTexelSize(ctx, shadowTexId);
  mat2 rotation = shadowPoissonRotation(shadow, ctx, shadowTexId);
  float lightAngleScale = tan(max(lightAngularRadiusRadians, 0.0)) *
                          max(shadow.fadeParams.w, 0.0);
  float depthSpanWorld = max(ctx.cascade.pcssParams.x, kEpsilon);
  float texelWorldSize = max(ctx.cascade.splitDepthTexelSize.z, kEpsilon);
  float sourceRadiusTexels = lightAngleScale * depthSpanWorld / texelWorldSize;
  float searchRadiusTexels =
      clamp(sourceRadiusTexels, 0.0,
            max(ctx.cascade.pcssParams.y, 0.0));

  float blockerDepthSum = 0.0;
  int blockerCount = 0;
  for (int sampleIndex = 0; sampleIndex < blockerSampleCount; ++sampleIndex) {
    vec2 offset = rotation * shadowPoissonDiskSample(sampleIndex);
    vec2 sampleUv = ctx.shadowUv + offset * searchRadiusTexels * texelSize;
    if (any(lessThan(sampleUv, vec2(0.0))) ||
        any(greaterThan(sampleUv, vec2(1.0)))) {
      continue;
    }
    float sampleDepth =
        textureBindless2D(shadowTexId, shadowRawSamplerId, sampleUv).r;
    if (sampleDepth < ctx.receiverCompareDepth) {
      blockerDepthSum += sampleDepth;
      ++blockerCount;
    }
  }

  float blockerRatio =
      float(blockerCount) / float(max(blockerSampleCount, 1));
  if (blockerCount == 0) {
    return makeDirectionalShadowFilterResult(1.0, blockerRatio, 0.0, 0.0);
  }

  float avgBlockerDepth = blockerDepthSum / float(blockerCount);
  float blockerReceiverDistanceWorld =
      max(ctx.receiverDepth - avgBlockerDepth, 0.0) * depthSpanWorld;
  float penumbraWorld =
      lightAngleScale * blockerReceiverDistanceWorld;
  float filterRadiusTexels =
      clamp(penumbraWorld / texelWorldSize, 0.0,
            max(ctx.cascade.pcssParams.z, 0.0));
  float filterRadiusRatio =
      max(ctx.cascade.pcssParams.z, 0.0) > kEpsilon
          ? saturate(filterRadiusTexels / ctx.cascade.pcssParams.z)
          : 0.0;
  float visibility = poissonPcfDirectionalShadowFactor(
      shadow, ctx, max(filterSampleCount, 1), filterRadiusTexels);
  return makeDirectionalShadowFilterResult(visibility, blockerRatio,
                                           avgBlockerDepth,
                                           filterRadiusRatio);
}

DirectionalShadowFilterResult pcssDirectionalShadowFactor(
    readonly ShadowFrameBuffer shadow, DirectionalShadowSampleContext ctx,
    float lightAngularRadiusRadians) {
  return pcssDirectionalShadowFactor(
      shadow, ctx, lightAngularRadiusRadians,
      directionalShadowPcssBlockerSampleCount(shadow),
      directionalShadowPcssFilterSampleCount(shadow));
}

DirectionalShadowFilterResult directionalShadowFactorFromContext(
    readonly ShadowFrameBuffer shadow, DirectionalShadowSampleContext ctx,
    uint filterMode, float lightAngularRadiusRadians) {
  if (filterMode == kShadowFilterModePCF3x3) {
    return makeDirectionalShadowFilterResult(
        pcfGridDirectionalShadowFactor(shadow, ctx), 0.0, 0.0, 0.0);
  }
  if (filterMode == kShadowFilterModePoissonPCF) {
    return poissonPcfDirectionalShadowFactor(shadow, ctx);
  }
  if (filterMode == kShadowFilterModePCSS) {
    return pcssDirectionalShadowFactor(shadow, ctx,
                                       lightAngularRadiusRadians);
  }
  return makeDirectionalShadowFilterResult(hardDirectionalShadowFactor(ctx),
                                           0.0, 0.0, 0.0);
}

DirectionalShadowFilterResult directionalShadowBlendFactorFromContext(
    readonly ShadowFrameBuffer shadow, DirectionalShadowSampleContext ctx,
    uint filterMode, float lightAngularRadiusRadians) {
  if (filterMode == kShadowFilterModePCSS) {
    // directionalShadowBlendFactorFromContext deliberately clamps
    // directionalShadowPcssBlockerSampleCount to 8 and
    // directionalShadowPcssFilterSampleCount to 16 before
    // pcssDirectionalShadowFactor for blend work. This can cause subtle
    // quality discontinuities when the primary cascade uses higher counts;
    // raise 8/16 here only after accepting the performance cost.
    const int blendBlockerSamples =
        min(directionalShadowPcssBlockerSampleCount(shadow), 8);
    const int blendFilterSamples =
        min(directionalShadowPcssFilterSampleCount(shadow), 16);
    return pcssDirectionalShadowFactor(shadow, ctx, lightAngularRadiusRadians,
                                       blendBlockerSamples,
                                       blendFilterSamples);
  }
  if (filterMode == kShadowFilterModePoissonPCF) {
    return poissonPcfDirectionalShadowFactor(shadow, ctx);
  }
  return directionalShadowFactorFromContext(shadow, ctx, filterMode,
                                            lightAngularRadiusRadians);
}

struct DirectionalShadowResult {
  float factor;
  float cascadeIndexDebug;
  float cascadeBlendDebug;
  float pcfFactorDebug;
  float receiverDepthDebug;
  float shadowMapDepthDebug;
  float pcssBlockerRatioDebug;
  float pcssAverageBlockerDepthDebug;
  float pcssFilterRadiusDebug;
};

struct DirectionalShadowCascadeState {
  DirectionalShadowSampleContext ctx;
  DirectionalShadowSampleContext nextCtx;
  float cascadeBlend;
  bool hasBlend;
};

DirectionalShadowCascadeState makeDirectionalShadowCascadeState(
    readonly ShadowFrameBuffer shadow, vec3 worldPos, vec3 surfaceNormal,
    vec3 lightDir) {
  DirectionalShadowCascadeState state;

  const float viewDepth = directionalShadowViewDepth(worldPos);
  const uint cascadeCount =
      clamp(shadow.flagsCascadeCountLightIndex.y, 1u, kMaxShadowCascades);
  const uint cascadeIndex = selectDirectionalShadowCascade(shadow, viewDepth);
  state.ctx = makeDirectionalShadowSampleContextForCascade(
      shadow.cascades[cascadeIndex], worldPos, surfaceNormal, lightDir);
  state.ctx.cascadeIndex = cascadeIndex;
  state.nextCtx = state.ctx;
  state.cascadeBlend = 0.0;
  state.hasBlend = false;

  if (cascadeIndex + 1u >= cascadeCount) {
    return state;
  }

  const ShadowCascadeGpuData cascade = shadow.cascades[cascadeIndex];
  const float cascadeSpan =
      max(cascade.splitDepthTexelSize.y - cascade.splitDepthTexelSize.x, 0.0);
  const float blendWidth = cascadeSpan * saturate(shadow.fadeParams.z);
  if (blendWidth <= kEpsilon) {
    return state;
  }

  const float blendStart = cascade.splitDepthTexelSize.y - blendWidth;
  if (viewDepth < blendStart || viewDepth > cascade.splitDepthTexelSize.y) {
    return state;
  }

  DirectionalShadowSampleContext nextCtx =
      makeDirectionalShadowSampleContextForCascade(
          shadow.cascades[cascadeIndex + 1u], worldPos, surfaceNormal,
          lightDir);
  nextCtx.cascadeIndex = cascadeIndex + 1u;
  if (!nextCtx.valid) {
    return state;
  }

  state.nextCtx = nextCtx;
  state.cascadeBlend = saturate((viewDepth - blendStart) / blendWidth);
  state.hasBlend = true;
  return state;
}

float directionalShadowMaxDistanceFade(readonly ShadowFrameBuffer shadow,
                                       float viewDepth) {
  const float fadeStart = shadow.fadeParams.x;
  const float fadeEnd = shadow.fadeParams.y;
  if (fadeEnd <= fadeStart + kEpsilon) {
    return 0.0;
  }
  return saturate((viewDepth - fadeStart) / (fadeEnd - fadeStart));
}

DirectionalShadowResult evaluateDirectionalShadow(
    readonly ShadowFrameBuffer shadow, vec3 worldPos, vec3 surfaceNormal,
    vec3 lightDir, float lightAngularRadiusRadians) {
  DirectionalShadowResult r;
  r.factor = 1.0;
  r.cascadeIndexDebug = 0.0;
  r.cascadeBlendDebug = 0.0;
  r.pcfFactorDebug = 1.0;
  r.receiverDepthDebug = 0.0;
  r.shadowMapDepthDebug = 0.0;
  r.pcssBlockerRatioDebug = 0.0;
  r.pcssAverageBlockerDepthDebug = 0.0;
  r.pcssFilterRadiusDebug = 0.0;

  const DirectionalShadowCascadeState state =
      makeDirectionalShadowCascadeState(shadow, worldPos, surfaceNormal,
                                        lightDir);

  const uint filterMode = shadow.flagsCascadeCountLightIndex.w;
  DirectionalShadowFilterResult filterResult =
      directionalShadowFactorFromContext(shadow, state.ctx, filterMode,
                                         lightAngularRadiusRadians);
  r.factor = filterResult.factor;
  r.pcssBlockerRatioDebug = filterResult.pcssBlockerRatio;
  r.pcssAverageBlockerDepthDebug = filterResult.pcssAverageBlockerDepth;
  r.pcssFilterRadiusDebug = filterResult.pcssFilterRadiusRatio;
  const bool needsPcfDebug =
      directionalShadowFrameFlag(shadow, kShadowFrameFlagVisualizePCFResult);
  const bool needsReceiverDepthDebug = directionalShadowFrameFlag(
      shadow, kShadowFrameFlagVisualizeReceiverDepth);
  const bool needsShadowMapDepthDebug = directionalShadowFrameFlag(
      shadow, kShadowFrameFlagVisualizeShadowMapDepth);
  const bool needsPcssDebug =
      directionalShadowFrameFlag(shadow, kShadowFrameFlagVisualizePCSSBlockers) ||
      directionalShadowFrameFlag(
          shadow, kShadowFrameFlagVisualizePCSSAverageBlockerDepth) ||
      directionalShadowFrameFlag(shadow,
                                 kShadowFrameFlagVisualizePCSSFilterRadius);
  if (needsPcfDebug) {
    r.pcfFactorDebug =
        (filterMode == kShadowFilterModePoissonPCF)
            ? filterResult.factor
            : poissonPcfDirectionalShadowFactor(shadow, state.ctx).factor;
  }
  if (needsReceiverDepthDebug) {
    r.receiverDepthDebug = state.ctx.valid ? state.ctx.receiverDepth : 0.0;
  }
  if (needsShadowMapDepthDebug) {
    r.shadowMapDepthDebug = sampleDirectionalShadowDepth(state.ctx);
  }
  r.cascadeIndexDebug = float(state.ctx.cascadeIndex);
  if (!state.hasBlend) {
    const float distanceFade = directionalShadowMaxDistanceFade(
        shadow, directionalShadowViewDepth(worldPos));
    r.factor = mix(r.factor, 1.0, distanceFade);
    return r;
  }

  DirectionalShadowFilterResult nextFilterResult =
      directionalShadowBlendFactorFromContext(shadow, state.nextCtx, filterMode,
                                              lightAngularRadiusRadians);
  r.cascadeBlendDebug = state.cascadeBlend;
  r.factor = mix(r.factor, nextFilterResult.factor, r.cascadeBlendDebug);
  if (needsPcfDebug) {
    float nextPcfFactor =
        (filterMode == kShadowFilterModePoissonPCF)
            ? nextFilterResult.factor
            : poissonPcfDirectionalShadowFactor(shadow, state.nextCtx).factor;
    r.pcfFactorDebug =
        mix(r.pcfFactorDebug, nextPcfFactor, r.cascadeBlendDebug);
  }
  if (needsReceiverDepthDebug) {
    r.receiverDepthDebug =
        mix(r.receiverDepthDebug,
            state.nextCtx.valid ? state.nextCtx.receiverDepth : 0.0,
            r.cascadeBlendDebug);
  }
  if (needsShadowMapDepthDebug) {
    r.shadowMapDepthDebug =
        mix(r.shadowMapDepthDebug, sampleDirectionalShadowDepth(state.nextCtx),
            r.cascadeBlendDebug);
  }
  if (needsPcssDebug) {
    r.pcssBlockerRatioDebug =
        mix(r.pcssBlockerRatioDebug, nextFilterResult.pcssBlockerRatio,
            r.cascadeBlendDebug);
    r.pcssAverageBlockerDepthDebug =
        mix(r.pcssAverageBlockerDepthDebug,
            nextFilterResult.pcssAverageBlockerDepth, r.cascadeBlendDebug);
    r.pcssFilterRadiusDebug =
        mix(r.pcssFilterRadiusDebug, nextFilterResult.pcssFilterRadiusRatio,
            r.cascadeBlendDebug);
  }
  const float distanceFade = directionalShadowMaxDistanceFade(
      shadow, directionalShadowViewDepth(worldPos));
  r.factor = mix(r.factor, 1.0, distanceFade);
  return r;
}

float directionalShadowFactor(readonly ShadowFrameBuffer shadow, vec3 worldPos,
                              vec3 surfaceNormal, vec3 lightDir) {
  const DirectionalShadowResult r =
      evaluateDirectionalShadow(shadow, worldPos, surfaceNormal, lightDir,
                                0.0);
  return r.factor;
}

HardShadowInspectResult inspectHardDirectionalShadow(
    readonly ShadowFrameBuffer shadow, vec3 worldPos, vec3 surfaceNormal,
    vec3 lightDir) {
  HardShadowInspectResult r;
  r.receiverDepth = 0.0;
  r.receiverCompareDepth = 0.0;
  r.sampledDepth = 0.0;
  r.cascadeIndexDebug = 0.0;
  r.cascadeBlendDebug = 0.0;
  r.valid = 0.0;

  const DirectionalShadowCascadeState state =
      makeDirectionalShadowCascadeState(shadow, worldPos, surfaceNormal,
                                        lightDir);
  r.cascadeIndexDebug = float(state.ctx.cascadeIndex);
  r.cascadeBlendDebug = state.cascadeBlend;

  const uint shadowTexId = state.ctx.cascade.textureSampler.x;
  const uint shadowCompareSamplerId = state.ctx.cascade.textureSampler.y;
  const uint shadowRawSamplerId = state.ctx.cascade.textureSampler.z;
  if (!state.ctx.valid || shadowTexId == kInvalidTextureBindlessIndex ||
      shadowCompareSamplerId == kInvalidSamplerBindlessIndex ||
      shadowRawSamplerId == kInvalidSamplerBindlessIndex) {
    return r;
  }

  r.receiverDepth = state.ctx.receiverDepth;
  r.receiverCompareDepth = state.ctx.receiverCompareDepth;
  r.sampledDepth =
      textureBindless2D(shadowTexId, shadowRawSamplerId, state.ctx.shadowUv).r;
  r.valid = 1.0;
  return r;
}

DirectLightingResult evaluateDirectLighting(ShadedMaterial sm, vec3 worldPos) {
  DirectLightingResult r;
  r.directDiffuse = vec3(0.0);
  r.directSpecular = vec3(0.0);
  r.directSheen = vec3(0.0);
  r.clearcoatDirectLighting = vec3(0.0);
  r.shadowFactorDebug = 1.0;
  r.shadowCascadeIndexDebug = 0.0;
  r.shadowCascadeBlendDebug = 0.0;
  r.shadowPcfFactorDebug = 1.0;
  r.shadowReceiverDepthDebug = 0.0;
  r.shadowMapDepthDebug = 0.0;
  r.shadowPcssBlockerRatioDebug = 0.0;
  r.shadowPcssAverageBlockerDepthDebug = 0.0;
  r.shadowPcssFilterRadiusDebug = 0.0;

  const bool frameShadowEnabled =
      (pc.frameData.shadowFlags & kShadowFrameFlagEnabled) != 0u;
  for (uint i = 0u; i < pc.frameData.directionalLightCount; ++i) {
    DirectionalLightGpuData light =
        pc.frameData.directionalLightBuffer.lights[i];
    vec3 l = normalize(-directionalLightDirection(light));
    float shadowFactor = 1.0;
    if (frameShadowEnabled) {
      ShadowFrameBuffer shadow = pc.frameData.shadowFrameBuffer;
      uvec4 shadowState = shadow.flagsCascadeCountLightIndex;
      if ((shadowState.x & kShadowFrameFlagEnabled) != 0u &&
          shadowState.y > 0u && shadowState.z == i) {
        DirectionalShadowResult shadowResult =
            evaluateDirectionalShadow(
                shadow, worldPos, sm.nGeom, l,
                directionalLightAngularRadiusRadians(light));
        shadowFactor = shadowResult.factor;
        r.shadowFactorDebug = shadowResult.factor;
        r.shadowCascadeIndexDebug = shadowResult.cascadeIndexDebug;
        r.shadowCascadeBlendDebug = shadowResult.cascadeBlendDebug;
        r.shadowPcfFactorDebug = shadowResult.pcfFactorDebug;
        r.shadowReceiverDepthDebug = shadowResult.receiverDepthDebug;
        r.shadowMapDepthDebug = shadowResult.shadowMapDepthDebug;
        r.shadowPcssBlockerRatioDebug =
            shadowResult.pcssBlockerRatioDebug;
        r.shadowPcssAverageBlockerDepthDebug =
            shadowResult.pcssAverageBlockerDepthDebug;
        r.shadowPcssFilterRadiusDebug =
            shadowResult.pcssFilterRadiusDebug;
      }
    }
    vec3 lr = directionalLightColor(light) * directionalLightIlluminance(light) *
              shadowFactor;
    accumulateSurfaceLightContribution(lr, l, sm, r.directDiffuse,
                                       r.directSpecular, r.directSheen,
                                       r.clearcoatDirectLighting);
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
    vec3 lr = localLightColor(light) * localLightIntensity(light) * att;
    accumulateSurfaceLightContribution(lr, l, sm, r.directDiffuse,
                                       r.directSpecular, r.directSheen,
                                       r.clearcoatDirectLighting);
  }

  return r;
}

IblResult evaluateIbl(ShadedMaterial sm) {
  IblResult r;
  r.iblDiffuse = vec3(0.0);
  r.iblSpecular = vec3(0.0);
  r.iblSheen = vec3(0.0);
  r.clearcoatIblSpecular = vec3(0.0);
  r.indirectScale = 1.0;
  r.hasIndirectLighting = false;

  vec3 baseBrdfLutSample = vec3(0.0);
  vec3 sheenBrdfLutSample = vec3(0.0);
  bool hasBrdfLut =
      (pc.frameData.flags & kFrameDataFlagHasBrdfLut) != 0u &&
      pc.frameData.brdfLutTexId != kInvalidTextureBindlessIndex;
  if (hasBrdfLut) {
    vec2 baseBrdfUv =
        clamp(vec2(sm.ndotv, 1.0 - sm.roughness * sm.roughness),
              vec2(0.0), vec2(1.0));
    baseBrdfLutSample =
        textureBindless2D(pc.frameData.brdfLutTexId, 0u, baseBrdfUv).rgb;
    if (sm.sheenWeight > 0.0) {
      vec2 sheenBrdfUv =
          clamp(vec2(sm.ndotv, 1.0 - sm.sheenRoughness * sm.sheenRoughness),
                vec2(0.0), vec2(1.0));
      sheenBrdfLutSample =
          textureBindless2D(pc.frameData.brdfLutTexId, 0u, sheenBrdfUv).rgb;
    }
  }
  if (sm.sheenWeight > 0.0 && hasBrdfLut) {
    r.indirectScale = computeSheenAlbedoScalingIndirect(
        sm.sheenColor, sm.sheenWeight, sheenBrdfLutSample);
  }

  if ((pc.frameData.flags & kFrameDataFlagHasIblDiffuse) != 0u &&
      pc.frameData.irradianceTexId != kInvalidTextureBindlessIndex) {
    vec3 irradiance =
        textureBindlessCube(pc.frameData.irradianceTexId,
                            pc.frameData.cubemapSamplerId,
                            sm.ambientBentNormal).rgb;
    if (!sm.iorCompatMode) {
      r.iblDiffuse = hasBrdfLut
                         ? computeIblDiffuse(sm.diffuseColor, sm.f0, sm.f90,
                                             irradiance, baseBrdfLutSample)
                         : sm.diffuseColor *
                               (1.0 -
                                max3(fresnelSchlick(sm.ndotv, sm.f0, sm.f90))) *
                               irradiance;
    }
    r.hasIndirectLighting = true;
  }

  if ((pc.frameData.flags & kFrameDataFlagHasIblSpecular) != 0u &&
      pc.frameData.prefilteredGgxTexId != kInvalidTextureBindlessIndex) {
    vec3 ref = reflect(-sm.v, sm.ambientBentNormal);
    if (hasBrdfLut) {
      float mipCount = float(
          textureBindlessQueryLevelsCube(pc.frameData.prefilteredGgxTexId));
      float lod = sm.roughness * max(mipCount - 1.0, 0.0);
      vec3 prefiltered =
          textureBindlessCubeLod(pc.frameData.prefilteredGgxTexId,
                                 pc.frameData.cubemapSamplerId, ref, lod).rgb;
      r.iblSpecular =
          computeIblSpecular(sm.f0, sm.f90, prefiltered, baseBrdfLutSample);
    } else {
      r.iblSpecular =
          textureBindlessCube(pc.frameData.prefilteredGgxTexId,
                              pc.frameData.cubemapSamplerId, ref).rgb *
          fresnelSchlick(sm.ndotv, sm.f0);
    }
    r.hasIndirectLighting = true;
  }

  if ((pc.frameData.flags & kFrameDataFlagHasIblSheen) != 0u &&
      pc.frameData.prefilteredCharlieTexId != kInvalidTextureBindlessIndex &&
      hasBrdfLut && sm.sheenWeight > 0.0) {
    vec3 ref = reflect(-sm.v, sm.ambientBentNormal);
    float mipCount = float(
        textureBindlessQueryLevelsCube(pc.frameData.prefilteredCharlieTexId));
    float lod = sm.sheenRoughness * max(mipCount - 1.0, 0.0);
    vec3 sheenEnv =
        textureBindlessCubeLod(pc.frameData.prefilteredCharlieTexId,
                               pc.frameData.cubemapSamplerId, ref, lod).rgb;
    r.iblSheen =
        computeIblSheen(sm.sheenColor, sm.sheenWeight, sheenEnv,
                        sheenBrdfLutSample);
    r.hasIndirectLighting = true;
  }

  if (sm.hasClearcoat && sm.clearcoat > 0.0 &&
      (pc.frameData.flags & kFrameDataFlagHasIblSpecular) != 0u &&
      pc.frameData.prefilteredGgxTexId != kInvalidTextureBindlessIndex) {
    vec3 ccRef = reflect(-sm.v, sm.nClearcoat);
    if (hasBrdfLut) {
      vec2 ccBrdfUv =
          clamp(vec2(sm.clearcoatNdotV,
                     1.0 - sm.clearcoatRoughness * sm.clearcoatRoughness),
                vec2(0.0), vec2(1.0));
      vec3 ccBrdfSample =
          textureBindless2D(pc.frameData.brdfLutTexId, 0u, ccBrdfUv).rgb;
      float mipCount = float(
          textureBindlessQueryLevelsCube(pc.frameData.prefilteredGgxTexId));
      float lod = sm.clearcoatRoughness * max(mipCount - 1.0, 0.0);
      vec3 prefiltered =
          textureBindlessCubeLod(pc.frameData.prefilteredGgxTexId,
                                 pc.frameData.cubemapSamplerId, ccRef, lod).rgb;
      r.clearcoatIblSpecular =
          sm.clearcoat *
          computeIblSpecular(sm.clearcoatF0, sm.clearcoatReflectance90,
                             prefiltered, ccBrdfSample);
    } else {
      r.clearcoatIblSpecular =
          sm.clearcoat *
          textureBindlessCube(pc.frameData.prefilteredGgxTexId,
                              pc.frameData.cubemapSamplerId, ccRef).rgb *
          fresnelSchlick(sm.clearcoatNdotV, sm.clearcoatF0);
    }
    r.hasIndirectLighting = true;
  }

  return r;
}
