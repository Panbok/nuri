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
  float ior;
  bool iorCompatMode;
  vec3 nGeom;
  vec3 nBase;
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
};

struct HardShadowInspectResult {
  float receiverDepth;
  float receiverCompareDepth;
  float sampledDepth;
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

DirectionalShadowSampleContext makeDirectionalShadowSampleContext(
    readonly ShadowFrameBuffer shadow, vec3 worldPos, vec3 surfaceNormal,
    vec3 lightDir) {
  const float viewDepth = directionalShadowViewDepth(worldPos);
  const uint cascadeIndex = selectDirectionalShadowCascade(shadow, viewDepth);
  DirectionalShadowSampleContext ctx =
      makeDirectionalShadowSampleContextForCascade(
          shadow.cascades[cascadeIndex], worldPos, surfaceNormal, lightDir);
  ctx.cascadeIndex = cascadeIndex;
  return ctx;
}

float hardDirectionalShadowFactor(DirectionalShadowSampleContext ctx) {
  const uint shadowTexId = ctx.cascade.textureSampler.x;
  const uint shadowSamplerId = ctx.cascade.textureSampler.y;
  if (!ctx.valid || shadowTexId == kInvalidTextureBindlessIndex ||
      shadowSamplerId == kInvalidSamplerBindlessIndex) {
    return 1.0;
  }

  return textureBindless2DShadow(
      shadowTexId, shadowSamplerId,
      vec3(ctx.shadowUv, ctx.receiverCompareDepth));
}

int directionalShadowPcfSampleCount(DirectionalShadowSampleContext ctx) {
  return int(clamp(ctx.cascade.textureSampler.w, 1u, kMaxShadowPcfSamples));
}

float sampleDirectionalShadowVisibility(DirectionalShadowSampleContext ctx,
                                        uint shadowTexId,
                                        uint shadowRawSamplerId,
                                        vec2 sampleUv) {
  float sampleDepth =
      textureBindless2D(shadowTexId, shadowRawSamplerId, sampleUv).r;
  return ctx.receiverCompareDepth <= sampleDepth ? 1.0 : 0.0;
}

float pcfGridDirectionalShadowFactor(DirectionalShadowSampleContext ctx) {
  const uint shadowTexId = ctx.cascade.textureSampler.x;
  const uint shadowRawSamplerId = ctx.cascade.textureSampler.z;
  if (!ctx.valid || shadowTexId == kInvalidTextureBindlessIndex ||
      shadowRawSamplerId == kInvalidSamplerBindlessIndex) {
    return 1.0;
  }

  ivec2 shadowSize = max(textureBindlessSize2D(shadowTexId), ivec2(1));
  vec2 texelSize = 1.0 / vec2(shadowSize);
  int sampleCount = directionalShadowPcfSampleCount(ctx);
  int gridSide = int(ceil(sqrt(float(sampleCount))));
  float halfGrid = (float(gridSide) - 1.0) * 0.5;

  float visibility = 0.0;
  int usedSamples = 0;
  for (int sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex) {
    int x = sampleIndex % gridSide;
    int y = sampleIndex / gridSide;
    vec2 gridOffset = vec2(float(x), float(y)) - vec2(halfGrid);
    visibility += sampleDirectionalShadowVisibility(
        ctx, shadowTexId, shadowRawSamplerId,
        ctx.shadowUv + gridOffset * texelSize);
    ++usedSamples;
  }
  return visibility / float(max(usedSamples, 1));
}

float poissonPcfDirectionalShadowFactor(DirectionalShadowSampleContext ctx) {
  // TODO: Replace this placeholder with true Poisson disk sampling; it
  // currently delegates to the regular grid PCF implementation.
  return pcfGridDirectionalShadowFactor(ctx);
}

float directionalShadowFactorFromContext(DirectionalShadowSampleContext ctx,
                                         uint filterMode) {
  if (filterMode == kShadowFilterModePCF3x3) {
    return pcfGridDirectionalShadowFactor(ctx);
  }
  if (filterMode == kShadowFilterModePoissonPCF) {
    return poissonPcfDirectionalShadowFactor(ctx);
  }
  return hardDirectionalShadowFactor(ctx);
}

float directionalShadowFactor(readonly ShadowFrameBuffer shadow, vec3 worldPos,
                              vec3 surfaceNormal, vec3 lightDir) {
  const float viewDepth = directionalShadowViewDepth(worldPos);
  const uint cascadeCount =
      clamp(shadow.flagsCascadeCountLightIndex.y, 1u, kMaxShadowCascades);
  const uint cascadeIndex = selectDirectionalShadowCascade(shadow, viewDepth);
  DirectionalShadowSampleContext ctx =
      makeDirectionalShadowSampleContextForCascade(
          shadow.cascades[cascadeIndex], worldPos, surfaceNormal, lightDir);
  ctx.cascadeIndex = cascadeIndex;

  const uint filterMode = shadow.flagsCascadeCountLightIndex.w;
  float shadowFactor = directionalShadowFactorFromContext(ctx, filterMode);
  if (cascadeIndex + 1u >= cascadeCount) {
    return shadowFactor;
  }

  const ShadowCascadeGpuData cascade = shadow.cascades[cascadeIndex];
  const float cascadeSpan = max(cascade.splitDepthTexelSize.y -
                                    cascade.splitDepthTexelSize.x,
                                0.0);
  const float blendWidth =
      cascadeSpan * saturate(shadow.fadeParams.z);
  if (blendWidth <= kEpsilon) {
    return shadowFactor;
  }

  const float blendStart = cascade.splitDepthTexelSize.y - blendWidth;
  if (viewDepth < blendStart || viewDepth > cascade.splitDepthTexelSize.y) {
    return shadowFactor;
  }

  DirectionalShadowSampleContext nextCtx =
      makeDirectionalShadowSampleContextForCascade(
          shadow.cascades[cascadeIndex + 1u], worldPos, surfaceNormal,
          lightDir);
  nextCtx.cascadeIndex = cascadeIndex + 1u;
  if (!nextCtx.valid) {
    return shadowFactor;
  }

  const float nextShadowFactor =
      directionalShadowFactorFromContext(nextCtx, filterMode);
  const float blendT = saturate((viewDepth - blendStart) / blendWidth);
  return mix(shadowFactor, nextShadowFactor, blendT);
}

struct DirectionalShadowResult {
  float factor;
  float cascadeIndexDebug;
  float cascadeBlendDebug;
};

DirectionalShadowResult evaluateDirectionalShadow(
    readonly ShadowFrameBuffer shadow, vec3 worldPos, vec3 surfaceNormal,
    vec3 lightDir) {
  DirectionalShadowResult r;
  r.factor = 1.0;
  r.cascadeIndexDebug = 0.0;
  r.cascadeBlendDebug = 0.0;

  const float viewDepth = directionalShadowViewDepth(worldPos);
  const uint cascadeCount =
      clamp(shadow.flagsCascadeCountLightIndex.y, 1u, kMaxShadowCascades);
  const uint cascadeIndex = selectDirectionalShadowCascade(shadow, viewDepth);
  DirectionalShadowSampleContext ctx =
      makeDirectionalShadowSampleContextForCascade(
          shadow.cascades[cascadeIndex], worldPos, surfaceNormal, lightDir);
  ctx.cascadeIndex = cascadeIndex;

  const uint filterMode = shadow.flagsCascadeCountLightIndex.w;
  r.factor = directionalShadowFactorFromContext(ctx, filterMode);
  r.cascadeIndexDebug = float(cascadeIndex);
  if (cascadeIndex + 1u >= cascadeCount) {
    return r;
  }

  const ShadowCascadeGpuData cascade = shadow.cascades[cascadeIndex];
  const float cascadeSpan = max(cascade.splitDepthTexelSize.y -
                                    cascade.splitDepthTexelSize.x,
                                0.0);
  const float blendWidth =
      cascadeSpan * saturate(shadow.fadeParams.z);
  if (blendWidth <= kEpsilon) {
    return r;
  }

  const float blendStart = cascade.splitDepthTexelSize.y - blendWidth;
  if (viewDepth < blendStart || viewDepth > cascade.splitDepthTexelSize.y) {
    return r;
  }

  DirectionalShadowSampleContext nextCtx =
      makeDirectionalShadowSampleContextForCascade(
          shadow.cascades[cascadeIndex + 1u], worldPos, surfaceNormal,
          lightDir);
  nextCtx.cascadeIndex = cascadeIndex + 1u;
  if (!nextCtx.valid) {
    return r;
  }

  const float nextShadowFactor =
      directionalShadowFactorFromContext(nextCtx, filterMode);
  r.cascadeBlendDebug = saturate((viewDepth - blendStart) / blendWidth);
  r.factor = mix(r.factor, nextShadowFactor, r.cascadeBlendDebug);
  return r;
}

HardShadowInspectResult inspectHardDirectionalShadow(
    readonly ShadowFrameBuffer shadow, vec3 worldPos, vec3 surfaceNormal,
    vec3 lightDir) {
  HardShadowInspectResult r;
  r.receiverDepth = 0.0;
  r.receiverCompareDepth = 0.0;
  r.sampledDepth = 0.0;
  r.valid = 0.0;

  DirectionalShadowSampleContext ctx =
      makeDirectionalShadowSampleContext(shadow, worldPos, surfaceNormal,
                                         lightDir);
  const uint shadowTexId = ctx.cascade.textureSampler.x;
  const uint shadowCompareSamplerId = ctx.cascade.textureSampler.y;
  const uint shadowRawSamplerId = ctx.cascade.textureSampler.z;
  if (!ctx.valid || shadowTexId == kInvalidTextureBindlessIndex ||
      shadowCompareSamplerId == kInvalidSamplerBindlessIndex ||
      shadowRawSamplerId == kInvalidSamplerBindlessIndex) {
    return r;
  }

  r.receiverDepth = ctx.receiverDepth;
  r.receiverCompareDepth = ctx.receiverCompareDepth;
  r.sampledDepth =
      textureBindless2D(shadowTexId, shadowRawSamplerId, ctx.shadowUv).r;
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
            evaluateDirectionalShadow(shadow, worldPos, sm.nBase, l);
        shadowFactor = shadowResult.factor;
        r.shadowFactorDebug = shadowResult.factor;
        r.shadowCascadeIndexDebug = shadowResult.cascadeIndexDebug;
        r.shadowCascadeBlendDebug = shadowResult.cascadeBlendDebug;
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
                            pc.frameData.cubemapSamplerId, sm.nBase).rgb;
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
    vec3 ref = reflect(-sm.v, sm.nBase);
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
    vec3 ref = reflect(-sm.v, sm.nBase);
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
