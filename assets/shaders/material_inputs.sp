float sampleMaterialSpecularWeight(MaterialGpuData material, vec2 uvSpecular,
                                   uint texId, uint samplerIndex) {
  float specularWeight = material.specularColorFactorSpecular.w;
  if (texId != kInvalidTextureBindlessIndex) {
    specularWeight *= textureBindless2D(texId, samplerIndex, uvSpecular).a;
  }
  return clamp(specularWeight, 0.0, 1.0);
}

vec3 sampleMaterialSpecularColor(MaterialGpuData material, vec2 uvSpecularColor,
                                 uint texId, uint samplerIndex) {
  vec3 specularColor = material.specularColorFactorSpecular.rgb;
  if (texId != kInvalidTextureBindlessIndex) {
    specularColor *= textureBindless2D(texId, samplerIndex, uvSpecularColor).rgb;
  }
  return max(specularColor, vec3(0.0));
}

vec4 sampleMaterialSpecularGlossiness(MaterialGpuData material,
                                      vec2 uvSpecularGlossiness, uint texId,
                                      uint samplerIndex) {
  vec4 specGloss = vec4(material.specularColorFactorSpecular.rgb,
                        material.specularColorFactorSpecular.w);
  if (texId != kInvalidTextureBindlessIndex) {
    specGloss *=
        textureBindless2D(texId, samplerIndex, uvSpecularGlossiness);
  }
  specGloss.rgb = max(specGloss.rgb, vec3(0.0));
  specGloss.a = clamp(specGloss.a, 0.0, 1.0);
  return specGloss;
}

void decodeMaterialBaseWorkflow(MaterialGpuData material, uint workflow,
                                vec4 baseColor, vec4 mrSample,
                                vec2 uvSpecular, uint specularTexId,
                                uint specularSampler, vec2 uvSpecularColor,
                                uint specularColorTexId,
                                uint specularColorSampler, float ior,
                                out float metallic, out float roughness,
                                out vec3 f0, out vec3 f90,
                                out vec3 diffuseColor) {
  metallic = 0.0;
  roughness = kBrdfMinRoughness;
  f0 = vec3(0.0);
  f90 = vec3(0.0);
  diffuseColor = vec3(0.0);

  if (workflow == kMaterialWorkflowSpecularGlossiness) {
    vec4 specGloss = sampleMaterialSpecularGlossiness(
        material, uvSpecularColor, specularColorTexId, specularColorSampler);
    roughness = clamp(1.0 - specGloss.a, kBrdfMinRoughness, 1.0);
    f0 = specGloss.rgb;
    f90 = vec3(1.0);
    diffuseColor = baseColor.rgb * (1.0 - max3(f0));
    return;
  }

  metallic = saturate(material.metallicRoughnessOcclusionAlphaCutoff.x *
                      mrSample.b);
  roughness = clamp(material.metallicRoughnessOcclusionAlphaCutoff.y *
                        mrSample.g,
                    kBrdfMinRoughness, 1.0);

  float specularWeight = sampleMaterialSpecularWeight(
      material, uvSpecular, specularTexId, specularSampler);
  vec3 specularColor = sampleMaterialSpecularColor(
      material, uvSpecularColor, specularColorTexId, specularColorSampler);
  vec3 dielectricF0 = vec3(0.0);
  vec3 dielectricF90 = vec3(0.0);
  computeDielectricSpecularTerms(ior, specularColor, specularWeight,
                                 dielectricF0, dielectricF90);
  f0 = mix(dielectricF0, baseColor.rgb, metallic);
  f90 = mix(dielectricF90, vec3(1.0), metallic);
  diffuseColor = mix(baseColor.rgb, vec3(0.0), metallic);
}

float sampleMaterialOcclusion(MaterialGpuData material, uint workflow,
                              vec4 mrSample, uint metallicRoughnessTexId,
                              vec2 uvOcclusion, uint occlusionTexId,
                              uint occlusionSampler) {
  float occlusion = 1.0;
  if (occlusionTexId != kInvalidTextureBindlessIndex) {
    occlusion =
        textureBindless2D(occlusionTexId, occlusionSampler, uvOcclusion).r;
  } else if (workflow == kMaterialWorkflowMetallicRoughness &&
             metallicRoughnessTexId != kInvalidTextureBindlessIndex) {
    occlusion = mrSample.r;
  }
  return occlusion;
}
