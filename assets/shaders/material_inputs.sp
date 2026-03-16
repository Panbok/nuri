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
