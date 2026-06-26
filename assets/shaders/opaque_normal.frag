#include "BRDF.sp"
#include "common.sp"

layout(location = 0) in PerVertex vtx;

layout(location = 0) out vec4 out_Normal;

void main() {
  const MaterialData material = loadMaterialData(pc.materialIndex);
  const uint materialSampler = pc.frameData.materialSamplerId;
  const uint normalSampler = pc.frameData.materialDataSamplerId;
  const uint alphaMode = materialAlphaMode(material);
  const uint baseColorSampler =
      alphaMode == kAlphaModeMask && pc.frameData.materialCoverageSamplerId !=
                                         kInvalidSamplerBindlessIndex
          ? pc.frameData.materialCoverageSamplerId
          : materialSampler;
  const uint baseColorTexId =
      getMaterialTextureIndex(material, kMaterialTextureSlotBaseColor);
  const vec2 baseColorUv =
      transformedUv(material, vtx, kMaterialTextureSlotBaseColor);

  vec4 baseColor = material.header.baseColorFactor;
  if (baseColorTexId != kInvalidTextureBindlessIndex) {
    baseColor *=
        textureBindless2D(baseColorTexId, baseColorSampler, baseColorUv);
  }

  const float alphaCutoff =
      material.header.metallicRoughnessOcclusionAlphaCutoff.w;
  if (alphaMode == kAlphaModeMask && baseColor.a < alphaCutoff) {
    discard;
  }

  vec3 worldNormal = normalize(vtx.worldNormal);
  if (!gl_FrontFacing) {
    worldNormal *= -1.0;
  }

  const uint normalTexId =
      getMaterialTextureIndex(material, kMaterialTextureSlotNormal);
  if (normalTexId != kInvalidTextureBindlessIndex) {
    const vec2 normalUv =
        transformedUv(material, vtx, kMaterialTextureSlotNormal);
    vec3 tangentNormal =
        textureBindless2D(normalTexId, normalSampler, normalUv).xyz * 2.0 - 1.0;
    tangentNormal.xy *= materialNormalScale(material);
    float tangentLen = length(tangentNormal);
    tangentNormal =
        tangentLen > 1.0e-6 ? tangentNormal / tangentLen : vec3(0.0, 0.0, 1.0);
    worldNormal = applyNormalMap(worldNormal, vtx.worldTangent, vtx.worldPos,
                                 normalUv, tangentNormal);
  }

  vec3 viewNormal = normalize(mat3(pc.frameData.view) * worldNormal);
  out_Normal = vec4(viewNormal * 0.5 + 0.5, 1.0);
}
