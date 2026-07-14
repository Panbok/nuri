#define NURI_OPAQUE_MESHLET_BATCHED 1
#include "BRDF.sp"
#include "meshlet_common.sp"
#include "opaque_meshlet_vertex.sp"

layout(location = 0) in OpaqueMeshletVertex vtx;
layout(location = 7) flat in uint meshletMaterialIndex;

layout(location = 0) out vec4 out_Normal;

void main() {
  const MaterialData material = loadMaterialData(meshletMaterialIndex);
  const PerVertex materialVertex = opaqueMeshletMaterialVertex(vtx);
  const uint materialSampler = pc.frameData.materialSamplerId;
  const uint normalSampler = pc.frameData.materialDataSamplerId;
  const uint alphaMode = materialAlphaMode(material);
  const uint baseColorSampler =
      alphaMode == kAlphaModeMask && pc.frameData.materialCoverageSamplerId !=
                                         kInvalidTextureBindlessIndex
          ? pc.frameData.materialCoverageSamplerId
          : materialSampler;
  const uint baseColorTexId =
      getMaterialTextureIndex(material, kMaterialTextureSlotBaseColor);
  const vec2 baseColorUv =
      transformedUv(material, materialVertex, kMaterialTextureSlotBaseColor);

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
        transformedUv(material, materialVertex, kMaterialTextureSlotNormal);
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
