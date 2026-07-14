#define NURI_OPAQUE_MESHLET_BATCHED 1
#include "meshlet_common.sp"
#include "opaque_meshlet_vertex.sp"

layout(location = 0) in OpaqueMeshletVertex vtx;
layout(location = 7) flat in uint meshletMaterialIndex;

void main() {
  const MaterialData material = loadMaterialData(meshletMaterialIndex);
  const PerVertex materialVertex = opaqueMeshletMaterialVertex(vtx);
  const uint alphaMode = materialAlphaMode(material);
  if (alphaMode != kAlphaModeMask) {
    return;
  }

  const uint baseColorTexId =
      getMaterialTextureIndex(material, kMaterialTextureSlotBaseColor);
  vec4 baseColor = material.header.baseColorFactor;
  if (baseColorTexId != kInvalidTextureBindlessIndex) {
    const uint baseColorSampler =
        pc.frameData.materialCoverageSamplerId != kInvalidSamplerBindlessIndex
            ? pc.frameData.materialCoverageSamplerId
            : pc.frameData.materialSamplerId;
    const vec2 baseColorUv =
        transformedUv(material, materialVertex, kMaterialTextureSlotBaseColor);
    baseColor *=
        textureBindless2D(baseColorTexId, baseColorSampler, baseColorUv);
  }

  if (baseColor.a < material.header.metallicRoughnessOcclusionAlphaCutoff.w) {
    discard;
  }
}
