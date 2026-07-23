#define NURI_OPAQUE_MESHLET_BATCHED 1
#include "meshlet_common.sp"
#include "opaque_meshlet_vertex.sp"

layout(location = 0) in OpaqueMeshletVertex vtx;
layout(location = 7) flat in uint meshletMaterialIndex;

layout(location = 0) out vec4 out_Coverage;

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

  // The MSAA depth prepass must produce the same coverage mask as the shaded
  // alpha pipeline. Otherwise depth-only samples block later opaque geometry
  // while the shaded pass leaves those color samples untouched.
  out_Coverage = vec4(0.0, 0.0, 0.0, baseColor.a);
}
