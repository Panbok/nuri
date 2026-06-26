#include "BRDF.sp"
#include "common.sp"
#include "material_inputs.sp"
#include "material_lighting.sp"

layout(location = 0) in PerVertex vtx;

layout(location = 0) out vec4 out_ShadowInspect;

const float kShadowInspectBlendPackScale = 0.125;

vec3 shadowInspectSurfaceNormal(MaterialData material, PerVertex vertex) {
  vec3 worldNormal = normalize(vertex.worldNormal);
  if (!gl_FrontFacing) {
    worldNormal *= -1.0;
  }

  const uint normalTexId =
      getMaterialTextureIndex(material, kMaterialTextureSlotNormal);
  if (normalTexId == kInvalidTextureBindlessIndex) {
    return worldNormal;
  }

  const vec2 normalUv =
      transformedUv(material, vertex, kMaterialTextureSlotNormal);
  vec3 tangentNormal =
      textureBindless2D(normalTexId, pc.frameData.materialDataSamplerId,
                        normalUv)
              .xyz *
          2.0 -
      1.0;
  tangentNormal.xy *= materialNormalScale(material);
  const float tangentLen = length(tangentNormal);
  tangentNormal =
      tangentLen > 1.0e-6 ? tangentNormal / tangentLen : vec3(0.0, 0.0, 1.0);
  return applyNormalMap(worldNormal, vertex.worldTangent, vertex.worldPos,
                        normalUv, tangentNormal);
}

void main() {
  const MaterialData material = loadMaterialData(pc.materialIndex);
  if (materialAlphaMode(material) == kAlphaModeMask) {
    const uint baseColorTexId =
        getMaterialTextureIndex(material, kMaterialTextureSlotBaseColor);
    const vec2 baseColorUv =
        transformedUv(material, vtx, kMaterialTextureSlotBaseColor);

    vec4 baseColor = material.header.baseColorFactor;
    if (baseColorTexId != kInvalidTextureBindlessIndex) {
      baseColor *= textureBindless2D(
          baseColorTexId, pc.frameData.materialCoverageSamplerId, baseColorUv);
    }

    if (baseColor.a < materialAlphaCutoff(material)) {
      discard;
    }
  }

  if ((pc.frameData.shadowFlags & kShadowFrameFlagEnabled) == 0u) {
    out_ShadowInspect = vec4(0.0, 0.0, 0.0, -1.0);
    return;
  }

  ShadowFrameBuffer shadow = pc.frameData.shadowFrameBuffer;
  uvec4 shadowState = shadow.flagsCascadeCountLightIndex;
  if (shadowState.y == 0u ||
      shadowState.z >= pc.frameData.directionalLightCount) {
    out_ShadowInspect = vec4(0.0, 0.0, 0.0, -1.0);
    return;
  }

  DirectionalLightGpuData light =
      pc.frameData.directionalLightBuffer.lights[shadowState.z];
  vec3 l = normalize(-directionalLightDirection(light));
  const vec3 surfaceNormal = shadowInspectSurfaceNormal(material, vtx);
  const HardShadowInspectResult inspect =
      inspectHardDirectionalShadow(shadow, vtx.worldPos, surfaceNormal, l);
  const float packedCascadeState =
      inspect.valid > 0.5
          ? inspect.cascadeIndexDebug + min(inspect.cascadeBlendDebug, 0.9999) *
                                            kShadowInspectBlendPackScale
          : -1.0;
  out_ShadowInspect = vec4(inspect.receiverDepth, inspect.receiverCompareDepth,
                           inspect.sampledDepth, packedCascadeState);
}
