#extension GL_EXT_buffer_reference : require

layout(std430, buffer_reference) readonly buffer FrameDataBuffer {
  mat4 view;
  mat4 proj;
  vec4 cameraPos;
  uint cubemapTexId;
  uint hasCubemap;
  uint irradianceTexId;
  uint prefilteredGgxTexId;
  uint prefilteredCharlieTexId;
  uint brdfLutTexId;
  uint flags;
  uint cubemapSamplerId;
};

const uint kInvalidTextureBindlessIndex = 0xFFFFFFFFu;
const uint kFrameDataFlagHasIblDiffuse = 1u << 0u;
const uint kFrameDataFlagHasIblSpecular = 1u << 1u;
const uint kFrameDataFlagHasIblSheen = 1u << 2u;
const uint kFrameDataFlagHasBrdfLut = 1u << 3u;
const uint kFrameDataFlagOutputLinearToSrgb = 1u << 4u;
const uint kMaterialFeatureMetallicRoughness = 1u << 0u;
const uint kMaterialFeatureSheen = 1u << 1u;
const uint kMaterialFeatureClearcoat = 1u << 2u;
const uint kMaterialTextureSlotBaseColor = 0u;
const uint kMaterialTextureSlotMetallicRoughness = 1u;
const uint kMaterialTextureSlotNormal = 2u;
const uint kMaterialTextureSlotOcclusion = 3u;
const uint kMaterialTextureSlotEmissive = 4u;
const uint kMaterialTextureSlotClearcoat = 5u;
const uint kMaterialTextureSlotClearcoatRoughness = 6u;
const uint kMaterialTextureSlotClearcoatNormal = 7u;
const uint kMaterialTextureSlotSheenColor = 8u;
const uint kMaterialTextureSlotSheenRoughness = 9u;
const uint kMaterialTextureSlotCount = 10u;

struct PackedVertex {
  // CPU packs each vertex into 9 x 32-bit words:
  // 0..2 = position.xyz as raw float bits
  // 3    = uv0 as half2
  // 4..5 = normal packed as snorm16 pairs (xy, then z + pad)
  // 6..7 = tangent packed as snorm16 pairs (xy, then zw)
  // 8    = uv1 as half2
  // Decode uses uintBitsToFloat/unpackHalf2x16/custom snorm16 unpack.
  uint word0;
  uint word1;
  uint word2;
  uint word3;
  uint word4;
  uint word5;
  uint word6;
  uint word7;
  uint word8;
};

layout(std430, buffer_reference) readonly buffer PackedVertexBuffer {
  PackedVertex vertices[];
};

layout(std430, buffer_reference) readonly buffer InstanceCentersPhaseBuffer {
  vec4 values[];
};

layout(std430, buffer_reference) readonly buffer InstanceBaseMatricesBuffer {
  mat4 matrices[];
};

struct MaterialGpuData {
  vec4 baseColorFactor;
  vec4 emissiveFactorNormalScale;
  vec4 metallicRoughnessOcclusionAlphaCutoff;
  vec4 sheenColorFactorWeight;
  vec4 sheenRoughnessClearcoatFactors; // (sheenRoughness, clearcoatFactor, clearcoatRoughness, clearcoatNormalScale)
  // Packed texture slot mapping shared by textureIndices*, textureUvSets*,
  // and textureSamplerIndices*:
  // 0=baseColor -> *0.x, 1=metallicRoughness -> *0.y,
  // 2=normal -> *0.z, 3=occlusion -> *0.w,
  // 4=emissive -> *1.x, 5=clearcoat -> *1.y,
  // 6=clearcoatRoughness -> *1.z, 7=clearcoatNormal -> *1.w,
  // 8=sheenColor -> *2.x, 9=sheenRoughness -> *2.y.
  uvec4 textureIndices0;
  uvec4 textureIndices1;
  uvec4 textureIndices2;
  uvec4 textureUvSets0;
  uvec4 textureUvSets1;
  uvec4 textureUvSets2;
  uvec4 textureSamplerIndices0;
  uvec4 textureSamplerIndices1;
  uvec4 textureSamplerIndices2;
  vec4 textureTransformOffsetScale[kMaterialTextureSlotCount];
  vec4 textureTransformRotation[kMaterialTextureSlotCount];
  uvec4 materialFlags; // Full std430 slot: x=alphaMode, y=doubleSided, z=featureMask, w=reserved
};

uint getPackedMaterialSlotValue(uvec4 packed0, uvec4 packed1, uvec4 packed2,
                                uint slot, uint defaultValue) {
  if (slot < 4u) {
    return packed0[int(slot)];
  }
  if (slot < 8u) {
    return packed1[int(slot - 4u)];
  }
  if (slot < kMaterialTextureSlotCount) {
    return packed2[int(slot - 8u)];
  }
  return defaultValue;
}

#define GET_TEXTURE_INDEX(material, slot)                                      \
  getPackedMaterialSlotValue((material).textureIndices0,                       \
                             (material).textureIndices1,                       \
                             (material).textureIndices2, (slot),               \
                             kInvalidTextureBindlessIndex)
#define GET_UV_SET(material, slot)                                             \
  getPackedMaterialSlotValue((material).textureUvSets0,                        \
                             (material).textureUvSets1,                        \
                             (material).textureUvSets2, (slot), 0u)
#define GET_SAMPLER_INDEX(material, slot)                                      \
  getPackedMaterialSlotValue((material).textureSamplerIndices0,                \
                             (material).textureSamplerIndices1,                \
                             (material).textureSamplerIndices2, (slot), 0u)

layout(std430, buffer_reference) readonly buffer MaterialBuffer {
  MaterialGpuData materials[];
};

layout(std430, buffer_reference) readonly buffer InstanceRemapBuffer {
  uint ids[];
};

layout(std430, buffer_reference) buffer InstanceMatricesBuffer {
  mat4 matrices[];
};

layout(push_constant) uniform PushConstants {
  FrameDataBuffer frameData;
  PackedVertexBuffer vertexBuffer;
  InstanceMatricesBuffer instanceMatrices;
  InstanceRemapBuffer instanceRemap;
  MaterialBuffer materialBuffer;
  InstanceCentersPhaseBuffer instanceCentersPhase;
  InstanceBaseMatricesBuffer instanceBaseMatrices;
  uint instanceCount;
  uint materialIndex;
  float timeSeconds;
  float tessNearDistance;
  float tessFarDistance;
  float tessMinFactor;
  float tessMaxFactor;
  uint debugVisualizationMode;
} pc;

const uint kDebugVisualizationNone = 0u;
const uint kDebugVisualizationWireOverlay = 1u;
const uint kDebugVisualizationWireframeOnly = 2u;
const uint kDebugVisualizationTessPatchEdgesHeatmap = 3u;

vec2 unpackSnorm2x16Custom(uint packed) {
  const int x = int(packed << 16u) >> 16;
  const int y = int(packed) >> 16;
  return clamp(vec2(float(x), float(y)) / 32767.0, vec2(-1.0), vec2(1.0));
}

vec3 decodePackedPosition(PackedVertex vertex) {
  return vec3(uintBitsToFloat(vertex.word0), uintBitsToFloat(vertex.word1),
              uintBitsToFloat(vertex.word2));
}

vec2 decodePackedUv(PackedVertex vertex) { return unpackHalf2x16(vertex.word3); }
vec2 decodePackedUv1(PackedVertex vertex) { return unpackHalf2x16(vertex.word8); }

vec2 applyTextureTransform(vec2 uv, vec4 offsetScale, vec4 rotationCs) {
  vec2 scaled = uv * offsetScale.zw;
  vec2 rotated =
      vec2(rotationCs.x * scaled.x - rotationCs.y * scaled.y,
           rotationCs.y * scaled.x + rotationCs.x * scaled.y);
  return offsetScale.xy + rotated;
}

vec3 decodePackedNormal(PackedVertex vertex) {
  const vec2 normalXY = unpackSnorm2x16Custom(vertex.word4);
  const vec2 normalZ = unpackSnorm2x16Custom(vertex.word5);
  return normalize(vec3(normalXY, normalZ.x));
}

vec4 decodePackedTangent(PackedVertex vertex) {
  const vec2 tangentXY = unpackSnorm2x16Custom(vertex.word6);
  const vec2 tangentZW = unpackSnorm2x16Custom(vertex.word7);
  vec3 tangent = vec3(tangentXY, tangentZW.x);
  const float tangentLen = length(tangent);
  if (tangentLen > 1.0e-6) {
    tangent /= tangentLen;
  } else {
    tangent = vec3(1.0, 0.0, 0.0);
  }
  float handedness = tangentZW.y >= 0.0 ? 1.0 : -1.0;
  return vec4(tangent, handedness);
}

struct PerVertex {
  vec2 uv0;
  vec2 uv1;
  vec3 worldNormal;
  vec4 worldTangent;
  vec3 worldPos;
  vec3 patchBarycentric;
  vec3 triBarycentric;
  vec3 patchOuterFactors;
  float patchInnerFactor;
  float tessellatedFlag;
};
