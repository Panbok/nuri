#extension GL_EXT_buffer_reference : require

const uint kInvalidTextureBindlessIndex = 0xFFFFFFFFu;
const uint kFrameDataFlagHasIblDiffuse = 1u << 0u;
const uint kFrameDataFlagHasIblSpecular = 1u << 1u;
const uint kFrameDataFlagHasIblSheen = 1u << 2u;
const uint kFrameDataFlagHasBrdfLut = 1u << 3u;
const uint kFrameDataFlagOutputLinearToSrgb = 1u << 4u;
const uint kFrameDataFlagHasSceneColor = 1u << 5u;
const uint kMaterialFeatureMetallicRoughness = 1u << 0u;
const uint kMaterialFeatureSheen = 1u << 1u;
const uint kMaterialFeatureClearcoat = 1u << 2u;
const uint kMaterialFeatureTransmission = 1u << 3u;
const uint kMaterialFeatureVolume = 1u << 4u;
const uint kMaterialFeatureSpecular = 1u << 5u;
const uint kMaterialTextureSlotBaseColor = 0u;
const uint kMaterialTextureSlotMetallicRoughness = 1u;
const uint kMaterialTextureSlotNormal = 2u;
const uint kMaterialTextureSlotOcclusion = 3u;
const uint kMaterialTextureSlotEmissive = 4u;
const uint kMaterialTextureSlotClearcoat = 5u;
const uint kMaterialTextureSlotClearcoatRoughness = 6u;
const uint kMaterialTextureSlotClearcoatNormal = 7u;
const uint kMaterialTextureSlotSpecular = 8u;
const uint kMaterialTextureSlotSpecularColor = 9u;
const uint kMaterialTextureSlotSheenColor = 10u;
const uint kMaterialTextureSlotSheenRoughness = 11u;
const uint kMaterialTextureSlotTransmission = 12u;
const uint kMaterialTextureSlotThickness = 13u;
const uint kMaterialTextureSlotCount = 14u;

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
  vec4 emissiveFactorStrength; // (emissiveFactor.rgb, emissiveStrength)
  vec4 metallicRoughnessOcclusionAlphaCutoff;
  vec4 specularColorFactorSpecular;
  vec4 sheenColorFactorWeight;
  vec4 sheenRoughnessClearcoatFactors; // (sheenRoughness, clearcoatFactor, clearcoatRoughness, clearcoatNormalScale)
  vec4 transmissionThicknessIorPadding; // (transmissionFactor, thicknessFactor, ior, normalScale)
  vec4 attenuationColorDistance; // (attenuationColor.rgb, attenuationDistance)
  // Packed texture slot mapping shared by textureIndices*, textureUvSets*,
  // and textureSamplerIndices*:
  // 0=baseColor -> *0.x, 1=metallicRoughness -> *0.y,
  // 2=normal -> *0.z, 3=occlusion -> *0.w,
  // 4=emissive -> *1.x, 5=clearcoat -> *1.y,
  // 6=clearcoatRoughness -> *1.z, 7=clearcoatNormal -> *1.w,
  // 8=specular -> *2.x, 9=specularColor -> *2.y,
  // 10=sheenColor -> *2.z, 11=sheenRoughness -> *2.w,
  // 12=transmission -> *3.x, 13=thickness -> *3.y.
  uvec4 textureIndices0;
  uvec4 textureIndices1;
  uvec4 textureIndices2;
  uvec4 textureIndices3;
  uvec4 textureUvSets0;
  uvec4 textureUvSets1;
  uvec4 textureUvSets2;
  uvec4 textureUvSets3;
  uvec4 textureSamplerIndices0;
  uvec4 textureSamplerIndices1;
  uvec4 textureSamplerIndices2;
  uvec4 textureSamplerIndices3;
  vec4 textureTransformOffsetScale[kMaterialTextureSlotCount];
  vec4 textureTransformRotation[kMaterialTextureSlotCount];
  uvec4 materialFlags; // Full std430 slot: x=alphaMode, y=doubleSided, z=featureMask, w=reserved
};

uint getPackedMaterialSlotValue(uvec4 packed0, uvec4 packed1, uvec4 packed2,
                                uvec4 packed3, uint slot,
                                uint defaultValue) {
  if (slot < 4u) {
    return packed0[int(slot)];
  }
  if (slot < 8u) {
    return packed1[int(slot - 4u)];
  }
  if (slot < 12u) {
    return packed2[int(slot - 8u)];
  }
  if (slot < kMaterialTextureSlotCount) {
    return packed3[int(slot - 12u)];
  }
  return defaultValue;
}

float materialEmissiveStrength(MaterialGpuData material) {
  return material.emissiveFactorStrength.w;
}

float materialNormalScale(MaterialGpuData material) {
  return material.transmissionThicknessIorPadding.w;
}

#define GET_TEXTURE_INDEX(material, slot)                                      \
  getPackedMaterialSlotValue((material).textureIndices0,                       \
                             (material).textureIndices1,                       \
                             (material).textureIndices2,                       \
                             (material).textureIndices3, (slot),               \
                             kInvalidTextureBindlessIndex)
#define GET_UV_SET(material, slot)                                             \
  getPackedMaterialSlotValue((material).textureUvSets0,                        \
                             (material).textureUvSets1,                        \
                             (material).textureUvSets2,                        \
                             (material).textureUvSets3, (slot), 0u)
#define GET_SAMPLER_INDEX(material, slot)                                      \
  getPackedMaterialSlotValue((material).textureSamplerIndices0,                \
                             (material).textureSamplerIndices1,                \
                             (material).textureSamplerIndices2,                \
                             (material).textureSamplerIndices3, (slot), 0u)

layout(std430, buffer_reference) readonly buffer MaterialBuffer {
  MaterialGpuData materials[];
};

struct DirectionalLightGpuData {
  vec4 directionIlluminance;
  vec4 colorReserved;
};

struct LocalLightGpuData {
  vec4 positionRange;
  vec4 directionOuterCos;
  vec4 colorIntensity;
  uvec4 innerCosTypeEnabledReserved;
};

layout(std430, buffer_reference) readonly buffer DirectionalLightBuffer {
  DirectionalLightGpuData lights[];
};

layout(std430, buffer_reference) readonly buffer LocalLightBuffer {
  LocalLightGpuData lights[];
};

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
  uint sceneColorTexId;
  uint sceneColorSamplerId;
  uint sceneColorHalfResTexId;
  uint sceneColorQuarterResTexId;
  DirectionalLightBuffer directionalLightBuffer;
  LocalLightBuffer localLightBuffer;
  uint directionalLightCount;
  uint localLightCount;
};

uint getSceneColorPyramidTexId(FrameDataBuffer frameData, uint level) {
  if (level == 0u) {
    return frameData.sceneColorTexId;
  }
  if (level == 1u) {
    return frameData.sceneColorHalfResTexId;
  }
  if (level == 2u) {
    return frameData.sceneColorQuarterResTexId;
  }
  return kInvalidTextureBindlessIndex;
}

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
const uint kLocalLightTypePoint = 0u;
const uint kLocalLightTypeSpot = 1u;

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

vec2 selectUv(vec2 uv0, vec2 uv1, uint uvSet) {
  return (uvSet == 1u) ? uv1 : uv0;
}

vec3 directionalLightDirection(DirectionalLightGpuData light) {
  return light.directionIlluminance.xyz;
}

float directionalLightIlluminance(DirectionalLightGpuData light) {
  return light.directionIlluminance.w;
}

vec3 directionalLightColor(DirectionalLightGpuData light) {
  return light.colorReserved.rgb;
}

vec3 localLightPosition(LocalLightGpuData light) { return light.positionRange.xyz; }

float localLightRange(LocalLightGpuData light) { return light.positionRange.w; }

vec3 localLightDirection(LocalLightGpuData light) {
  return light.directionOuterCos.xyz;
}

float localLightOuterCos(LocalLightGpuData light) {
  return light.directionOuterCos.w;
}

vec3 localLightColor(LocalLightGpuData light) { return light.colorIntensity.rgb; }

float localLightIntensity(LocalLightGpuData light) {
  return light.colorIntensity.w;
}

float localLightInnerCos(LocalLightGpuData light) {
  return uintBitsToFloat(light.innerCosTypeEnabledReserved.x);
}

uint localLightType(LocalLightGpuData light) {
  return light.innerCosTypeEnabledReserved.y;
}

bool localLightEnabled(LocalLightGpuData light) {
  return light.innerCosTypeEnabledReserved.z != 0u;
}

float punctualRangeAttenuation(float distanceSq, float range) {
  const float epsilon = 1.0e-6;
  if (range <= 0.0) {
    return 1.0 / max(distanceSq, epsilon);
  }
  float distance = sqrt(max(distanceSq, epsilon));
  float normalizedDistance = distance / max(range, epsilon);
  float window = clamp(1.0 - pow(normalizedDistance, 4.0), 0.0, 1.0);
  return (window * window) / max(distanceSq, epsilon);
}

float spotAngularAttenuation(vec3 lightDirection, vec3 pointToLight,
                             float innerCos, float outerCos) {
  float cosTheta = dot(normalize(lightDirection), normalize(-pointToLight));
  if (cosTheta <= outerCos) {
    return 0.0;
  }
  float denom = max(innerCos - outerCos, 1.0e-4);
  float weight = clamp((cosTheta - outerCos) / denom, 0.0, 1.0);
  return weight * weight;
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

vec2 transformedUv(MaterialGpuData material, PerVertex vertex, uint slot) {
  return applyTextureTransform(selectUv(vertex.uv0, vertex.uv1,
                                        GET_UV_SET(material, slot)),
                               material.textureTransformOffsetScale[slot],
                               material.textureTransformRotation[slot]);
}
