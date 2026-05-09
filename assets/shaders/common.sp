#extension GL_EXT_buffer_reference : require

const uint kInvalidTextureBindlessIndex = 0xFFFFFFFFu;
const uint kInvalidSamplerBindlessIndex = 0xFFFFFFFFu;
const uint kAlphaModeOpaque = 0u;
const uint kAlphaModeMask = 1u;
const uint kAlphaModeBlend = 2u;
const uint kInvalidMaterialExtensionIndex = 0xFFFFFFFFu;

const uint kFrameDataFlagHasIblDiffuse = 1u << 0u;
const uint kFrameDataFlagHasIblSpecular = 1u << 1u;
const uint kFrameDataFlagHasIblSheen = 1u << 2u;
const uint kFrameDataFlagHasBrdfLut = 1u << 3u;
const uint kFrameDataFlagOutputLinearToSrgb = 1u << 4u;
const uint kFrameDataFlagHasSceneColor = 1u << 5u;
const uint kFrameDataFlagHasSceneDepth = 1u << 6u;
const uint kFrameDataFlagHasSceneDepthPyramid = 1u << 7u;
const uint kFrameDataFlagTransmissionMipDebug = 1u << 8u;
const uint kFrameDataFlagHasAmbientOcclusion = 1u << 9u;
const uint kFrameDataFlagHasAmbientBentNormal = 1u << 10u;
const uint kAmbientOcclusionFlagScalarAo = 1u << 0u;
const uint kAmbientOcclusionFlagBentNormal = 1u << 1u;
const uint kAmbientOcclusionDebugViewShift = 8u;
const uint kAmbientOcclusionDebugViewMask = 0xFFu;
const uint kAmbientOcclusionDebugViewNone = 0u;
const uint kAmbientOcclusionDebugViewVisibility = 1u;
const uint kAmbientOcclusionDebugViewBentNormal = 2u;
const uint kAmbientOcclusionDebugViewNormals = 3u;
const uint kMaxSceneDepthPyramidLevels = 16u;
const uint kSceneDepthPyramidTexIdPackWidth = 4u;
const uint kSceneDepthPyramidArraySize =
    (kMaxSceneDepthPyramidLevels + kSceneDepthPyramidTexIdPackWidth - 1u) /
    kSceneDepthPyramidTexIdPackWidth;
const uint kMaxShadowCascades = 4u;
const uint kMaxShadowPcfSamples = 64u;
const uint kShadowFrameFlagEnabled = 1u << 0u;
const uint kShadowFrameFlagVisualizeShadowFactor = 1u << 1u;
const uint kShadowFrameFlagVisualizeCascadeIndex = 1u << 2u;
const uint kShadowFrameFlagVisualizePCSSBlockers = 1u << 3u;
const uint kShadowFrameFlagFixedPoissonRotation = 1u << 4u;
const uint kShadowFrameFlagVisualizePCFResult = 1u << 5u;
const uint kShadowFrameFlagVisualizeReceiverDepth = 1u << 6u;
const uint kShadowFrameFlagVisualizeShadowMapDepth = 1u << 7u;
const uint kShadowFrameFlagVisualizePCSSAverageBlockerDepth = 1u << 8u;
const uint kShadowFrameFlagVisualizePCSSFilterRadius = 1u << 9u;
const uint kShadowFilterModeHard = 0u;
const uint kShadowFilterModePCF3x3 = 1u;
const uint kShadowFilterModePoissonPCF = 2u;
const uint kShadowFilterModePCSS = 3u;

const uint kMaterialFeatureMetallicRoughness = 1u << 0u;
const uint kMaterialFeatureSheen = 1u << 1u;
const uint kMaterialFeatureClearcoat = 1u << 2u;
const uint kMaterialFeatureTransmission = 1u << 3u;
const uint kMaterialFeatureVolume = 1u << 4u;
const uint kMaterialFeatureSpecular = 1u << 5u;

const uint kMaterialWorkflowMetallicRoughness = 0u;
const uint kMaterialWorkflowSpecularGlossiness = 1u;

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

const uint kPackedVertexFormatStaticQuantized20 = 0u;
const uint kPackedVertexFormatAnimatedFloat24 = 1u;
const uint kPackedVertexFormatAnimatedFloat32 = 2u;

const uint kMaterialFlagsAlphaModeMask = 0x3u;
const uint kMaterialFlagsDoubleSidedBit = 1u << 2u;
const uint kMaterialFlagsWorkflowShift = 8u;
const uint kMaterialFlagsFeatureShift = 16u;

layout(std430, buffer_reference) readonly buffer PackedVertexWordBuffer {
  uint words[];
};

layout(std430, buffer_reference) readonly buffer InstanceCentersPhaseBuffer {
  vec4 values[];
};

layout(std430, buffer_reference) readonly buffer InstanceBaseMatricesBuffer {
  mat4 matrices[];
};

struct PackedMaterialTransformGpuData {
  uint offsetXY;
  uint scaleXY;
  uint rotCS;
};

struct MaterialHeaderGpuData {
  vec4 baseColorFactor;
  vec4 emissiveFactorStrength;
  vec4 metallicRoughnessOcclusionAlphaCutoff;
  vec4 normalScaleIorReserved;
  uvec4 commonTextureIndices;
  uint emissiveTextureIndex;
  uint uvSetBits;
  uint materialFlags;
  uint clearcoatExtensionIndex;
  uint sheenExtensionIndex;
  uint transmissionExtensionIndex;
  uint specularExtensionIndex;
  uint reserved0;
  uint reserved1;
  uint reserved2;
  PackedMaterialTransformGpuData commonTransforms[5];
};

struct MaterialClearcoatGpuData {
  vec4 clearcoatFactors;
  uvec4 textureIndices;
  PackedMaterialTransformGpuData transforms[3];
  uint uvSetBits;
  uint reserved0;
  uint reserved1;
};

struct MaterialSheenGpuData {
  vec4 sheenColorFactorWeight;
  vec4 sheenRoughnessReserved;
  uvec4 textureIndices;
  PackedMaterialTransformGpuData transforms[2];
  uint uvSetBits;
  uint reserved0;
};

struct MaterialTransmissionGpuData {
  vec4 transmissionThicknessDistance;
  vec4 attenuationColorReserved;
  uvec4 textureIndices;
  PackedMaterialTransformGpuData transforms[2];
  uint uvSetBits;
  uint reserved0;
};

struct MaterialSpecularGpuData {
  vec4 specularColorFactorSpecular;
  uvec4 textureIndices;
  PackedMaterialTransformGpuData transforms[2];
  uint uvSetBits;
  uint reserved0;
};

layout(std430, buffer_reference) readonly buffer MaterialHeaderBuffer {
  MaterialHeaderGpuData materials[];
};

layout(std430, buffer_reference) readonly buffer MaterialClearcoatBuffer {
  MaterialClearcoatGpuData materials[];
};

layout(std430, buffer_reference) readonly buffer MaterialSheenBuffer {
  MaterialSheenGpuData materials[];
};

layout(std430, buffer_reference) readonly buffer MaterialTransmissionBuffer {
  MaterialTransmissionGpuData materials[];
};

layout(std430, buffer_reference) readonly buffer MaterialSpecularBuffer {
  MaterialSpecularGpuData materials[];
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

struct ShadowCascadeGpuData {
  mat4 lightViewProj;
  mat4 lightView;
  vec4 splitDepthTexelSize;
  vec4 uvScaleBias;
  vec4 biasParams;
  // x: PCSS receiver-depth world scale, y/z: search/filter radius clamps.
  vec4 pcssParams;
  // x: depth texture, y: compare sampler, z: raw sampler, w: square map size.
  uvec4 textureSampler;
};

layout(std430, buffer_reference) readonly buffer DirectionalLightBuffer {
  DirectionalLightGpuData lights[];
};

layout(std430, buffer_reference) readonly buffer LocalLightBuffer {
  LocalLightGpuData lights[];
};

layout(std430, buffer_reference) readonly buffer ShadowFrameBuffer {
  uvec4 flagsCascadeCountLightIndex;
  vec4 fadeParams;
  uvec4 filterParams;
  ShadowCascadeGpuData cascades[kMaxShadowCascades];
};

layout(std430, buffer_reference) readonly buffer FrameDataBuffer {
  mat4 view;
  // Current scene projection. Temporal AA applies jitter here when enabled;
  // CPU CameraFrameState retains unjittered matrices for overlay and velocity use.
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
  uint materialSamplerId;
  uint sceneColorTexId;
  uint sceneColorSamplerId;
  uint sceneColorHalfResTexId;
  uint sceneColorQuarterResTexId;
  uint sceneDepthTexId;
  uint sceneDepthSamplerId;
  uint sceneDepthPyramidLevelCount;
  uvec4 sceneDepthPyramidTexIds[kSceneDepthPyramidArraySize];
  uint ambientOcclusionTexId;
  uint ambientOcclusionSamplerId;
  uint ambientOcclusionFlags;
  uint ambientOcclusionReserved0;
  DirectionalLightBuffer directionalLightBuffer;
  LocalLightBuffer localLightBuffer;
  MaterialHeaderBuffer materialHeaderBuffer;
  MaterialClearcoatBuffer materialClearcoatBuffer;
  MaterialSheenBuffer materialSheenBuffer;
  MaterialTransmissionBuffer materialTransmissionBuffer;
  MaterialSpecularBuffer materialSpecularBuffer;
  uint directionalLightCount;
  uint localLightCount;
  ShadowFrameBuffer shadowFrameBuffer;
  uint shadowFlags;
  // Base-color texture indices remain per-material/per-draw; only the coverage
  // sampler is frame-scoped so alpha-test passes can bypass TAA material mip
  // bias.
  uint materialCoverageSamplerId;
};

uint getAmbientOcclusionDebugView(FrameDataBuffer frameData) {
  return (frameData.ambientOcclusionFlags >> kAmbientOcclusionDebugViewShift) &
         kAmbientOcclusionDebugViewMask;
}

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

uint getSceneDepthPyramidTexId(FrameDataBuffer frameData, uint level) {
  if (level >= frameData.sceneDepthPyramidLevelCount ||
      level >= kMaxSceneDepthPyramidLevels) {
    return kInvalidTextureBindlessIndex;
  }
  return frameData.sceneDepthPyramidTexIds[
      level / kSceneDepthPyramidTexIdPackWidth]
      [level % kSceneDepthPyramidTexIdPackWidth];
}

layout(std430, buffer_reference) readonly buffer InstanceRemapBuffer {
  uint ids[];
};

struct InstanceData {
  mat4 modelMatrix;
  vec4 normalMatCol0;
  vec4 normalMatCol1;
  vec4 normalMatCol2;
};

layout(std430, buffer_reference) buffer InstanceMatricesBuffer {
  InstanceData instances[];
};

layout(std430, buffer_reference) readonly buffer ReadonlyInstanceMatricesBuffer {
  InstanceData instances[];
};

layout(std430, buffer_reference) readonly buffer VelocityInstanceFlagsBuffer {
  uint flags[];
};

const uint kVelocityInstanceFlagsModeBuffer = 0u;
const uint kVelocityInstanceFlagsModeAllValid = 1u;
const uint kVelocityInstanceFlagsModeAllInvalid = 2u;

struct VelocityFrameData {
  mat4 currentViewProjNoJitter;
  mat4 previousViewProjNoJitter;
  uvec4 instanceFlagsMode;
};

layout(std430, buffer_reference) readonly buffer VelocityFrameDataBuffer {
  VelocityFrameData data;
};

struct StaticVertexDecodeGpuData {
  vec4 offset;
  vec4 scale;
};

layout(std430, buffer_reference) readonly buffer StaticVertexDecodeBuffer {
  StaticVertexDecodeGpuData values[];
};

layout(push_constant) uniform PushConstants {
  FrameDataBuffer frameData;
  PackedVertexWordBuffer vertexBuffer;
  StaticVertexDecodeBuffer vertexDecodeBuffer;
  InstanceMatricesBuffer instanceMatrices;
  ReadonlyInstanceMatricesBuffer previousInstanceMatrices;
  InstanceRemapBuffer instanceRemap;
  InstanceCentersPhaseBuffer instanceCentersPhase;
  InstanceBaseMatricesBuffer instanceBaseMatrices;
  VelocityInstanceFlagsBuffer velocityInstanceFlags;
  VelocityFrameDataBuffer velocityFrameData;
  uint instanceCount;
  uint materialIndex;
  uint vertexDecodeIndex;
  uint packedVertexFormat;
  float timeSeconds;
  float tessNearDistance;
  float tessFarDistance;
  float tessMinFactor;
  float tessMaxFactor;
  uint debugVisualizationMode;
  uint shadowCascadeIndex;
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

vec2 signNotZero(vec2 value) {
  return vec2(value.x >= 0.0 ? 1.0 : -1.0, value.y >= 0.0 ? 1.0 : -1.0);
}

vec3 decodeOctNormal(vec2 encoded) {
  vec3 normal = vec3(encoded.xy, 1.0 - abs(encoded.x) - abs(encoded.y));
  if (normal.z < 0.0) {
    normal.xy = (vec2(1.0) - abs(normal.yx)) * signNotZero(normal.xy);
  }
  const float lenSq = dot(normal, normal);
  if (lenSq <= 1.0e-10) {
    return vec3(0.0, 1.0, 0.0);
  }
  return normalize(normal);
}

uint packedVertexStrideWords(uint packedVertexFormat) {
  if (packedVertexFormat == kPackedVertexFormatAnimatedFloat32) {
    return 8u;
  }
  return packedVertexFormat == kPackedVertexFormatAnimatedFloat24 ? 6u : 5u;
}

uint packedVertexWord(uint vertexIndex, uint wordIndex) {
  return pc.vertexBuffer
      .words[vertexIndex * packedVertexStrideWords(pc.packedVertexFormat) +
             wordIndex];
}

vec3 decodePackedPosition(uint vertexIndex) {
  if (pc.packedVertexFormat == kPackedVertexFormatAnimatedFloat24 ||
      pc.packedVertexFormat == kPackedVertexFormatAnimatedFloat32) {
    return vec3(uintBitsToFloat(packedVertexWord(vertexIndex, 0u)),
                uintBitsToFloat(packedVertexWord(vertexIndex, 1u)),
                uintBitsToFloat(packedVertexWord(vertexIndex, 2u)));
  }

  StaticVertexDecodeGpuData decode =
      pc.vertexDecodeBuffer.values[pc.vertexDecodeIndex];
  const vec2 xy = unpackUnorm2x16(packedVertexWord(vertexIndex, 0u));
  const vec2 zPad = unpackUnorm2x16(packedVertexWord(vertexIndex, 1u));
  const vec3 normalized = vec3(xy, zPad.x);
  return decode.offset.xyz + decode.scale.xyz * normalized;
}

vec3 decodePackedNormal(uint vertexIndex) {
  const bool animatedFormat =
      pc.packedVertexFormat == kPackedVertexFormatAnimatedFloat24 ||
      pc.packedVertexFormat == kPackedVertexFormatAnimatedFloat32;
  const uint normalWord = packedVertexWord(vertexIndex,
                                           animatedFormat ? 3u : 2u);
  return decodeOctNormal(unpackSnorm2x16Custom(normalWord));
}

vec4 decodePackedTangent(uint vertexIndex) {
  const vec3 normal = decodePackedNormal(vertexIndex);
  const vec3 tangentHelper =
      abs(normal.x) < 0.999 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
  const vec4 fallbackTangent =
      vec4(normalize(cross(tangentHelper, normal)), 1.0);
  if (pc.packedVertexFormat != kPackedVertexFormatAnimatedFloat32) {
    return fallbackTangent;
  }
  const uint handednessWord = packedVertexWord(vertexIndex, 5u);
  // word5 stores +/-1.0f handedness; 0u is the "no tangent encoded"
  // sentinel, so return an orthogonal fallback to keep TBN construction valid.
  if (handednessWord == 0u) {
    return fallbackTangent;
  }
  return vec4(decodeOctNormal(unpackSnorm2x16Custom(
                  packedVertexWord(vertexIndex, 4u))),
              uintBitsToFloat(handednessWord));
}

vec2 decodePackedUv(uint vertexIndex) {
  if (pc.packedVertexFormat == kPackedVertexFormatAnimatedFloat32) {
    return unpackHalf2x16(packedVertexWord(vertexIndex, 6u));
  }
  return unpackHalf2x16(
      packedVertexWord(vertexIndex,
                       pc.packedVertexFormat == kPackedVertexFormatAnimatedFloat24
                           ? 4u
                           : 3u));
}

vec2 decodePackedUv1(uint vertexIndex) {
  if (pc.packedVertexFormat == kPackedVertexFormatAnimatedFloat32) {
    return unpackHalf2x16(packedVertexWord(vertexIndex, 7u));
  }
  return unpackHalf2x16(
      packedVertexWord(vertexIndex,
                       pc.packedVertexFormat == kPackedVertexFormatAnimatedFloat24
                           ? 5u
                           : 4u));
}

uint getPackedUvBit(uint uvBits, uint bitIndex) { return (uvBits >> bitIndex) & 1u; }

vec2 applyTextureTransform(vec2 uv, PackedMaterialTransformGpuData transform) {
  vec2 offset = unpackHalf2x16(transform.offsetXY);
  vec2 scale = unpackHalf2x16(transform.scaleXY);
  vec2 rotationCs = unpackSnorm2x16Custom(transform.rotCS);
  vec2 scaled = uv * scale;
  vec2 rotated = vec2(rotationCs.x * scaled.x - rotationCs.y * scaled.y,
                      rotationCs.y * scaled.x + rotationCs.x * scaled.y);
  return offset + rotated;
}

vec2 selectUv(vec2 uv0, vec2 uv1, uint uvSet) {
  return uvSet == 1u ? uv1 : uv0;
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

float directionalLightAngularRadiusRadians(DirectionalLightGpuData light) {
  return max(light.colorReserved.w, 0.0);
}

vec3 localLightPosition(LocalLightGpuData light) { return light.positionRange.xyz; }
float localLightRange(LocalLightGpuData light) { return light.positionRange.w; }
vec3 localLightDirection(LocalLightGpuData light) {
  return light.directionOuterCos.xyz;
}
float localLightOuterCos(LocalLightGpuData light) { return light.directionOuterCos.w; }
vec3 localLightColor(LocalLightGpuData light) { return light.colorIntensity.rgb; }
float localLightIntensity(LocalLightGpuData light) { return light.colorIntensity.w; }
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
  float cosTheta = dot(lightDirection, normalize(-pointToLight));
  if (cosTheta <= outerCos) {
    return 0.0;
  }
  float denom = max(innerCos - outerCos, 1.0e-4);
  float weight = clamp((cosTheta - outerCos) / denom, 0.0, 1.0);
  return weight * weight;
}

struct MaterialData {
  MaterialHeaderGpuData header;
  MaterialClearcoatGpuData clearcoat;
  MaterialSheenGpuData sheen;
  MaterialTransmissionGpuData transmission;
  MaterialSpecularGpuData specular;
  bool hasClearcoat;
  bool hasSheen;
  bool hasTransmission;
  bool hasSpecular;
};

PackedMaterialTransformGpuData defaultMaterialTransformData() {
  return PackedMaterialTransformGpuData(0u, 0u, 0u);
}

MaterialClearcoatGpuData defaultMaterialClearcoatData() {
  MaterialClearcoatGpuData data;
  data.clearcoatFactors = vec4(0.0, 0.0, 1.0, 0.0);
  data.textureIndices = uvec4(kInvalidTextureBindlessIndex,
                              kInvalidTextureBindlessIndex,
                              kInvalidTextureBindlessIndex, 0u);
  data.transforms[0] = defaultMaterialTransformData();
  data.transforms[1] = defaultMaterialTransformData();
  data.transforms[2] = defaultMaterialTransformData();
  data.uvSetBits = 0u;
  data.reserved0 = 0u;
  data.reserved1 = 0u;
  return data;
}

MaterialSheenGpuData defaultMaterialSheenData() {
  MaterialSheenGpuData data;
  data.sheenColorFactorWeight = vec4(0.0);
  data.sheenRoughnessReserved = vec4(0.0);
  data.textureIndices =
      uvec4(kInvalidTextureBindlessIndex, kInvalidTextureBindlessIndex, 0u, 0u);
  data.transforms[0] = defaultMaterialTransformData();
  data.transforms[1] = defaultMaterialTransformData();
  data.uvSetBits = 0u;
  data.reserved0 = 0u;
  return data;
}

MaterialTransmissionGpuData defaultMaterialTransmissionData() {
  MaterialTransmissionGpuData data;
  data.transmissionThicknessDistance = vec4(0.0);
  data.attenuationColorReserved = vec4(1.0, 1.0, 1.0, 0.0);
  data.textureIndices =
      uvec4(kInvalidTextureBindlessIndex, kInvalidTextureBindlessIndex, 0u, 0u);
  data.transforms[0] = defaultMaterialTransformData();
  data.transforms[1] = defaultMaterialTransformData();
  data.uvSetBits = 0u;
  data.reserved0 = 0u;
  return data;
}

MaterialSpecularGpuData defaultMaterialSpecularData() {
  MaterialSpecularGpuData data;
  data.specularColorFactorSpecular = vec4(1.0);
  data.textureIndices =
      uvec4(kInvalidTextureBindlessIndex, kInvalidTextureBindlessIndex, 0u, 0u);
  data.transforms[0] = defaultMaterialTransformData();
  data.transforms[1] = defaultMaterialTransformData();
  data.uvSetBits = 0u;
  data.reserved0 = 0u;
  return data;
}

MaterialData loadMaterialDataCore(uint materialIndex, bool includeTransmission) {
  MaterialData material;
  material.clearcoat = defaultMaterialClearcoatData();
  material.sheen = defaultMaterialSheenData();
  material.transmission = defaultMaterialTransmissionData();
  material.specular = defaultMaterialSpecularData();
  material.header = pc.frameData.materialHeaderBuffer.materials[materialIndex];
  material.hasClearcoat =
      material.header.clearcoatExtensionIndex != kInvalidMaterialExtensionIndex;
  material.hasSheen =
      material.header.sheenExtensionIndex != kInvalidMaterialExtensionIndex;
  material.hasTransmission =
      includeTransmission &&
      material.header.transmissionExtensionIndex !=
      kInvalidMaterialExtensionIndex;
  material.hasSpecular =
      material.header.specularExtensionIndex != kInvalidMaterialExtensionIndex;

  if (material.hasClearcoat) {
    material.clearcoat = pc.frameData.materialClearcoatBuffer.materials
        [material.header.clearcoatExtensionIndex];
  }
  if (material.hasSheen) {
    material.sheen =
        pc.frameData.materialSheenBuffer.materials[material.header
                                                       .sheenExtensionIndex];
  }
  if (material.hasTransmission) {
    material.transmission = pc.frameData.materialTransmissionBuffer.materials
        [material.header.transmissionExtensionIndex];
  }
  if (material.hasSpecular) {
    material.specular = pc.frameData.materialSpecularBuffer.materials
        [material.header.specularExtensionIndex];
  }
  return material;
}

MaterialData loadMaterialData(uint materialIndex) {
  return loadMaterialDataCore(materialIndex, true);
}

MaterialData loadMaterialDataWithoutTransmission(uint materialIndex) {
  return loadMaterialDataCore(materialIndex, false);
}

void disableMaterialTextures(inout MaterialData material) {
  material.header.commonTextureIndices = uvec4(kInvalidTextureBindlessIndex);
  material.header.emissiveTextureIndex = kInvalidTextureBindlessIndex;
  material.clearcoat.textureIndices =
      uvec4(kInvalidTextureBindlessIndex, kInvalidTextureBindlessIndex,
            kInvalidTextureBindlessIndex, 0u);
  material.sheen.textureIndices =
      uvec4(kInvalidTextureBindlessIndex, kInvalidTextureBindlessIndex, 0u, 0u);
  material.transmission.textureIndices =
      uvec4(kInvalidTextureBindlessIndex, kInvalidTextureBindlessIndex, 0u, 0u);
  material.specular.textureIndices =
      uvec4(kInvalidTextureBindlessIndex, kInvalidTextureBindlessIndex, 0u, 0u);
}

uint materialAlphaMode(MaterialData material) {
  return material.header.materialFlags & kMaterialFlagsAlphaModeMask;
}

bool materialDoubleSided(MaterialData material) {
  return (material.header.materialFlags & kMaterialFlagsDoubleSidedBit) != 0u;
}

uint materialWorkflow(MaterialData material) {
  return (material.header.materialFlags >> kMaterialFlagsWorkflowShift) & 0xFFu;
}

uint materialFeatureMask(MaterialData material) {
  return material.header.materialFlags >> kMaterialFlagsFeatureShift;
}

float materialEmissiveStrength(MaterialData material) {
  return material.header.emissiveFactorStrength.w;
}

float materialNormalScale(MaterialData material) {
  return material.header.normalScaleIorReserved.x;
}

float materialIor(MaterialData material) {
  return material.header.normalScaleIorReserved.y;
}

uint getMaterialTextureIndex(MaterialData material, uint slot) {
  switch (slot) {
  case kMaterialTextureSlotBaseColor:
    return material.header.commonTextureIndices.x;
  case kMaterialTextureSlotMetallicRoughness:
    return material.header.commonTextureIndices.y;
  case kMaterialTextureSlotNormal:
    return material.header.commonTextureIndices.z;
  case kMaterialTextureSlotOcclusion:
    return material.header.commonTextureIndices.w;
  case kMaterialTextureSlotEmissive:
    return material.header.emissiveTextureIndex;
  case kMaterialTextureSlotClearcoat:
    if (!material.hasClearcoat) {
      return kInvalidTextureBindlessIndex;
    }
    return material.clearcoat.textureIndices.x;
  case kMaterialTextureSlotClearcoatRoughness:
    if (!material.hasClearcoat) {
      return kInvalidTextureBindlessIndex;
    }
    return material.clearcoat.textureIndices.y;
  case kMaterialTextureSlotClearcoatNormal:
    if (!material.hasClearcoat) {
      return kInvalidTextureBindlessIndex;
    }
    return material.clearcoat.textureIndices.z;
  case kMaterialTextureSlotSpecular:
    if (!material.hasSpecular) {
      return kInvalidTextureBindlessIndex;
    }
    return material.specular.textureIndices.x;
  case kMaterialTextureSlotSpecularColor:
    if (!material.hasSpecular) {
      return kInvalidTextureBindlessIndex;
    }
    return material.specular.textureIndices.y;
  case kMaterialTextureSlotSheenColor:
    if (!material.hasSheen) {
      return kInvalidTextureBindlessIndex;
    }
    return material.sheen.textureIndices.x;
  case kMaterialTextureSlotSheenRoughness:
    if (!material.hasSheen) {
      return kInvalidTextureBindlessIndex;
    }
    return material.sheen.textureIndices.y;
  case kMaterialTextureSlotTransmission:
    if (!material.hasTransmission) {
      return kInvalidTextureBindlessIndex;
    }
    return material.transmission.textureIndices.x;
  case kMaterialTextureSlotThickness:
    if (!material.hasTransmission) {
      return kInvalidTextureBindlessIndex;
    }
    return material.transmission.textureIndices.y;
  default:
    return kInvalidTextureBindlessIndex;
  }
}

uint getMaterialUvSet(MaterialData material, uint slot) {
  switch (slot) {
  case kMaterialTextureSlotBaseColor:
    return getPackedUvBit(material.header.uvSetBits, 0u);
  case kMaterialTextureSlotMetallicRoughness:
    return getPackedUvBit(material.header.uvSetBits, 1u);
  case kMaterialTextureSlotNormal:
    return getPackedUvBit(material.header.uvSetBits, 2u);
  case kMaterialTextureSlotOcclusion:
    return getPackedUvBit(material.header.uvSetBits, 3u);
  case kMaterialTextureSlotEmissive:
    return getPackedUvBit(material.header.uvSetBits, 4u);
  case kMaterialTextureSlotClearcoat:
    if (!material.hasClearcoat) {
      return 0u;
    }
    return getPackedUvBit(material.clearcoat.uvSetBits, 0u);
  case kMaterialTextureSlotClearcoatRoughness:
    if (!material.hasClearcoat) {
      return 0u;
    }
    return getPackedUvBit(material.clearcoat.uvSetBits, 1u);
  case kMaterialTextureSlotClearcoatNormal:
    if (!material.hasClearcoat) {
      return 0u;
    }
    return getPackedUvBit(material.clearcoat.uvSetBits, 2u);
  case kMaterialTextureSlotSpecular:
    if (!material.hasSpecular) {
      return 0u;
    }
    return getPackedUvBit(material.specular.uvSetBits, 0u);
  case kMaterialTextureSlotSpecularColor:
    if (!material.hasSpecular) {
      return 0u;
    }
    return getPackedUvBit(material.specular.uvSetBits, 1u);
  case kMaterialTextureSlotSheenColor:
    if (!material.hasSheen) {
      return 0u;
    }
    return getPackedUvBit(material.sheen.uvSetBits, 0u);
  case kMaterialTextureSlotSheenRoughness:
    if (!material.hasSheen) {
      return 0u;
    }
    return getPackedUvBit(material.sheen.uvSetBits, 1u);
  case kMaterialTextureSlotTransmission:
    if (!material.hasTransmission) {
      return 0u;
    }
    return getPackedUvBit(material.transmission.uvSetBits, 0u);
  case kMaterialTextureSlotThickness:
    if (!material.hasTransmission) {
      return 0u;
    }
    return getPackedUvBit(material.transmission.uvSetBits, 1u);
  default:
    return 0u;
  }
}

PackedMaterialTransformGpuData getMaterialTransform(MaterialData material,
                                                    uint slot) {
  switch (slot) {
  case kMaterialTextureSlotBaseColor:
    return material.header.commonTransforms[0];
  case kMaterialTextureSlotMetallicRoughness:
    return material.header.commonTransforms[1];
  case kMaterialTextureSlotNormal:
    return material.header.commonTransforms[2];
  case kMaterialTextureSlotOcclusion:
    return material.header.commonTransforms[3];
  case kMaterialTextureSlotEmissive:
    return material.header.commonTransforms[4];
  case kMaterialTextureSlotClearcoat:
    if (!material.hasClearcoat) {
      return PackedMaterialTransformGpuData(0u, 0u, 0u);
    }
    return material.clearcoat.transforms[0];
  case kMaterialTextureSlotClearcoatRoughness:
    if (!material.hasClearcoat) {
      return PackedMaterialTransformGpuData(0u, 0u, 0u);
    }
    return material.clearcoat.transforms[1];
  case kMaterialTextureSlotClearcoatNormal:
    if (!material.hasClearcoat) {
      return PackedMaterialTransformGpuData(0u, 0u, 0u);
    }
    return material.clearcoat.transforms[2];
  case kMaterialTextureSlotSpecular:
    if (!material.hasSpecular) {
      return PackedMaterialTransformGpuData(0u, 0u, 0u);
    }
    return material.specular.transforms[0];
  case kMaterialTextureSlotSpecularColor:
    if (!material.hasSpecular) {
      return PackedMaterialTransformGpuData(0u, 0u, 0u);
    }
    return material.specular.transforms[1];
  case kMaterialTextureSlotSheenColor:
    if (!material.hasSheen) {
      return PackedMaterialTransformGpuData(0u, 0u, 0u);
    }
    return material.sheen.transforms[0];
  case kMaterialTextureSlotSheenRoughness:
    if (!material.hasSheen) {
      return PackedMaterialTransformGpuData(0u, 0u, 0u);
    }
    return material.sheen.transforms[1];
  case kMaterialTextureSlotTransmission:
    if (!material.hasTransmission) {
      return PackedMaterialTransformGpuData(0u, 0u, 0u);
    }
    return material.transmission.transforms[0];
  case kMaterialTextureSlotThickness:
    if (!material.hasTransmission) {
      return PackedMaterialTransformGpuData(0u, 0u, 0u);
    }
    return material.transmission.transforms[1];
  default:
    return PackedMaterialTransformGpuData(0u, 0u, 0u);
  }
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

vec2 transformedUv(MaterialData material, PerVertex vertex, uint slot) {
  return applyTextureTransform(
      selectUv(vertex.uv0, vertex.uv1, getMaterialUvSet(material, slot)),
      getMaterialTransform(material, slot));
}
