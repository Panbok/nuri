#extension GL_EXT_buffer_reference : require

layout(std430, buffer_reference) readonly buffer FrameDataBuffer {
  mat4 view;
  mat4 proj;
  vec4 cameraPos;
  uint cubemapTexId;
  uint hasCubemap;
  uint _padding0;
  uint _padding1;
};

struct PackedVertex {
  // CPU packs each vertex into 8 x 32-bit words:
  // 0..2 = position.xyz as raw float bits
  // 3    = uv as half2
  // 4..5 = normal packed as snorm16 pairs (xy, then z + pad)
  // 6..7 = tangent packed as snorm16 pairs (xy, then zw)
  // Decode uses uintBitsToFloat/unpackHalf2x16/custom snorm16 unpack.
  uint word0;
  uint word1;
  uint word2;
  uint word3;
  uint word4;
  uint word5;
  uint word6;
  uint word7;
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

layout(std430, buffer_reference) readonly buffer InstanceMetaBuffer {
  uint albedoTexIds[];
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
  InstanceMetaBuffer instanceMeta;
  InstanceCentersPhaseBuffer instanceCentersPhase;
  InstanceBaseMatricesBuffer instanceBaseMatrices;
  uint instanceCount;
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

vec3 decodePackedNormal(PackedVertex vertex) {
  const vec2 normalXY = unpackSnorm2x16Custom(vertex.word4);
  const vec2 normalZ = unpackSnorm2x16Custom(vertex.word5);
  return normalize(vec3(normalXY, normalZ.x));
}

struct PerVertex {
  vec2 uv;
  vec3 worldNormal;
  vec3 worldPos;
  vec3 patchBarycentric;
  vec3 triBarycentric;
  vec3 patchOuterFactors;
  float patchInnerFactor;
  float tessellatedFlag;
};
