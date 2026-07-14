#extension GL_EXT_buffer_reference : require

const uint kVisibilityGpuFlagFrustumCulling = 1u << 0u;
const uint kVisibilityGpuFlagOcclusionCulling = 1u << 1u;
const uint kVisibilityGpuFlagVisibleOnUncertain = 1u << 2u;

const uint kVisibilityCandidateConservativeVisible = 1u << 0u;
const uint kVisibilityInvalidTextureBindlessIndex = 0xFFFFFFFFu;
const uint kVisibilityDepthPyramidTexIdPackWidth = 4u;
const uint kVisibilityMaxDepthPyramidLevels = 16u;
const uint kVisibilityOcclusionMaxTexelSpan = 6u;
const float kVisibilityOcclusionTargetTexelSpan = 4.0;
const float kVisibilityOcclusionDepthBias = 0.0001;

struct VisibilityCandidateGpu {
  uvec4 ids;
  uvec4 ranges;
  vec4 bounds;
  vec4 boundsMin;
  vec4 boundsMax;
};

struct VisibilityPassGpuData {
  mat4 view;
  mat4 proj;
  mat4 viewProj;
  mat4 previousViewProj;
  vec4 planes[6];
  vec4 cameraOrLightPos;
  vec4 volumeMin;
  vec4 volumeMax;
  uvec4 passInfo;
  uvec4 depthPyramidInfo;
  uvec4 depthPyramidTexIds[4];
};

struct VisibilityCounterGpuData {
  uvec4 main;
  uvec4 status;
  uvec4 indirect;
  uvec4 meshlet;
  uvec4 meshlet2;
  uvec4 meshlet3;
};

layout(std430, buffer_reference) buffer VisibilityIndirectWordBuffer {
  uint words[];
};

layout(std430, buffer_reference) readonly buffer VisibilityInstanceRemapBuffer {
  uint indices[];
};

layout(std430, buffer_reference) readonly buffer VisibilityCandidateBuffer {
  VisibilityCandidateGpu values[];
};

layout(std430, buffer_reference) readonly buffer VisibilityPassBuffer {
  VisibilityPassGpuData data;
};

layout(std430, buffer_reference) buffer VisibilityVisibleIndexBuffer {
  uint indices[];
};

layout(std430, buffer_reference) buffer VisibilityCounterBuffer {
  VisibilityCounterGpuData data;
};

struct VisibilityMeshletBatchGpuData {
  uvec2 vertexBufferAddress;
  uvec2 vertexDecodeBufferAddress;
  uvec2 meshletBufferAddress;
  uvec2 meshletVertexIndexBufferAddress;
  uvec2 meshletPrimitiveIndexBufferAddress;
  uvec2 meshletLodRangeBufferAddress;
  uvec4 draw;
  uvec4 mesh;
  uvec4 flags;
};

struct VisibilityMeshletDispatchGpuData {
  uvec4 groups;
  uvec4 batches;
};

layout(std430, buffer_reference) readonly buffer VisibilityMeshletBatchBuffer {
  VisibilityMeshletBatchGpuData values[];
};

layout(std430, buffer_reference) readonly buffer
    VisibilityMeshletDispatchBuffer {
  VisibilityMeshletDispatchGpuData values[];
};

layout(std430, buffer_reference) buffer VisibilityMeshDispatchCommandBuffer {
  uint words[];
};

bool visibilitySphereOutsideFrustum(vec4 planes[6], vec3 center, float radius) {
  float safeRadius = max(radius, 0.0);
  for (uint i = 0u; i < 6u; ++i) {
    vec4 plane = planes[i];
    if (dot(plane.xyz, center) + plane.w < -safeRadius) {
      return true;
    }
  }
  return false;
}

uint visibilityDepthPyramidTexId(VisibilityPassGpuData passData, uint level) {
  if (level >= passData.depthPyramidInfo.z ||
      level >= kVisibilityMaxDepthPyramidLevels) {
    return kVisibilityInvalidTextureBindlessIndex;
  }
  return passData.depthPyramidTexIds[
      level / kVisibilityDepthPyramidTexIdPackWidth]
      [level % kVisibilityDepthPyramidTexIdPackWidth];
}

bool visibilityBoundsOccludedByPreviousDepth(
    VisibilityPassGpuData passData, VisibilityCandidateGpu candidate) {
  uint levelCount = passData.depthPyramidInfo.z;
  uint samplerId = passData.depthPyramidInfo.w;
  if (levelCount == 0u || passData.depthPyramidInfo.x == 0u ||
      passData.depthPyramidInfo.y == 0u) {
    return false;
  }

  vec3 bmin = candidate.boundsMin.xyz;
  vec3 bmax = candidate.boundsMax.xyz;
  vec2 uvMin = vec2(1.0);
  vec2 uvMax = vec2(0.0);
  float nearestDepth = 1.0e20;

  for (uint cornerIndex = 0u; cornerIndex < 8u; ++cornerIndex) {
    vec3 corner = vec3((cornerIndex & 1u) != 0u ? bmax.x : bmin.x,
                       (cornerIndex & 2u) != 0u ? bmax.y : bmin.y,
                       (cornerIndex & 4u) != 0u ? bmax.z : bmin.z);
    vec4 clip = passData.previousViewProj * vec4(corner, 1.0);
    if (clip.w <= 1.0e-5) {
      return false;
    }

    vec3 ndc = clip.xyz / clip.w;
    if (ndc.z < -1.0 || ndc.z > 1.0) {
      return false;
    }

    vec2 uv = ndc.xy * 0.5 + vec2(0.5);
    uvMin = min(uvMin, uv);
    uvMax = max(uvMax, uv);
    nearestDepth = min(nearestDepth, ndc.z * 0.5 + 0.5);
  }

  if (uvMin.x < 0.0 || uvMin.y < 0.0 || uvMax.x > 1.0 || uvMax.y > 1.0) {
    return false;
  }

  vec2 pyramidSize = vec2(passData.depthPyramidInfo.xy);
  vec2 pixelSpan = max((uvMax - uvMin) * pyramidSize, vec2(1.0));
  float maxPixelSpan = max(pixelSpan.x, pixelSpan.y);
  uint level =
      uint(clamp(ceil(log2(max(maxPixelSpan /
                                      kVisibilityOcclusionTargetTexelSpan,
                                  1.0))),
                 0.0, float(levelCount - 1u)));
  uint texId = visibilityDepthPyramidTexId(passData, level);
  if (texId == kVisibilityInvalidTextureBindlessIndex) {
    return false;
  }

  vec2 mipSize =
      max(ceil(pyramidSize / exp2(float(level))), vec2(1.0, 1.0));
  ivec2 mipExtent = ivec2(mipSize);
  ivec2 texelMin =
      clamp(ivec2(floor(uvMin * mipSize)), ivec2(0), mipExtent - ivec2(1));
  ivec2 texelMax =
      clamp(ivec2(floor(uvMax * mipSize)), ivec2(0), mipExtent - ivec2(1));
  ivec2 texelSpan = texelMax - texelMin + ivec2(1);
  if (texelSpan.x <= 0 || texelSpan.y <= 0 ||
      texelSpan.x > int(kVisibilityOcclusionMaxTexelSpan) ||
      texelSpan.y > int(kVisibilityOcclusionMaxTexelSpan)) {
    return false;
  }

  float occluderMaxDepth = 0.0;
  for (uint y = 0u; y < kVisibilityOcclusionMaxTexelSpan; ++y) {
    if (y >= uint(texelSpan.y)) {
      break;
    }
    for (uint x = 0u; x < kVisibilityOcclusionMaxTexelSpan; ++x) {
      if (x >= uint(texelSpan.x)) {
        break;
      }
      ivec2 texel = texelMin + ivec2(x, y);
      vec2 uv = (vec2(texel) + vec2(0.5)) / mipSize;
      occluderMaxDepth =
          max(occluderMaxDepth, textureBindless2D(texId, samplerId, uv).g);
    }
  }
  return nearestDepth > occluderMaxDepth + kVisibilityOcclusionDepthBias;
}
