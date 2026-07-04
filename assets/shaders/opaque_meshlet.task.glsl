#extension GL_EXT_mesh_shader : require
#extension GL_EXT_nonuniform_qualifier : require

#define NURI_OPAQUE_MESHLET_BATCHED 1
#include "meshlet_common.sp"

layout(local_size_x = 32) in;

layout(set = 0, binding = 0) uniform texture2D kTextures2D[];
layout(set = 0, binding = 1) uniform sampler kSamplers[];

taskPayloadSharedEXT MeshletTaskPayload meshletPayload;
shared uint meshletVisibleMask;

vec4 matrixRow(mat4 value, uint row) {
  return vec4(value[0][row], value[1][row], value[2][row], value[3][row]);
}

bool sphereOutsideViewPlane(vec4 plane, vec4 viewCenter, float radius) {
  const float signedDistance = dot(plane, viewCenter);
  const float planeRadius = radius * length(plane.xyz);
  return signedDistance < -planeRadius;
}

bool meshletFrustumCull(MeshletDescriptorGpuData meshlet, InstanceData inst,
                        uint meshletFlags) {
  if ((meshletFlags & kMeshletFlagFrustumCulling) == 0u) {
    return false;
  }

  const vec3 localCenter = meshlet.boundsSphere.xyz;
  const float localRadius = max(meshlet.boundsSphere.w, 0.0);
  const vec4 worldCenter4 = inst.modelMatrix * vec4(localCenter, 1.0);
  const vec4 viewCenter = pc.frameData.view * worldCenter4;
  const float radius = localRadius * meshletInstanceMaxScale(inst.modelMatrix);

  const vec4 clipRowX = matrixRow(pc.frameData.proj, 0u);
  const vec4 clipRowY = matrixRow(pc.frameData.proj, 1u);
  const vec4 clipRowZ = matrixRow(pc.frameData.proj, 2u);
  const vec4 clipRowW = matrixRow(pc.frameData.proj, 3u);

  return sphereOutsideViewPlane(clipRowW + clipRowX, viewCenter, radius) ||
         sphereOutsideViewPlane(clipRowW - clipRowX, viewCenter, radius) ||
         sphereOutsideViewPlane(clipRowW + clipRowY, viewCenter, radius) ||
         sphereOutsideViewPlane(clipRowW - clipRowY, viewCenter, radius) ||
         sphereOutsideViewPlane(clipRowW + clipRowZ, viewCenter, radius) ||
         sphereOutsideViewPlane(clipRowW - clipRowZ, viewCenter, radius);
}

bool meshletConeCull(MeshletDescriptorGpuData meshlet, InstanceData inst,
                     uint meshletFlags) {
  if ((meshletFlags & kMeshletFlagConeCulling) == 0u ||
      (meshletFlags & kMeshletFlagDoubleSided) != 0u) {
    return false;
  }

  const vec3 localAxis = meshlet.coneAxisCutoff.xyz;
  if (dot(localAxis, localAxis) <= 1.0e-10) {
    return false;
  }
  const mat3 modelBasis = mat3(inst.modelMatrix);
  const float handedness =
      dot(cross(modelBasis[0], modelBasis[1]), modelBasis[2]);
  if (handedness < 0.0) {
    return false;
  }
  const mat3 normalMatrix = mat3(inst.normalMatCol0.xyz, inst.normalMatCol1.xyz,
                                 inst.normalMatCol2.xyz);
  const vec3 transformedAxis = normalMatrix * localAxis;
  if (dot(transformedAxis, transformedAxis) <= 1.0e-10) {
    return false;
  }
  const vec3 worldAxis = normalize(transformedAxis);
  const vec3 worldApex =
      (inst.modelMatrix * vec4(meshlet.coneApex.xyz, 1.0)).xyz;
  const vec3 cameraToApex = worldApex - pc.frameData.cameraPos.xyz;
  const float distanceSq = dot(cameraToApex, cameraToApex);
  if (distanceSq <= 1.0e-10) {
    return false;
  }
  const vec3 viewDir = cameraToApex * inversesqrt(distanceSq);
  return dot(viewDir, worldAxis) >= meshlet.coneAxisCutoff.w + 0.02;
}

const uint kMeshletOcclusionMaxTexelSpan = 6u;
const float kMeshletOcclusionTargetTexelSpan = 4.0;
const float kMeshletOcclusionDepthBias = 0.0001;
const uint kMeshletCounterFlagEnabled = 1u << 0u;

bool meshletOcclusionCull(MeshletDescriptorGpuData meshlet, InstanceData inst,
                          uint meshletFlags) {
  if ((meshletFlags & kMeshletFlagOcclusionCulling) == 0u ||
      (pc.frameData.sceneDepthPyramidInfo.w &
       kFrameDataDepthPyramidPreviousFrame) == 0u) {
    return false;
  }

  const uint levelCount = min(pc.frameData.sceneDepthPyramidInfo.z,
                              pc.frameData.sceneDepthPyramidLevelCount);
  const uint samplerId = pc.frameData.sceneDepthSamplerId;
  if (levelCount == 0u || pc.frameData.sceneDepthPyramidInfo.x == 0u ||
      pc.frameData.sceneDepthPyramidInfo.y == 0u ||
      samplerId == kInvalidSamplerBindlessIndex) {
    return false;
  }

  const vec3 localCenter = meshlet.boundsSphere.xyz;
  const float localRadius = max(meshlet.boundsSphere.w, 0.0);
  const vec3 worldCenter = (inst.modelMatrix * vec4(localCenter, 1.0)).xyz;
  const float radius = localRadius * meshletInstanceMaxScale(inst.modelMatrix);
  if (radius <= 0.0) {
    return false;
  }

  const vec3 bmin = worldCenter - vec3(radius);
  const vec3 bmax = worldCenter + vec3(radius);
  vec2 uvMin = vec2(1.0);
  vec2 uvMax = vec2(0.0);
  float nearestDepth = 1.0e20;

  for (uint cornerIndex = 0u; cornerIndex < 8u; ++cornerIndex) {
    vec3 corner = vec3((cornerIndex & 1u) != 0u ? bmax.x : bmin.x,
                       (cornerIndex & 2u) != 0u ? bmax.y : bmin.y,
                       (cornerIndex & 4u) != 0u ? bmax.z : bmin.z);
    vec4 clip = pc.frameData.previousViewProj * vec4(corner, 1.0);
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

  const vec2 pyramidSize = vec2(pc.frameData.sceneDepthPyramidInfo.xy);
  const vec2 pixelSpan = max((uvMax - uvMin) * pyramidSize, vec2(1.0));
  const float maxPixelSpan = max(pixelSpan.x, pixelSpan.y);
  const uint level = uint(clamp(
      ceil(log2(max(maxPixelSpan / kMeshletOcclusionTargetTexelSpan, 1.0))),
      0.0, float(levelCount - 1u)));
  const uint texId = getSceneDepthPyramidTexId(pc.frameData, level);
  if (texId == kInvalidTextureBindlessIndex) {
    return false;
  }

  const vec2 mipSize =
      max(ceil(pyramidSize / exp2(float(level))), vec2(1.0, 1.0));
  const ivec2 mipExtent = ivec2(mipSize);
  const ivec2 texelMin =
      clamp(ivec2(floor(uvMin * mipSize)), ivec2(0), mipExtent - ivec2(1));
  const ivec2 texelMax =
      clamp(ivec2(floor(uvMax * mipSize)), ivec2(0), mipExtent - ivec2(1));
  const ivec2 texelSpan = texelMax - texelMin + ivec2(1);
  if (texelSpan.x <= 0 || texelSpan.y <= 0 ||
      texelSpan.x > int(kMeshletOcclusionMaxTexelSpan) ||
      texelSpan.y > int(kMeshletOcclusionMaxTexelSpan)) {
    return false;
  }

  float occluderMaxDepth = 0.0;
  for (uint y = 0u; y < kMeshletOcclusionMaxTexelSpan; ++y) {
    if (y >= uint(texelSpan.y)) {
      break;
    }
    for (uint x = 0u; x < kMeshletOcclusionMaxTexelSpan; ++x) {
      if (x >= uint(texelSpan.x)) {
        break;
      }
      const ivec2 texel = texelMin + ivec2(x, y);
      const vec2 uv = (vec2(texel) + vec2(0.5)) / mipSize;
      vec4 sampleValue =
          texture(nonuniformEXT(sampler2D(kTextures2D[nonuniformEXT(texId)],
                                          kSamplers[nonuniformEXT(samplerId)])),
                  uv);
      occluderMaxDepth = max(occluderMaxDepth, sampleValue.g);
    }
  }
  return nearestDepth > occluderMaxDepth + kMeshletOcclusionDepthBias;
}

void markMeshletCounterFrame(uint meshletFlags) {
  if ((pc.meshletCounterFlags & kMeshletCounterFlagEnabled) == 0u) {
    return;
  }
  pc.visibilityCounters.data.status.z = pc.sourceFrameIndex;
  pc.visibilityCounters.data.status.w = 1u;
  if ((meshletFlags & kMeshletFlagOcclusionCulling) != 0u) {
    pc.visibilityCounters.data.status.y = 1u;
  }
}

void countMeshletFrustumReject() {
  if ((pc.meshletCounterFlags & kMeshletCounterFlagEnabled) != 0u) {
    atomicAdd(pc.visibilityCounters.data.meshlet.x, 1u);
  }
}

void countMeshletConeReject() {
  if ((pc.meshletCounterFlags & kMeshletCounterFlagEnabled) != 0u) {
    atomicAdd(pc.visibilityCounters.data.meshlet.y, 1u);
  }
}

void countMeshletOcclusionReject() {
  if ((pc.meshletCounterFlags & kMeshletCounterFlagEnabled) != 0u) {
    atomicAdd(pc.visibilityCounters.data.meshlet.z, 1u);
  }
}

void countMeshletPayloadOverflow() {
  if ((pc.meshletCounterFlags & kMeshletCounterFlagEnabled) != 0u) {
    atomicAdd(pc.visibilityCounters.data.meshlet.w, 1u);
  }
}

void countMeshletTaskGroup() {
  if ((pc.meshletCounterFlags & kMeshletCounterFlagEnabled) != 0u) {
    atomicAdd(pc.visibilityCounters.data.meshlet2.y, 1u);
  }
}

void countMeshletEmitted(uint emittedCount) {
  if ((pc.meshletCounterFlags & kMeshletCounterFlagEnabled) != 0u &&
      emittedCount != 0u) {
    atomicAdd(pc.visibilityCounters.data.meshlet2.x, emittedCount);
  }
}

uint resolveAvailableMeshletLod(SubmeshMeshletLodRangeGpuData lodRange,
                                uint requestedLod) {
  const uint lodCount = clamp(lodRange.metadata.x, 1u, 4u);
  uint candidate = min(requestedLod, lodCount - 1u);
  while (candidate > 0u && lodRange.meshletCount[candidate] == 0u) {
    --candidate;
  }
  return candidate;
}

uint meshletCandidateSpan(SubmeshMeshletLodRangeGpuData lodRange) {
  return max(max(lodRange.meshletCount.x, lodRange.meshletCount.y),
             max(lodRange.meshletCount.z, lodRange.meshletCount.w));
}

uint meshletForcedLod(uint meshletFlags) {
  return (meshletFlags >> kMeshletFlagForcedLodShift) &
         kMeshletFlagForcedLodMask;
}

uint selectMeshletLod(SubmeshMeshletLodRangeGpuData lodRange,
                      uint globalInstanceId, uint meshletFlags) {
  if ((meshletFlags & kMeshletFlagGpuLod) == 0u) {
    return resolveAvailableMeshletLod(lodRange, meshletForcedLod(meshletFlags));
  }

  const vec4 lodBounds = pc.instanceLodBounds.values[globalInstanceId];
  const vec3 delta = pc.frameData.cameraPos.xyz - lodBounds.xyz;
  const float normalizedDistanceSq = dot(delta, delta) * max(lodBounds.w, 0.0);
  const vec3 thresholdsSq = pc.lodThresholds.xyz * pc.lodThresholds.xyz;

  uint requestedLod = 0u;
  if (normalizedDistanceSq >= thresholdsSq.z) {
    requestedLod = 3u;
  } else if (normalizedDistanceSq >= thresholdsSq.y) {
    requestedLod = 2u;
  } else if (normalizedDistanceSq >= thresholdsSq.x) {
    requestedLod = 1u;
  }
  return resolveAvailableMeshletLod(lodRange, requestedLod);
}

void main() {
  const uint batchIndex = pc.batchBase + gl_WorkGroupID.y;
  MeshletBatchGpuData batch = pc.meshletBatches.values[batchIndex];
  const uint meshletFlags = batch.flags.x;
  const SubmeshMeshletLodRangeGpuData lodRange =
      batch.meshletLodRanges.values[batch.mesh.z];
  const uint candidateSpan = batch.mesh.w;
  const uint instanceCount = batch.draw.x;

  if (gl_LocalInvocationIndex == 0u) {
    if (gl_WorkGroupID.x == 0u && gl_WorkGroupID.y == 0u) {
      markMeshletCounterFrame(meshletFlags);
    }
    countMeshletTaskGroup();
    meshletPayload.visibleCount = 0u;
    meshletVisibleMask = 0u;
  }
  barrier();

  bool acceptedMeshlet = false;
  uint acceptedMeshletIndex = 0u;
  uint acceptedGlobalInstanceId = 0u;
  uint acceptedSelectedLod = 0u;
  if (candidateSpan != 0u && instanceCount != 0u) {
    const uint candidate =
        pc.candidateOffset +
        gl_WorkGroupID.x * kOpaqueMeshletTaskPayloadCapacity +
        gl_LocalInvocationIndex;
    const uint instanceLocal = candidate / candidateSpan;
    if (instanceLocal < instanceCount) {
      const uint meshletLocal = candidate - instanceLocal * candidateSpan;
      const uint remapIndex = batch.draw.y + instanceLocal;
      const uint globalInstanceId = pc.instanceRemap.ids[remapIndex];
      const uint selectedLod =
          selectMeshletLod(lodRange, globalInstanceId, meshletFlags);
      const uint selectedMeshletCount = lodRange.meshletCount[selectedLod];
      if (meshletLocal < selectedMeshletCount) {
        const uint meshletIndex =
            lodRange.meshletOffset[selectedLod] + meshletLocal;
        const InstanceData inst =
            pc.instanceMatrices.instances[globalInstanceId];
        const MeshletDescriptorGpuData meshlet =
            batch.meshlets.values[meshletIndex];
        const bool rejectedByFrustum =
            meshletFrustumCull(meshlet, inst, meshletFlags);
        const bool rejectedByCone =
            !rejectedByFrustum && meshletConeCull(meshlet, inst, meshletFlags);
        const bool rejectedByOcclusion =
            !rejectedByFrustum && !rejectedByCone &&
            meshletOcclusionCull(meshlet, inst, meshletFlags);
        if (rejectedByFrustum) {
          countMeshletFrustumReject();
        } else if (rejectedByCone) {
          countMeshletConeReject();
        } else if (rejectedByOcclusion) {
          countMeshletOcclusionReject();
        } else {
          acceptedMeshlet = true;
          acceptedMeshletIndex = meshletIndex;
          acceptedGlobalInstanceId = globalInstanceId;
          acceptedSelectedLod = selectedLod;
          atomicOr(meshletVisibleMask, 1u << gl_LocalInvocationIndex);
        }
      }
    }
  }

  barrier();
  if (acceptedMeshlet) {
    const uint payloadIndex =
        bitCount(meshletVisibleMask & ((1u << gl_LocalInvocationIndex) - 1u));
    if (payloadIndex < kOpaqueMeshletTaskPayloadCapacity) {
      meshletPayload.meshletIndex[payloadIndex] = acceptedMeshletIndex;
      meshletPayload.globalInstanceId[payloadIndex] = acceptedGlobalInstanceId;
      meshletPayload.selectedLod[payloadIndex] = acceptedSelectedLod;
      meshletPayload.batchIndex[payloadIndex] = batchIndex;
    } else {
      countMeshletPayloadOverflow();
    }
  }

  barrier();
  if (gl_LocalInvocationIndex == 0u) {
    const uint emittedCount =
        min(bitCount(meshletVisibleMask), kOpaqueMeshletTaskPayloadCapacity);
    meshletPayload.visibleCount = emittedCount;
    countMeshletEmitted(emittedCount);
    EmitMeshTasksEXT(emittedCount, 1u, 1u);
  }
}
