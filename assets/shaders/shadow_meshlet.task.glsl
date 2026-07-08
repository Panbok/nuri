#extension GL_EXT_mesh_shader : require

#define NURI_MESHLET_TASK_PAYLOAD_BATCHED 1
#include "meshlet_common.sp"

layout(local_size_x = 32) in;

taskPayloadSharedEXT MeshletTaskPayload meshletPayload;
shared uint meshletVisibleMask;

const uint kShadowMeshletCounterFlagEnabled = 1u << 0u;

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

uint meshletForcedLod() {
  return (pc.meshletFlags >> kMeshletFlagForcedLodShift) &
         kMeshletFlagForcedLodMask;
}

uint meshletCounterSourceFrame() { return floatBitsToUint(pc.lodThresholds.x); }

uint meshletCounterFlags() { return floatBitsToUint(pc.lodThresholds.y); }

uint meshletCascadeIndex() { return floatBitsToUint(pc.lodThresholds.z); }

uint meshletSubmeshIndex() { return floatBitsToUint(pc.lodThresholds.w); }

void markShadowMeshletCounterFrame() {
  if ((meshletCounterFlags() & kShadowMeshletCounterFlagEnabled) == 0u) {
    return;
  }
  pc.visibilityCounters.data.status.z = meshletCounterSourceFrame();
  pc.visibilityCounters.data.status.w = 1u;
}

void countShadowMeshletBoundsReject() {
  if ((meshletCounterFlags() & kShadowMeshletCounterFlagEnabled) != 0u) {
    atomicAdd(pc.visibilityCounters.data.meshlet.x, 1u);
  }
}

bool meshletOutsideShadowCascade(MeshletDescriptorGpuData meshlet,
                                 InstanceData inst, uint meshletFlags,
                                 uint cascadeIndex) {
  if ((meshletFlags & kMeshletFlagShadowCascadeCulling) == 0u) {
    return false;
  }

  const ShadowCascadeGpuData cascade =
      pc.frameData.shadowFrameBuffer.cascades[cascadeIndex];
  const vec3 localCenter = meshlet.boundsSphere.xyz;
  const float localRadius = max(meshlet.boundsSphere.w, 0.0);
  const vec4 worldCenter = inst.modelMatrix * vec4(localCenter, 1.0);
  const vec3 lightCenter = (cascade.lightView * worldCenter).xyz;
  const float radius = localRadius * meshletInstanceMaxScale(inst.modelMatrix);
  const vec3 boundsMin = cascade.cullingBoundsMin.xyz;
  const vec3 boundsMax = cascade.cullingBoundsMax.xyz;

  return lightCenter.x + radius < boundsMin.x ||
         lightCenter.x - radius > boundsMax.x ||
         lightCenter.y + radius < boundsMin.y ||
         lightCenter.y - radius > boundsMax.y ||
         lightCenter.z + radius < boundsMin.z ||
         lightCenter.z - radius > boundsMax.z;
}

void main() {
  const uint lane = gl_LocalInvocationIndex;
  if (lane == 0u) {
    meshletPayload.visibleCount = 0u;
    meshletVisibleMask = 0u;
    if (gl_WorkGroupID.x == 0u) {
      markShadowMeshletCounterFrame();
    }
  }
  barrier();

  const SubmeshMeshletLodRangeGpuData lodRange =
      pc.meshletLodRanges.values[meshletSubmeshIndex()];
  const uint selectedLod =
      resolveAvailableMeshletLod(lodRange, meshletForcedLod());
  const uint selectedMeshletCount = lodRange.meshletCount[selectedLod];

  bool acceptedMeshlet = false;
  uint acceptedMeshletIndex = 0u;
  uint acceptedGlobalInstanceId = 0u;
  if (selectedMeshletCount != 0u && pc.instanceCount != 0u) {
    const uint candidate =
        pc.candidateOffset +
        gl_WorkGroupID.x * kOpaqueMeshletTaskPayloadCapacity + lane;
    const uint instanceLocal = candidate / selectedMeshletCount;
    if (instanceLocal < pc.instanceCount) {
      const uint meshletLocal =
          candidate - instanceLocal * selectedMeshletCount;
      const uint remapIndex = pc.firstInstance + instanceLocal;
      const uint meshletIndex =
          lodRange.meshletOffset[selectedLod] + meshletLocal;
      const uint globalInstanceId = pc.instanceRemap.ids[remapIndex];
      const MeshletDescriptorGpuData meshlet = pc.meshlets.values[meshletIndex];
      const InstanceData inst = pc.instanceMatrices.instances[globalInstanceId];
      if (meshletOutsideShadowCascade(meshlet, inst, pc.meshletFlags,
                                      meshletCascadeIndex())) {
        countShadowMeshletBoundsReject();
      } else {
        acceptedMeshlet = true;
        acceptedMeshletIndex = meshletIndex;
        acceptedGlobalInstanceId = globalInstanceId;
        atomicOr(meshletVisibleMask, 1u << lane);
      }
    }
  }

  barrier();
  if (acceptedMeshlet) {
    const uint payloadIndex =
        bitCount(meshletVisibleMask & ((1u << lane) - 1u));
    if (payloadIndex < kOpaqueMeshletTaskPayloadCapacity) {
      meshletPayload.meshletIndex[payloadIndex] = acceptedMeshletIndex;
      meshletPayload.globalInstanceId[payloadIndex] = acceptedGlobalInstanceId;
      meshletPayload.selectedLod[payloadIndex] = selectedLod;
    }
  }

  barrier();
  if (lane == 0u) {
    const uint emittedCount =
        min(bitCount(meshletVisibleMask), kOpaqueMeshletTaskPayloadCapacity);
    meshletPayload.visibleCount = emittedCount;
    EmitMeshTasksEXT(emittedCount, 1u, 1u);
  }
}
