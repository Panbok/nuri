#extension GL_EXT_mesh_shader : require

#include "meshlet_common.sp"

layout(local_size_x = 1) in;

taskPayloadSharedEXT MeshletTaskPayload meshletPayload;

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

uint meshletSubmeshIndex() { return floatBitsToUint(pc.lodThresholds.w); }

void main() {
  const SubmeshMeshletLodRangeGpuData lodRange =
      pc.meshletLodRanges.values[meshletSubmeshIndex()];
  const uint candidateSpan = meshletCandidateSpan(lodRange);
  if (candidateSpan == 0u || pc.instanceCount == 0u) {
    EmitMeshTasksEXT(0u, 1u, 1u);
    return;
  }

  const uint candidate = pc.candidateOffset + gl_WorkGroupID.x;
  const uint instanceLocal = candidate / candidateSpan;
  if (instanceLocal >= pc.instanceCount) {
    EmitMeshTasksEXT(0u, 1u, 1u);
    return;
  }

  const uint meshletLocal = candidate - instanceLocal * candidateSpan;
  const uint selectedLod = resolveAvailableMeshletLod(lodRange,
                                                      meshletForcedLod());
  const uint selectedMeshletCount = lodRange.meshletCount[selectedLod];
  if (meshletLocal >= selectedMeshletCount) {
    EmitMeshTasksEXT(0u, 1u, 1u);
    return;
  }

  const uint remapIndex = pc.firstInstance + instanceLocal;
  meshletPayload.meshletIndex =
      lodRange.meshletOffset[selectedLod] + meshletLocal;
  meshletPayload.globalInstanceId = pc.instanceRemap.ids[remapIndex];
  meshletPayload.selectedLod = selectedLod;
  EmitMeshTasksEXT(1u, 1u, 1u);
}
