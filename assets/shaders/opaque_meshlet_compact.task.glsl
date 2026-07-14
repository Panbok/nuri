#extension GL_EXT_mesh_shader : require

#define NURI_OPAQUE_MESHLET_BATCHED 1
#define NURI_OPAQUE_MESHLET_COMPACT_TASK 1
#include "meshlet_common.sp"

layout(local_size_x = 32) in;

taskPayloadSharedEXT MeshletTaskPayload meshletPayload;

const uint kMeshletCounterFlagEnabled = 1u << 0u;

void main() {
  const uint compactCount = pc.compactionCounters.counts[pc.batchBase];
  const uint compactGroup =
      gl_WorkGroupID.y * pc.sourceFrameIndex + gl_WorkGroupID.x;
  const uint groupRecordBase = compactGroup * kOpaqueMeshletTaskPayloadCapacity;
  const uint recordLocal = groupRecordBase + gl_LocalInvocationIndex;

  if (recordLocal < compactCount) {
    const CompactedMeshletGpuData compacted =
        pc.compactedMeshlets.values[pc.candidateOffset + recordLocal];
    meshletPayload.meshletIndex[gl_LocalInvocationIndex] = compacted.ids.x;
    meshletPayload.globalInstanceId[gl_LocalInvocationIndex] = compacted.ids.y;
    meshletPayload.selectedLod[gl_LocalInvocationIndex] = compacted.ids.z;
    meshletPayload.batchIndex[gl_LocalInvocationIndex] = compacted.ids.w;
  }

  barrier();
  if (gl_LocalInvocationIndex == 0u) {
    const uint emittedCount = groupRecordBase < compactCount
                                  ? min(compactCount - groupRecordBase,
                                        kOpaqueMeshletTaskPayloadCapacity)
                                  : 0u;
    meshletPayload.visibleCount = emittedCount;
    if ((pc.meshletCounterFlags & kMeshletCounterFlagEnabled) != 0u &&
        emittedCount != 0u) {
      atomicAdd(pc.visibilityCounters.data.meshlet2.x, emittedCount);
      atomicAdd(pc.visibilityCounters.data.meshlet2.y, 1u);
    }
    EmitMeshTasksEXT(emittedCount, 1u, 1u);
  }
}
