#version 460 core
#extension GL_EXT_buffer_reference : require

#define NURI_CUSTOM_PUSH_CONSTANTS 1
#include "common.sp"

layout(push_constant) uniform PushConstants {
  mat4 viewProjection;
  vec4 cameraRightAndScale;
  vec4 cameraUpAndDebugView;
  DDGIFrameBuffer frame;
  uint volumeSlot;
  uint submittedSequence;
} pc;

layout(location = 0) out vec2 outLocal;
layout(location = 1) out vec4 outColor;

const vec2 kCorners[6] = vec2[6](
    vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
    vec2(1.0, 1.0), vec2(-1.0, 1.0), vec2(-1.0, -1.0));

vec3 probeNominalLocalPosition(DDGIVolumeGpuData volume,
                               uvec3 coordinate) {
  const ivec3 cameraCell = ivec3(volume.generations.x,
                                 volume.generations.y,
                                 volume.generations.w);
  return (vec3(coordinate) -
          0.5 * vec3(volume.probeCountsAndCount.xyz - 1u)) *
             volume.probeSpacingAndBias.xyz +
         vec3(cameraCell) * volume.probeSpacingAndBias.xyz;
}

vec4 stateColor(uint state, float relocation, float age, uint debugView) {
  if (debugView == 10u) {
    const float amount = clamp(relocation, 0.0, 1.0);
    return vec4(mix(vec3(0.1, 0.7, 1.0), vec3(1.0, 0.18, 0.05), amount),
                0.9);
  }
  if (debugView == 11u) {
    return vec4(mix(vec3(0.1, 1.0, 0.2), vec3(1.0, 0.12, 0.05), age),
                0.9);
  }
  if (state == kDDGIProbeStateUninitialized) return vec4(1.0, 0.15, 1.0, 0.9);
  if (state == kDDGIProbeStateOff) return vec4(1.0, 0.08, 0.05, 0.75);
  if (state == kDDGIProbeStateSleeping) return vec4(0.25, 0.3, 0.35, 0.65);
  if (state == kDDGIProbeStateNewlyAwake) return vec4(1.0, 0.72, 0.12, 0.9);
  if (state == kDDGIProbeStateAwake) return vec4(1.0, 0.95, 0.18, 0.9);
  if (state == kDDGIProbeStateNewlyVigilant) return vec4(0.2, 1.0, 0.85, 0.9);
  return vec4(0.15, 1.0, 0.25, 0.9);
}

void main() {
  DDGIVolumeGpuData volume = pc.frame.volumes[pc.volumeSlot];
  const uint probe = uint(gl_InstanceIndex);
  const uvec3 counts = volume.probeCountsAndCount.xyz;
  const uint xy = counts.x * counts.y;
  const uvec3 physical = uvec3(probe % counts.x,
      (probe % xy) / counts.x, probe / xy);
  const uvec3 logical =
      (physical + counts - volume.ringOriginAndFlags.xyz) % counts;
  const DDGIProbeStateGpuData state = volume.probeStates.values[probe];
  const vec3 localPosition =
      probeNominalLocalPosition(volume, logical) + state.relocation.xyz;
  const vec3 worldPosition =
      (volume.worldFromLocal * vec4(localPosition, 1.0)).xyz;
  const vec2 corner = kCorners[gl_VertexIndex];
  const float scale = pc.cameraRightAndScale.w;
  const vec3 billboardPosition = worldPosition +
      (pc.cameraRightAndScale.xyz * corner.x +
       pc.cameraUpAndDebugView.xyz * corner.y) * scale;
  outLocal = corner;
  const float minSpacing = min(min(volume.probeSpacingAndBias.x,
                                   volume.probeSpacingAndBias.y),
                               volume.probeSpacingAndBias.z);
  const float relocation = length(state.relocation.xyz) /
                           max(0.5 * minSpacing, 1.0e-6);
  const float age = clamp(float(pc.submittedSequence -
                                min(pc.submittedSequence,
                                    state.stateAgeFlags.y)) / 64.0,
                          0.0, 1.0);
  outColor = stateColor(state.stateAgeFlags.x, relocation, age,
                        uint(pc.cameraUpAndDebugView.w));
  gl_Position = pc.viewProjection * vec4(billboardPosition, 1.0);
}
