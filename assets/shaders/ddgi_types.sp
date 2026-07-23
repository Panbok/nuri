const uint kDDGIFrameGpuDataVersion = 2u;
const uint kDDGIMaxVolumes = 8u;
const uint kDDGIMaxSampledVolumes = 2u;
const uint kDDGIEffectiveKindAuthored = 0u;
const uint kDDGIEffectiveKindSceneFit = 1u;
const uint kDDGIEffectiveKindClipmapCascade = 2u;
const uint kDDGIProbeStateUninitialized = 0u;
const uint kDDGIProbeStateOff = 1u;
const uint kDDGIProbeStateSleeping = 2u;
const uint kDDGIProbeStateNewlyAwake = 3u;
const uint kDDGIProbeStateAwake = 4u;
const uint kDDGIProbeStateNewlyVigilant = 5u;
const uint kDDGIProbeStateVigilant = 6u;

struct DDGIProbeStateGpuData {
  vec4 relocation;
  uvec4 stateAgeFlags;
};

struct DDGIProbeUpdateEntryGpu {
  uint volumeSlot;
  uint probeId;
  uint rayBase;
  uint rayCount;
  uint flags;
  uint resultState;
  uint resultSubmittedSequence;
  uint resultIteration;
  vec4 resultRelocation;
};

struct DDGIRayResultGpuData {
  vec4 radianceAndDistance;
  uvec4 metadata;
};

layout(std430, buffer_reference) readonly buffer DDGIProbeStateBuffer {
  DDGIProbeStateGpuData values[];
};

struct DDGIVolumeGpuData {
  mat4 worldFromLocal;
  mat4 localFromWorld;
  DDGIProbeStateBuffer probeStates;
  uvec2 resourceFlagsReserved;
  vec4 probeSpacingAndBias;
  vec4 centerHalfExtentsAndMaxDistance;
  uvec4 probeCountsAndCount;
  uvec4 irradianceAtlas;
  uvec4 distanceAtlas;
  uvec4 ringOriginAndFlags;
  uvec4 generations;
  uvec4 effectiveIdentity;
  uvec4 tierTransitionCoverageFlags;
  vec4 continuousCameraLocal;
  vec4 fadeStartHalfExtents;
  vec4 fadeEndHalfExtents;
};

layout(std430, buffer_reference) readonly buffer DDGIFrameBuffer {
  uvec4 activeCountDebugFlagsSampler;
  DDGIVolumeGpuData volumes[kDDGIMaxVolumes];
};
