const uint kDDGIMaxVolumes = 4u;
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
};

layout(std430, buffer_reference) readonly buffer DDGIFrameBuffer {
  uvec4 activeCountDebugFlagsSampler;
  DDGIVolumeGpuData volumes[kDDGIMaxVolumes];
};
