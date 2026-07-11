#extension GL_EXT_buffer_reference : require

layout(location = 0) in vec2 uv;
layout(location = 0) out float outMotionClass;

const float kMotionClassInvalid = 0.0 / 255.0;
const float kMotionClassStaticCameraOnly = 1.0 / 255.0;
const float kMotionClassFull = 2.0 / 255.0;
const float kMotionClassBackgroundRotation = 3.0 / 255.0;

layout(push_constant) uniform MotionClassPushConstants {
  uint depthTexId;
  uint motionTexId;
  uint reactiveTexId;
  uint pointSamplerId;
  uint forceInvalidGeometry;
  uint provenStaticCameraOnly;
}
pc;

void main() {
  const vec2 screenUv = vec2(uv.x, uv.y - 1.0);
  const float depth =
      textureBindless2D(pc.depthTexId, pc.pointSamplerId, screenUv).r;
  if (depth >= 0.999999) {
    outMotionClass = kMotionClassBackgroundRotation;
    return;
  }

  if (pc.forceInvalidGeometry != 0u) {
    outMotionClass = kMotionClassInvalid;
    return;
  }

  const vec2 motion =
      textureBindless2D(pc.motionTexId, pc.pointSamplerId, screenUv).rg;
  if (any(isnan(motion)) || any(isinf(motion)) ||
      any(greaterThan(abs(motion), vec2(8.0)))) {
    outMotionClass = kMotionClassInvalid;
    return;
  }

  const float reactive =
      textureBindless2D(pc.reactiveTexId, pc.pointSamplerId, screenUv).r;
  if (reactive > (0.5 / 255.0)) {
    outMotionClass = kMotionClassInvalid;
    return;
  }

  if (pc.provenStaticCameraOnly != 0u) {
    outMotionClass = kMotionClassStaticCameraOnly;
    return;
  }

  outMotionClass = kMotionClassFull;
}
