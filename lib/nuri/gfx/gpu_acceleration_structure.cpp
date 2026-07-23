#include "nuri/gfx/gpu_descriptors.h"

#include <cmath>
#include <limits>

namespace nuri {
namespace {

[[nodiscard]] bool
buildFlagsValid(AccelerationStructureBuildFlags flags) noexcept {
  return !(hasAccelerationStructureBuildFlag(
               flags, AccelerationStructureBuildFlags::PreferFastTrace) &&
           hasAccelerationStructureBuildFlag(
               flags, AccelerationStructureBuildFlags::PreferFastBuild));
}

[[nodiscard]] bool
transformFinite(const AccelerationStructureTransform &transform) noexcept {
  for (const float value : transform.rowMajor3x4) {
    if (!std::isfinite(value)) {
      return false;
    }
  }
  return true;
}

} // namespace

AccelerationStructureValidationError
validateBlasCreateDesc(const BlasCreateDesc &desc,
                       const RayTracingCapabilities &caps) noexcept {
  if (!caps.accelerationStructure || !caps.bufferDeviceAddress) {
    return {AccelerationStructureValidationReason::Unsupported, 0u};
  }
  if (!buildFlagsValid(desc.buildFlags)) {
    return {AccelerationStructureValidationReason::ConflictingBuildPreference,
            0u};
  }
  if (desc.geometries.empty()) {
    return {AccelerationStructureValidationReason::EmptyGeometry, 0u};
  }
  if (desc.geometries.size() > caps.maxGeometryCount) {
    return {AccelerationStructureValidationReason::CapacityLimitExceeded, 0u};
  }
  uint64_t totalPrimitiveCount = 0u;
  for (uint32_t index = 0u; index < desc.geometries.size(); ++index) {
    const AccelerationStructureTriangleGeometryDesc &geometry =
        desc.geometries[index];
    if (!nuri::isValid(geometry.vertexBuffer) ||
        !nuri::isValid(geometry.indexBuffer)) {
      return {AccelerationStructureValidationReason::InvalidBuffer, index};
    }
    if (nuri::isValid(geometry.transformBuffer)) {
      return {AccelerationStructureValidationReason::UnsupportedTransformBuffer,
              index};
    }
    if (geometry.vertexFormat != Format::RGB32_FLOAT) {
      return {AccelerationStructureValidationReason::UnsupportedVertexFormat,
              index};
    }
    if (geometry.vertexCount == 0u || geometry.vertexStrideBytes < 12u ||
        geometry.vertexStrideBytes % 4u != 0u ||
        geometry.vertexByteOffset % 4u != 0u) {
      return {AccelerationStructureValidationReason::InvalidVertexRange, index};
    }
    const uint32_t indexStride =
        geometry.indexFormat == IndexFormat::U16 ? 2u : 4u;
    if (geometry.indexFormat == IndexFormat::Count ||
        geometry.indexCount == 0u || geometry.indexCount % 3u != 0u ||
        geometry.indexByteOffset % indexStride != 0u) {
      return {AccelerationStructureValidationReason::InvalidIndexRange, index};
    }
    totalPrimitiveCount += geometry.indexCount / 3u;
    if (totalPrimitiveCount > caps.maxPrimitiveCount) {
      return {AccelerationStructureValidationReason::CapacityLimitExceeded,
              index};
    }
  }
  return {};
}

AccelerationStructureValidationError
validateTlasCreateDesc(const TlasCreateDesc &desc,
                       const RayTracingCapabilities &caps) noexcept {
  if (!caps.accelerationStructure || !caps.bufferDeviceAddress) {
    return {AccelerationStructureValidationReason::Unsupported, 0u};
  }
  if (!buildFlagsValid(desc.buildFlags)) {
    return {AccelerationStructureValidationReason::ConflictingBuildPreference,
            0u};
  }
  if (desc.maxInstanceCount == 0u ||
      desc.maxInstanceCount > caps.maxInstanceCount) {
    return {AccelerationStructureValidationReason::CapacityLimitExceeded, 0u};
  }
  return {};
}

AccelerationStructureValidationError validateAccelerationStructureInstances(
    std::span<const AccelerationStructureInstanceDesc> instances,
    uint32_t maximumCount) noexcept {
  if (instances.size() > maximumCount) {
    return {AccelerationStructureValidationReason::CapacityLimitExceeded, 0u};
  }
  for (uint32_t index = 0u; index < instances.size(); ++index) {
    const AccelerationStructureInstanceDesc &instance = instances[index];
    if (!nuri::isValid(instance.bottomLevel) || instance.mask == 0u ||
        instance.customIndex > 0x00ff'ffffu ||
        !transformFinite(instance.transform)) {
      return {AccelerationStructureValidationReason::InvalidInstance, index};
    }
  }
  return {};
}

} // namespace nuri
