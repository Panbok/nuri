#include "tests_pch.h"

#include "nuri/gfx/gpu_descriptors.h"

#include <array>
#include <limits>

namespace {

[[nodiscard]] nuri::RayTracingCapabilities supportedCaps() {
  return {.accelerationStructure = true,
          .rayQuery = true,
          .bufferDeviceAddress = true,
          .rayTracingPipeline = false,
          .maxGeometryCount = 64u,
          .maxInstanceCount = 1024u,
          .maxPrimitiveCount = 1'000'000u,
          .minScratchOffsetAlignment = 256u};
}

[[nodiscard]] nuri::AccelerationStructureTriangleGeometryDesc triangle() {
  return {.vertexBuffer = nuri::BufferHandle{1u, 1u},
          .indexBuffer = nuri::BufferHandle{2u, 1u},
          .vertexFormat = nuri::Format::RGB32_FLOAT,
          .indexFormat = nuri::IndexFormat::U32,
          .vertexByteOffset = 0u,
          .indexByteOffset = 0u,
          .vertexStrideBytes = 16u,
          .vertexCount = 3u,
          .indexCount = 3u};
}

TEST(AccelerationStructureContractTests,
     RejectsMutuallyExclusiveBuildPreferences) {
  const std::array geometries{triangle()};
  const nuri::BlasCreateDesc desc{
      .geometries = geometries,
      .buildFlags = nuri::AccelerationStructureBuildFlags::PreferFastTrace |
                    nuri::AccelerationStructureBuildFlags::PreferFastBuild};

  const auto error = nuri::validateBlasCreateDesc(desc, supportedCaps());

  EXPECT_EQ(error.reason,
            nuri::AccelerationStructureValidationReason::
                ConflictingBuildPreference);
}

TEST(AccelerationStructureContractTests, ValidatesExactTriangleInputContract) {
  auto geometry = triangle();
  const std::array geometries{geometry};
  EXPECT_EQ(nuri::validateBlasCreateDesc({.geometries = geometries},
                                        supportedCaps())
                .reason,
            nuri::AccelerationStructureValidationReason::None);

  geometry.vertexFormat = nuri::Format::RGBA32_FLOAT;
  const std::array invalidGeometries{geometry};
  EXPECT_EQ(nuri::validateBlasCreateDesc({.geometries = invalidGeometries},
                                        supportedCaps())
                .reason,
            nuri::AccelerationStructureValidationReason::
                UnsupportedVertexFormat);
}

TEST(AccelerationStructureContractTests, RejectsUnsupportedAndOverCapacityTLAS) {
  nuri::RayTracingCapabilities caps{};
  EXPECT_EQ(nuri::validateTlasCreateDesc({.maxInstanceCount = 1u}, caps).reason,
            nuri::AccelerationStructureValidationReason::Unsupported);

  caps = supportedCaps();
  EXPECT_EQ(nuri::validateTlasCreateDesc({.maxInstanceCount = 1025u}, caps)
                .reason,
            nuri::AccelerationStructureValidationReason::
                CapacityLimitExceeded);
}

TEST(AccelerationStructureContractTests, ValidatesInstanceIdentityAndMask) {
  nuri::AccelerationStructureInstanceDesc instance{};
  instance.bottomLevel = nuri::AccelerationStructureHandle{3u, 1u};
  const std::array instances{instance};
  EXPECT_EQ(nuri::validateAccelerationStructureInstances(instances, 1u).reason,
            nuri::AccelerationStructureValidationReason::None);

  instance.mask = 0u;
  const std::array invalidInstances{instance};
  EXPECT_EQ(nuri::validateAccelerationStructureInstances(invalidInstances, 1u)
                .reason,
            nuri::AccelerationStructureValidationReason::InvalidInstance);
}

TEST(AccelerationStructureContractTests, TypedHandlesRemainDistinct) {
  static_assert(!std::is_same_v<nuri::AccelerationStructureHandle,
                                nuri::BufferHandle>);
  EXPECT_TRUE(nuri::isValid(nuri::AccelerationStructureHandle{1u, 1u}));
  EXPECT_FALSE(nuri::isValid(nuri::AccelerationStructureHandle{}));
}

} // namespace
