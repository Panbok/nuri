#include "nuri/gfx/gpu_types.h"

#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <limits>

namespace nuri {
namespace {

TEST(RasterPipelineStateTest, CanonicalizesSignedZeroForStableKeys) {
  const RasterPipelineState positiveZero = makeRasterPipelineState(
      DepthState{.compareOp = CompareOp::LessEqual,
                 .isDepthWriteEnabled = false},
      true, 0.0f, 0.0f, 0.0f);
  const RasterPipelineState negativeZero = makeRasterPipelineState(
      DepthState{.compareOp = CompareOp::LessEqual,
                 .isDepthWriteEnabled = false},
      true, -0.0f, -0.0f, -0.0f);

  EXPECT_EQ(positiveZero, negativeZero);
  EXPECT_EQ(std::bit_cast<uint32_t>(negativeZero.depthBiasSlope), 0u);
  EXPECT_EQ(std::bit_cast<uint32_t>(negativeZero.depthBiasClamp), 0u);
}

TEST(RasterPipelineStateTest, DisabledBiasErasesIrrelevantValues) {
  const RasterPipelineState state = canonicalRasterPipelineState({
      .compareOp = CompareOp::Equal,
      .depthWrite = false,
      .depthBiasEnable = false,
      .depthBiasConstant = 17,
      .depthBiasSlope = std::numeric_limits<float>::infinity(),
      .depthBiasClamp = std::numeric_limits<float>::quiet_NaN(),
  });

  EXPECT_EQ(state.compareOp, CompareOp::Equal);
  EXPECT_FALSE(state.depthWrite);
  EXPECT_EQ(state.depthBiasConstant, 0);
  EXPECT_EQ(state.depthBiasSlope, 0.0f);
  EXPECT_EQ(state.depthBiasClamp, 0.0f);
}

TEST(RasterPipelineStateTest, LowersFloatConstantToNvrhiIntegralState) {
  const RasterPipelineState roundedPositive = makeRasterPipelineState(
      DepthState{}, true, 1.5f);
  const RasterPipelineState roundedNegative = makeRasterPipelineState(
      DepthState{}, true, -1.5f);
  const RasterPipelineState clamped = makeRasterPipelineState(
      DepthState{}, true, std::numeric_limits<float>::max());

  EXPECT_EQ(roundedPositive.depthBiasConstant, 2);
  EXPECT_EQ(roundedNegative.depthBiasConstant, -2);
  EXPECT_EQ(clamped.depthBiasConstant, std::numeric_limits<int32_t>::max());
}

} // namespace
} // namespace nuri
