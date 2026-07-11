#include "tests_pch.h"

#include <gtest/gtest.h>

#include "nuri/gfx/frame/external_temporal_provider.h"

namespace {

using namespace nuri;

[[nodiscard]] TextureHandle texture(uint32_t index) {
  return TextureHandle{.index = index, .generation = 1u};
}

TEST(ExternalTemporalProviderTest, FidelityFxIdentityIsExactlyPinned) {
  EXPECT_EQ(kFidelityFxSdkTag, "v1.1.4");
  EXPECT_EQ(kFidelityFxSdkRevision, "c6efa6bf7f2027b3ec94f28578bb5965eabb9e55");
  EXPECT_EQ(kFidelityFxSdkVersion, "1.1.4");
  EXPECT_EQ(kFidelityFxFsrUpscalerVersion, "3.1.4");
  EXPECT_EQ(kFidelityFxLicenseSpdx, "MIT");
}

TEST(ExternalTemporalProviderTest, ProbeRequiresEveryRuntimeCapability) {
  ExternalTemporalProviderProbeDesc desc{.buildRequested = true};
  EXPECT_EQ(probeFidelityFxFsr31(desc).status,
            ExternalTemporalProviderStatus::DependencyMissing);

  desc.dependencyPresent = true;
  EXPECT_EQ(probeFidelityFxFsr31(desc).status,
            ExternalTemporalProviderStatus::BackendUnavailable);

  desc.backendCompiled = true;
  EXPECT_EQ(probeFidelityFxFsr31(desc).status,
            ExternalTemporalProviderStatus::RuntimeUnavailable);

  desc.runtimeLoaded = true;
  desc.reportedProviderVersion = "3.1.5";
  EXPECT_EQ(probeFidelityFxFsr31(desc).status,
            ExternalTemporalProviderStatus::VersionMismatch);

  desc.reportedProviderVersion = kFidelityFxFsrUpscalerVersion;
  const ExternalTemporalProviderProbe probe = probeFidelityFxFsr31(desc);
  EXPECT_EQ(probe.status, ExternalTemporalProviderStatus::Ready);
  EXPECT_TRUE(probe.available);
}

TEST(ExternalTemporalProviderTest, DefaultProviderIsAnInactiveNoOp) {
  std::unique_ptr<ExternalTemporalProvider> provider =
      createExternalTemporalProvider();
  ASSERT_NE(provider, nullptr);
  EXPECT_FALSE(provider->probe().available);
  EXPECT_FALSE(provider->capabilities().available);

  auto plan = provider->prepareFrame({
      .renderExtent = {1280u, 720u},
      .outputExtent = {1280u, 720u},
      .configurationEpoch = 7u,
  });
  ASSERT_FALSE(plan.hasError()) << plan.error();
  EXPECT_FALSE(plan.value().reconstructionActive);
  EXPECT_FALSE(plan.value().sceneJitterActive);
  EXPECT_EQ(plan.value().outputExtent, glm::uvec2(1280u, 720u));
  EXPECT_EQ(plan.value().configurationEpoch, 7u);

  auto execute = provider->execute(RecordingContextHandle{},
                                   ExternalTemporalProviderExecuteDesc{});
  ASSERT_TRUE(execute.hasError());
  EXPECT_NE(execute.error().find("unavailable"), std::string::npos);
}

TEST(ExternalTemporalProviderTest,
     ExecuteValidationRejectsInvalidMotionAndAcceptsNativeAaInputs) {
  const ExternalTemporalProviderProbe readyProbe = probeFidelityFxFsr31({
      .buildRequested = true,
      .dependencyPresent = true,
      .backendCompiled = true,
      .runtimeLoaded = true,
      .reportedProviderVersion = kFidelityFxFsrUpscalerVersion,
  });
  const ExternalTemporalProviderCapabilities capabilities =
      fidelityFxFsr31Capabilities(readyProbe);
  ASSERT_TRUE(capabilities.available);
  EXPECT_FALSE(capabilities.explicitMotionValidity);

  ExternalTemporalProviderExecuteDesc desc{
      .sceneColor = texture(1u),
      .sceneDepth = texture(2u),
      .motionVectors = texture(3u),
      .reactiveMask = texture(4u),
      .compositionMask = texture(5u),
      .output = texture(6u),
      .renderExtent = {1920u, 1080u},
      .outputExtent = {1920u, 1080u},
      .frameTimeDeltaMilliseconds = 16.6667f,
  };
  auto result = validateExternalTemporalExecuteDesc(desc, capabilities);
  ASSERT_FALSE(result.hasError()) << result.error();

  desc.invalidMotionCoveragePercent = 0.01f;
  result = validateExternalTemporalExecuteDesc(desc, capabilities);
  ASSERT_TRUE(result.hasError());
  EXPECT_NE(result.error().find("invalid motion"), std::string::npos);
}

} // namespace
