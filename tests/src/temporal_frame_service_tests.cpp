#include "tests_pch.h"

#include <gtest/gtest.h>

#include "nuri/gfx/frame/history_registry.h"
#include "nuri/gfx/frame/presentation_aa_plan.h"
#include "nuri/gfx/frame/temporal_frame_service.h"
#include "nuri/gfx/frame/temporal_motion.h"
#include "nuri/scene/camera.h"

namespace {

using namespace nuri;

[[nodiscard]] PresentationAAPlan requirePlan(const RenderSettings &settings) {
  constexpr PresentationAAGpuCapabilities kTestGpuCapabilities{
      .sample4Color = true,
      .sample4Depth = true,
      .sample8Color = true,
      .sample8Depth = true,
      .depthResolveMin = true,
      .alphaToCoverage = true,
      .sampleRateShading = false,
  };
  auto result = buildPresentationAAPlan(settings, {}, kTestGpuCapabilities);
  EXPECT_FALSE(result.hasError()) << (result.hasError() ? result.error() : "");
  return result.hasError() ? PresentationAAPlan{} : result.value();
}

TEST(TemporalFrameServiceTest,
     PresentationPlanKeepsCoverageReconstructionAndGtaoOrthogonal) {
  RenderSettings settings{};
  settings.ambientOcclusion.mode = AmbientOcclusionMode::GTAO;
  settings.ambientOcclusion.preset = AmbientOcclusionPreset::Ultra;
  settings.ambientOcclusion.temporalAccumulation = true;

  settings.antiAliasing.mode = AntiAliasingMode::None;
  PresentationAAPlan plan = requirePlan(settings);
  EXPECT_EQ(plan.coverage, CoverageMode::Sample1);
  EXPECT_EQ(plan.reconstruction, ColorReconstruction::Off);
  EXPECT_TRUE(plan.gtaoTemporal);
  EXPECT_TRUE(plan.needsMotion);
  EXPECT_TRUE(plan.needsMotionClass);
  EXPECT_TRUE(plan.needsReactiveMask);
  EXPECT_FALSE(plan.jitterScene);

  settings.antiAliasing.mode = AntiAliasingMode::SpatialFallback;
  plan = requirePlan(settings);
  EXPECT_EQ(plan.coverage, CoverageMode::Sample1);
  EXPECT_EQ(plan.reconstruction, ColorReconstruction::Off);
  EXPECT_TRUE(plan.gtaoTemporal);
  EXPECT_TRUE(plan.needsMotion);
  EXPECT_TRUE(plan.needsMotionClass);
  EXPECT_TRUE(plan.needsReactiveMask);
  EXPECT_FALSE(plan.jitterScene);

  settings.antiAliasing.mode = AntiAliasingMode::MSAA4x;
  plan = requirePlan(settings);
  EXPECT_EQ(plan.coverage, CoverageMode::Sample4);
  EXPECT_EQ(plan.reconstruction, ColorReconstruction::Off);
  EXPECT_TRUE(plan.gtaoTemporal);
  EXPECT_TRUE(plan.needsMotion);
  EXPECT_TRUE(plan.needsMotionClass);
  EXPECT_TRUE(plan.needsReactiveMask);
  EXPECT_FALSE(plan.jitterScene);
  EXPECT_EQ(plan.spatialCleanup, SpatialCleanupPoint::Off);

  settings.antiAliasing.mode = AntiAliasingMode::MSAA8x;
  plan = requirePlan(settings);
  EXPECT_EQ(plan.coverage, CoverageMode::Sample8);
  EXPECT_EQ(plan.reconstruction, ColorReconstruction::Off);
  EXPECT_TRUE(plan.gtaoTemporal);
  EXPECT_FALSE(plan.jitterScene);

  settings.antiAliasing.mode = AntiAliasingMode::TAA;
  settings.antiAliasing.temporalProvider =
      TemporalReconstructionProvider::Reference;
  plan = requirePlan(settings);
  EXPECT_EQ(plan.coverage, CoverageMode::Sample1);
  EXPECT_EQ(plan.reconstruction, ColorReconstruction::ReferenceTAA);
  EXPECT_EQ(plan.spatialCleanup, SpatialCleanupPoint::Off);
  EXPECT_TRUE(plan.needsMotion);
  EXPECT_TRUE(plan.needsReactiveMask);
  EXPECT_TRUE(plan.jitterScene);
}

TEST(TemporalFrameServiceTest, PostAAPlanResolvesAndCanonicalizesEveryMode) {
  RenderSettings settings{};
  settings.antiAliasing.mode = AntiAliasingMode::MSAA4x;

  PresentationAAPlan plan = requirePlan(settings);
  EXPECT_FALSE(plan.postAA.requested);
  EXPECT_FALSE(plan.postAA.active);
  EXPECT_EQ(plan.postAA.inactiveReason, PostAAInactiveReason::NotRequested);
  EXPECT_EQ(plan.postAA.specular, PostAASpecularAlgorithm::InheritCurrent);
  EXPECT_EQ(plan.postAA.spatial, PostAASpatialAlgorithm::Off);
  EXPECT_EQ(plan.postAA.resolvedMaterialSpecularAA,
            ResolvedMaterialSpecularAA::LegacyShadingNormalDerivative);

  settings.antiAliasing.postAA.enabled = true;
  settings.antiAliasing.postAA.specular = PostAASpecularAlgorithm::BakedClean;
  settings.antiAliasing.postAA.spatial = PostAASpatialAlgorithm::Smaa1x;
  plan = requirePlan(settings);
  EXPECT_TRUE(plan.postAA.requested);
  EXPECT_TRUE(plan.postAA.active);
  EXPECT_EQ(plan.postAA.inactiveReason, PostAAInactiveReason::None);
  EXPECT_EQ(plan.postAA.specular, PostAASpecularAlgorithm::BakedClean);
  EXPECT_EQ(plan.postAA.spatial, PostAASpatialAlgorithm::Smaa1x);
  EXPECT_EQ(plan.postAA.resolvedMaterialSpecularAA,
            ResolvedMaterialSpecularAA::BakedClean);

  settings.antiAliasing.postAA.specular =
      PostAASpecularAlgorithm::InheritCurrent;
  settings.antiAliasing.postAA.spatial = PostAASpatialAlgorithm::Off;
  plan = requirePlan(settings);
  EXPECT_TRUE(plan.postAA.requested);
  EXPECT_FALSE(plan.postAA.active);
  EXPECT_EQ(plan.postAA.inactiveReason,
            PostAAInactiveReason::NoComponentEnabled);
  EXPECT_EQ(plan.postAA.specular, PostAASpecularAlgorithm::InheritCurrent);
  EXPECT_EQ(plan.postAA.spatial, PostAASpatialAlgorithm::Off);

  settings.antiAliasing.postAA.specular = PostAASpecularAlgorithm::BakedClean;
  settings.antiAliasing.postAA.spatial = PostAASpatialAlgorithm::Smaa1x;
  settings.antiAliasing.mode = AntiAliasingMode::TAA;
  plan = requirePlan(settings);
  EXPECT_TRUE(plan.postAA.requested);
  EXPECT_FALSE(plan.postAA.active);
  EXPECT_EQ(plan.postAA.inactiveReason,
            PostAAInactiveReason::CoverageIsSingleSample);
  EXPECT_EQ(plan.postAA.specular, PostAASpecularAlgorithm::InheritCurrent);
  EXPECT_EQ(plan.postAA.spatial, PostAASpatialAlgorithm::Off);
  EXPECT_EQ(plan.postAA.resolvedMaterialSpecularAA,
            ResolvedMaterialSpecularAA::LegacyShadingNormalDerivative);
}

TEST(TemporalFrameServiceTest,
     PostAADiagnosticsAndTuningSanitizeWithoutContaminatingTemporalModes) {
  RenderSettings settings{};
  settings.antiAliasing.mode = AntiAliasingMode::MSAA8x;
  settings.antiAliasing.postAA.enabled = true;
  settings.antiAliasing.postAA.specular = PostAASpecularAlgorithm::BakedClean;
  settings.antiAliasing.postAA.spatial = PostAASpatialAlgorithm::Off;
  settings.antiAliasing.postAA.materialVarianceScale =
      std::numeric_limits<float>::quiet_NaN();
  settings.antiAliasing.postAA.geometricVarianceScale =
      std::numeric_limits<float>::infinity();
  settings.antiAliasing.postAA.maxSlopeVariance = -1.0f;
  settings.antiAliasing.debug.view = AntiAliasingDebugView::SpecularAAVariance;
  settings.antiAliasing.debug.specularAAOverride =
      SpecularAADebugOverride::ForceOff;

  PresentationAAPlan plan = requirePlan(settings);
  EXPECT_EQ(plan.postAA.resolvedMaterialSpecularAA,
            ResolvedMaterialSpecularAA::Off);
  EXPECT_EQ(plan.postAA.specularAADebugOverride,
            SpecularAADebugOverride::ForceOff);
  EXPECT_EQ(plan.postAA.debugView, AntiAliasingDebugView::SpecularAAVariance);
  EXPECT_FLOAT_EQ(plan.postAA.materialVarianceScale, 1.0f);
  EXPECT_FLOAT_EQ(plan.postAA.geometricVarianceScale, 0.35f);
  EXPECT_FLOAT_EQ(plan.postAA.maxSlopeVariance, 0.0f);

  settings.antiAliasing.postAA.enabled = false;
  plan = requirePlan(settings);
  EXPECT_EQ(plan.postAA.resolvedMaterialSpecularAA,
            ResolvedMaterialSpecularAA::Off);
  EXPECT_EQ(plan.postAA.debugView, AntiAliasingDebugView::None);

  settings.antiAliasing.mode = AntiAliasingMode::TAA;
  settings.antiAliasing.postAA.enabled = true;
  plan = requirePlan(settings);
  EXPECT_EQ(plan.postAA.specularAADebugOverride, SpecularAADebugOverride::None);
  EXPECT_EQ(plan.postAA.debugView, AntiAliasingDebugView::None);
  EXPECT_EQ(plan.postAA.resolvedMaterialSpecularAA,
            ResolvedMaterialSpecularAA::LegacyShadingNormalDerivative);
}

TEST(TemporalFrameServiceTest,
     PostAAChangesDoNotAlterTemporalContinuityProjection) {
  RenderSettings settings{};
  settings.antiAliasing.mode = AntiAliasingMode::MSAA4x;
  PresentationAAPlan baseline = requirePlan(settings);
  settings.antiAliasing.postAA.enabled = true;
  settings.antiAliasing.postAA.specular = PostAASpecularAlgorithm::BakedClean;
  settings.antiAliasing.postAA.spatial = PostAASpatialAlgorithm::Smaa1x;
  settings.antiAliasing.postAA.materialVarianceScale = 1.7f;
  const PresentationAAPlan postAA = requirePlan(settings);

  EXPECT_NE(baseline, postAA);
  EXPECT_TRUE(temporalAAContinuityEquivalent(baseline, postAA));

  Camera camera{};
  TemporalCameraFrameDesc desc{};
  desc.renderExtent = {640u, 360u};
  TemporalFrameService service;
  auto frame = service.prepareFrame(camera, 16.0f / 9.0f, settings.antiAliasing,
                                    baseline, desc, 0u, 0.0, 1.0 / 60.0);
  ASSERT_FALSE(frame.hasError()) << frame.error();
  ASSERT_TRUE(service.commitFrame(0u));
  const uint32_t providerEpoch = service.facts().epochs.providerConfiguration;

  frame = service.prepareFrame(camera, 16.0f / 9.0f, settings.antiAliasing,
                               postAA, desc, 1u, 1.0, 1.0 / 60.0);
  ASSERT_FALSE(frame.hasError()) << frame.error();
  EXPECT_FALSE(hasTemporalResetReason(
      service.facts().resetReasons, TemporalResetReasonFlags::ProviderChange));
  EXPECT_EQ(service.facts().epochs.providerConfiguration, providerEpoch);
}

TEST(TemporalFrameServiceTest,
     HistoryRegistryAdvancesOnlyCommittedWritesAndAvoidsReadWriteAliasing) {
  HistoryRegistry registry;
  auto lease = registry.prepareFrame(0u, 3u);
  ASSERT_FALSE(lease.hasError()) << lease.error();
  EXPECT_FALSE(lease.value().readValid);
  registry.abandonFrame(0u);

  lease = registry.prepareFrame(1u, 3u);
  ASSERT_FALSE(lease.hasError()) << lease.error();
  EXPECT_FALSE(lease.value().readValid);
  ASSERT_TRUE(registry.commitFrame(1u));

  lease = registry.prepareFrame(4u, 3u);
  ASSERT_FALSE(lease.hasError()) << lease.error();
  EXPECT_TRUE(lease.value().readValid);
  EXPECT_EQ(lease.value().readSlot, 1u);
  EXPECT_NE(lease.value().writeSlot, lease.value().readSlot);
  registry.abandonFrame(4u);

  lease = registry.prepareFrame(5u, 3u);
  ASSERT_FALSE(lease.hasError()) << lease.error();
  EXPECT_EQ(lease.value().readSlot, 1u);
  registry.invalidate(HistoryInvalidationReason::ResourceRecreation);
  lease = registry.prepareFrame(6u, 3u);
  ASSERT_FALSE(lease.hasError()) << lease.error();
  EXPECT_FALSE(lease.value().readValid);
  EXPECT_EQ(lease.value().generation, 1u);
}

TEST(TemporalFrameServiceTest,
     ExternalProviderRequiresExplicitCapabilitiesAndMasks) {
  RenderSettings settings{};
  settings.antiAliasing.mode = AntiAliasingMode::TAA;
  settings.antiAliasing.temporalProvider =
      TemporalReconstructionProvider::External;

  auto result = buildPresentationAAPlan(settings);
  ASSERT_TRUE(result.hasError());

  PresentationAAProviderCapabilities capabilities{};
  capabilities.externalTemporal = true;
  capabilities.compositionMask = true;
  result = buildPresentationAAPlan(settings, capabilities);
  ASSERT_FALSE(result.hasError()) << result.error();
  EXPECT_EQ(result.value().reconstruction,
            ColorReconstruction::ExternalTemporal);
  EXPECT_TRUE(result.value().needsReactiveMask);
  EXPECT_TRUE(result.value().needsCompositionMask);
}

TEST(TemporalFrameServiceTest,
     AbandonedFramesDoNotAdvanceCommittedCameraHistory) {
  RenderSettings settings{};
  settings.antiAliasing.mode = AntiAliasingMode::TAA;
  const PresentationAAPlan plan = requirePlan(settings);
  ASSERT_TRUE(plan.valid);

  Camera camera{};
  TemporalCameraFrameDesc desc{};
  desc.renderExtent = {1280u, 720u};
  TemporalFrameService service;

  auto first = service.prepareFrame(camera, 16.0f / 9.0f, settings.antiAliasing,
                                    plan, desc, 10u, 1.0, 1.0 / 60.0);
  ASSERT_FALSE(first.hasError()) << first.error();
  EXPECT_FALSE(first.value().cameraContinuityValid);
  EXPECT_TRUE(hasTemporalResetReason(service.facts().resetReasons,
                                     TemporalResetReasonFlags::FirstUse));
  service.abandonFrame(10u);
  EXPECT_FALSE(service.cameraHistory().initialized);
  EXPECT_EQ(service.facts().renderedFrameSerial, 0u);

  auto submitted =
      service.prepareFrame(camera, 16.0f / 9.0f, settings.antiAliasing, plan,
                           desc, 11u, 2.0, 1.0 / 60.0);
  ASSERT_FALSE(submitted.hasError()) << submitted.error();
  EXPECT_FALSE(submitted.value().cameraContinuityValid);
  EXPECT_TRUE(
      hasTemporalResetReason(service.facts().resetReasons,
                             TemporalResetReasonFlags::SkippedHistoryWrite));
  ASSERT_TRUE(service.commitFrame(11u));
  EXPECT_TRUE(service.cameraHistory().initialized);
  EXPECT_EQ(service.facts().renderedFrameSerial, 1u);

  auto steady = service.prepareFrame(
      camera, 16.0f / 9.0f, settings.antiAliasing, plan, desc, 12u, 2.5, 0.5);
  ASSERT_FALSE(steady.hasError()) << steady.error();
  EXPECT_TRUE(steady.value().cameraContinuityValid);
  EXPECT_DOUBLE_EQ(service.facts().deltaFromCommittedSeconds, 0.5);
  ASSERT_TRUE(service.commitFrame(12u));
  EXPECT_EQ(service.facts().renderedFrameSerial, 2u);
}

TEST(TemporalFrameServiceTest, ProviderAndBackendChangesPublishSeparateEpochs) {
  RenderSettings settings{};
  settings.antiAliasing.mode = AntiAliasingMode::TAA;
  PresentationAAPlan plan = requirePlan(settings);
  Camera camera{};
  TemporalCameraFrameDesc desc{};
  desc.renderExtent = {960u, 540u};
  TemporalFrameService service;

  auto frame = service.prepareFrame(camera, 16.0f / 9.0f, settings.antiAliasing,
                                    plan, desc, 0u, 0.0, 1.0 / 60.0);
  ASSERT_FALSE(frame.hasError()) << frame.error();
  ASSERT_TRUE(service.commitFrame(0u));
  const TemporalEpochs initialEpochs = service.facts().epochs;

  settings.antiAliasing.temporalProvider =
      TemporalReconstructionProvider::Reference;
  plan = requirePlan(settings);
  frame = service.prepareFrame(camera, 16.0f / 9.0f, settings.antiAliasing,
                               plan, desc, 1u, 1.0, 1.0 / 60.0);
  ASSERT_FALSE(frame.hasError()) << frame.error();
  EXPECT_TRUE(frame.value().cameraContinuityValid);
  EXPECT_TRUE(hasTemporalResetReason(service.facts().resetReasons,
                                     TemporalResetReasonFlags::ProviderChange));
  EXPECT_EQ(service.facts().epochs.providerConfiguration,
            initialEpochs.providerConfiguration + 1u);
  ASSERT_TRUE(service.commitFrame(1u));

  service.invalidateBackend();
  frame = service.prepareFrame(camera, 16.0f / 9.0f, settings.antiAliasing,
                               plan, desc, 2u, 2.0, 1.0 / 60.0);
  ASSERT_FALSE(frame.hasError()) << frame.error();
  EXPECT_FALSE(frame.value().cameraContinuityValid);
  EXPECT_TRUE(
      hasTemporalResetReason(service.facts().resetReasons,
                             TemporalResetReasonFlags::BackendRecreation));
  EXPECT_EQ(service.facts().epochs.resourceGeneration, 1u);
}

TEST(TemporalFrameServiceTest,
     AnalyticMotionOracleDetectsPixelScaleAndSignErrors) {
  const glm::uvec2 extent{1000u, 500u};
  const glm::mat4 projection =
      glm::ortho(-10.0f, 10.0f, -5.0f, 5.0f, 0.1f, 100.0f);
  const TemporalMotionEndpoint endpoint = projectTemporalMotionEndpoint(
      glm::vec3(0.04f, 0.0f, -1.0f), glm::vec3(0.0f, 0.0f, -1.0f), projection,
      projection);

  ASSERT_TRUE(endpoint.valid);
  EXPECT_NEAR(endpoint.velocityUv.x * static_cast<float>(extent.x), -2.0f,
              1.0e-4f);
  EXPECT_NEAR(endpoint.velocityUv.y * static_cast<float>(extent.y), 0.0f,
              1.0e-4f);
  EXPECT_NEAR(
      temporalMotionEndpointErrorPixels(endpoint.velocityUv, endpoint, extent),
      0.0f, 1.0e-6f);
  EXPECT_NEAR(
      temporalMotionEndpointErrorPixels(-endpoint.velocityUv, endpoint, extent),
      4.0f, 1.0e-4f);
  EXPECT_NEAR(temporalMotionEndpointErrorPixels(endpoint.velocityUv * 0.5f,
                                                endpoint, extent),
              1.0f, 1.0e-4f);
}

TEST(TemporalFrameServiceTest,
     AnalyticMotionOracleCoversExactQuarterPixelCameraPanAndStaticGate) {
  const glm::uvec2 extent{1000u, 500u};
  const glm::mat4 projection =
      glm::ortho(-10.0f, 10.0f, -5.0f, 5.0f, 0.1f, 100.0f);
  const glm::mat4 previousView{1.0f};
  const glm::mat4 currentView =
      glm::translate(glm::mat4(1.0f), glm::vec3(-0.005f, 0.0f, 0.0f));
  const glm::vec3 worldPosition{0.0f, 0.0f, -1.0f};
  const TemporalMotionEndpoint pan = projectTemporalMotionEndpoint(
      worldPosition, worldPosition, projection * currentView,
      projection * previousView);

  ASSERT_TRUE(pan.valid);
  EXPECT_NEAR(pan.velocityUv.x * static_cast<float>(extent.x), 0.25f, 1.0e-4f);
  EXPECT_NEAR(temporalMotionEndpointErrorPixels(pan.velocityUv, pan, extent),
              0.0f, 1.0e-6f);

  for (int y = -4; y <= 4; ++y) {
    for (int x = -8; x <= 8; ++x) {
      const glm::vec3 samplePosition{static_cast<float>(x) * 0.25f,
                                     static_cast<float>(y) * 0.25f, -2.0f};
      const TemporalMotionEndpoint stable = projectTemporalMotionEndpoint(
          samplePosition, samplePosition, projection, projection);
      ASSERT_TRUE(stable.valid);
      EXPECT_LT(
          temporalMotionEndpointErrorPixels(glm::vec2(0.0f), stable, extent),
          0.01f);
    }
  }
}

} // namespace
