#include "tests_pch.h"

#include "nuri/gfx/frame/render_frame_context.h"

#include <limits>

namespace {

TEST(DDGISettingsTests, NamedPresetsOverrideStaleOwnedFields) {
  nuri::RenderSettings settings{};
  settings.ddgi.preset = nuri::DDGIQualityPreset::High;
  settings.ddgi.raysPerProbe = 17u;
  settings.ddgi.maxProbeUpdatesPerFrame = 1u;
  settings.ddgi.maxRadianceProbeUpdatesPerFrame = 1u;
  settings.ddgi.maxMaintenanceProbeUpdatesPerFrame = 1u;
  settings.ddgi.maxRayQueriesPerFrame = 34u;
  settings.ddgi.irradianceHysteresis = 0.1f;
  settings.ddgi.distanceHysteresis = 0.2f;
  settings.ddgi.selfShadowBias = 0.75f;

  const nuri::RenderSettings resolved = nuri::resolveRenderSettings(settings);

  EXPECT_EQ(resolved.ddgi.preset, nuri::DDGIQualityPreset::High);
  EXPECT_EQ(resolved.ddgi.requestedPreset, nuri::DDGIQualityPreset::High);
  EXPECT_EQ(resolved.ddgi.raysPerProbe, 256u);
  EXPECT_EQ(resolved.ddgi.maxProbeUpdatesPerFrame, 512u);
  EXPECT_EQ(resolved.ddgi.maxRadianceProbeUpdatesPerFrame, 128u);
  EXPECT_EQ(resolved.ddgi.maxMaintenanceProbeUpdatesPerFrame, 64u);
  EXPECT_EQ(resolved.ddgi.maxRayQueriesPerFrame, 131'072u);
  EXPECT_FLOAT_EQ(resolved.ddgi.irradianceHysteresis, 0.95f);
  EXPECT_FLOAT_EQ(resolved.ddgi.distanceHysteresis, 0.98f);
  EXPECT_FLOAT_EQ(resolved.ddgi.selfShadowBias, 0.25f);
}

TEST(DDGISettingsTests, QualityAndCoveragePresetsResolveIndependently) {
  nuri::RenderSettings settings{};
  settings.ddgi.preset = nuri::DDGIQualityPreset::High;
  settings.ddgi.coveragePreset = nuri::DDGICoveragePreset::Automatic;
  settings.ddgi.coverage.cascadeCount = 1u;
  settings.ddgi.coverage.requestedNearSpacing = {99.0f, 99.0f, 99.0f};

  const nuri::RenderSettings resolved = nuri::resolveRenderSettings(settings);

  EXPECT_EQ(resolved.ddgi.preset, nuri::DDGIQualityPreset::High);
  EXPECT_EQ(resolved.ddgi.raysPerProbe, 256u);
  EXPECT_EQ(resolved.ddgi.coveragePreset, nuri::DDGICoveragePreset::Automatic);
  EXPECT_EQ(resolved.ddgi.requestedCoveragePreset,
            nuri::DDGICoveragePreset::Automatic);
  EXPECT_EQ(resolved.ddgi.coverage.mode, nuri::DDGICoverageMode::SceneFit);
  EXPECT_EQ(resolved.ddgi.coverage.cascadeCount, 3u);
  EXPECT_FLOAT_EQ(resolved.ddgi.coverage.requestedNearSpacing.x, 2.0f);
}

TEST(DDGISettingsTests, LegacyRawCoverageBecomesCustomWithoutChangingQuality) {
  nuri::RenderSettings settings{};
  settings.ddgi.preset = nuri::DDGIQualityPreset::High;
  settings.ddgi.coverage.mode = nuri::DDGICoverageMode::Hybrid;
  settings.ddgi.coverage.cascadeCount = 4u;

  const nuri::RenderSettings resolved = nuri::resolveRenderSettings(settings);

  EXPECT_EQ(resolved.ddgi.preset, nuri::DDGIQualityPreset::High);
  EXPECT_EQ(resolved.ddgi.requestedPreset, nuri::DDGIQualityPreset::High);
  EXPECT_EQ(resolved.ddgi.coveragePreset, nuri::DDGICoveragePreset::Custom);
  EXPECT_EQ(resolved.ddgi.requestedCoveragePreset,
            nuri::DDGICoveragePreset::Authored);
  EXPECT_EQ(resolved.ddgi.coverage.mode, nuri::DDGICoverageMode::Hybrid);
  EXPECT_EQ(resolved.ddgi.coverage.cascadeCount, 4u);
}

TEST(DDGISettingsTests, CustomSettingsClampEveryNormativeRange) {
  nuri::RenderSettings::DDGISettings settings{};
  settings.preset = nuri::DDGIQualityPreset::Custom;
  settings.raysPerProbe = 0u;
  settings.maxProbeUpdatesPerFrame = 0u;
  settings.maxRadianceProbeUpdatesPerFrame = 99u;
  settings.maxMaintenanceProbeUpdatesPerFrame = 99u;
  settings.maxRayQueriesPerFrame = 0u;
  settings.maxLocalLightsPerHit = 99u;
  settings.maxCandidateIntersectionsPerRay = 1u;
  settings.irradianceHysteresis = std::numeric_limits<float>::infinity();
  settings.distanceHysteresis = -1.0f;
  settings.changeIrradianceHysteresisScale = 2.0f;
  settings.changeDistanceHysteresisScale = -1.0f;
  settings.selfShadowBias = 4.0f;
  settings.primaryProbeBias = 1.0f;
  settings.localShadowBias = 4.0f;
  settings.directionalShadowBias = -1.0f;
  settings.classificationBias = std::numeric_limits<float>::quiet_NaN();
  settings.multiBounceLuminanceClamp = 0.0f;

  nuri::sanitizeDDGISettings(settings, 4'096u);

  EXPECT_EQ(settings.raysPerProbe, 16u);
  EXPECT_EQ(settings.maxProbeUpdatesPerFrame, 1u);
  EXPECT_EQ(settings.maxRadianceProbeUpdatesPerFrame, 1u);
  EXPECT_EQ(settings.maxMaintenanceProbeUpdatesPerFrame, 1u);
  EXPECT_EQ(settings.maxRayQueriesPerFrame, 32u);
  EXPECT_EQ(settings.maxLocalLightsPerHit, 16u);
  EXPECT_EQ(settings.maxCandidateIntersectionsPerRay, 8u);
  EXPECT_FLOAT_EQ(settings.irradianceHysteresis, 0.97f);
  EXPECT_FLOAT_EQ(settings.distanceHysteresis, 0.0f);
  EXPECT_FLOAT_EQ(settings.changeIrradianceHysteresisScale, 1.0f);
  EXPECT_FLOAT_EQ(settings.changeDistanceHysteresisScale, 0.0f);
  EXPECT_FLOAT_EQ(settings.selfShadowBias, 0.25f);
  EXPECT_FLOAT_EQ(settings.primaryProbeBias, 0.25f);
  EXPECT_FLOAT_EQ(settings.localShadowBias, 2.0f);
  EXPECT_FLOAT_EQ(settings.directionalShadowBias, 0.0f);
  EXPECT_FLOAT_EQ(settings.classificationBias, 0.30f);
  EXPECT_FLOAT_EQ(settings.multiBounceLuminanceClamp, 32.0f);
}

TEST(DDGISettingsTests, PersistedRadianceClampRemainsHalfFloatRepresentable) {
  nuri::RenderSettings::DDGISettings settings{};
  settings.preset = nuri::DDGIQualityPreset::Custom;
  settings.multiBounceLuminanceClamp =
      nuri::kMaxDDGIPersistedRadianceLuminance * 4.0f;

  nuri::sanitizeDDGISettings(settings, 4'096u);

  EXPECT_FLOAT_EQ(settings.multiBounceLuminanceClamp,
                  nuri::kMaxDDGIPersistedRadianceLuminance);
}

TEST(DDGISettingsTests, RayQueryCapAccountsForReservedSecondaryQueries) {
  nuri::RenderSettings::DDGISettings settings{};
  settings.preset = nuri::DDGIQualityPreset::Custom;
  settings.raysPerProbe = 1024u;
  settings.maxRayQueriesPerFrame = std::numeric_limits<uint32_t>::max();

  nuri::sanitizeDDGISettings(settings, 8'192u);

  EXPECT_EQ(settings.maxRayQueriesPerFrame, 8'192u);
  settings.maxRayQueriesPerFrame = 1u;
  nuri::sanitizeDDGISettings(settings, 8'192u);
  EXPECT_EQ(settings.maxRayQueriesPerFrame, 2'048u);
}

TEST(DDGISettingsTests, ConvertingNamedPresetToCustomCopiesResolvedSeed) {
  nuri::RenderSettings::DDGISettings settings{};
  settings.preset = nuri::DDGIQualityPreset::Low;
  settings.raysPerProbe = 999u;

  nuri::copyDDGIQualityPresetToCustom(settings);

  EXPECT_EQ(settings.preset, nuri::DDGIQualityPreset::Custom);
  EXPECT_EQ(settings.requestedPreset, nuri::DDGIQualityPreset::Low);
  EXPECT_EQ(settings.raysPerProbe, 64u);
  EXPECT_EQ(settings.maxRadianceProbeUpdatesPerFrame, 512u);
  EXPECT_EQ(settings.maxMaintenanceProbeUpdatesPerFrame, 64u);
  EXPECT_EQ(settings.maxRayQueriesPerFrame, 32'768u);
}

TEST(DDGISettingsTests, RequestedEpochRemainsPendingUntilConsumedSubmission) {
  nuri::DDGICommandEpochs requested{
      .resetHistory = 3u, .forceFullUpdate = 7u, .rebuildRayTracingScene = 11u};
  nuri::DDGICommandEpochs consumed{
      .resetHistory = 2u, .forceFullUpdate = 7u, .rebuildRayTracingScene = 10u};

  EXPECT_TRUE(
      nuri::ddgiEpochIsPending(requested.resetHistory, consumed.resetHistory));
  EXPECT_FALSE(nuri::ddgiEpochIsPending(requested.forceFullUpdate,
                                        consumed.forceFullUpdate));
  EXPECT_TRUE(nuri::ddgiEpochIsPending(requested.rebuildRayTracingScene,
                                       consumed.rebuildRayTracingScene));

  consumed.resetHistory = requested.resetHistory;
  EXPECT_FALSE(
      nuri::ddgiEpochIsPending(requested.resetHistory, consumed.resetHistory));
}

TEST(DDGISettingsTests, ProductProfileFingerprintCoversResolvedPolicyFamilies) {
  const nuri::RenderSettings::DDGISettings baseline{};
  const uint64_t fingerprint = nuri::ddgiProductProfileFingerprint(baseline);

  auto behavior = baseline;
  behavior.maxLocalLightsPerHit += 1u;
  EXPECT_NE(nuri::ddgiProductProfileFingerprint(behavior), fingerprint);

  auto radiometric = baseline;
  radiometric.directionalShadowBias += 0.125f;
  EXPECT_NE(nuri::ddgiProductProfileFingerprint(radiometric), fingerprint);

  auto gather = baseline;
  gather.traceMultiBounceGatherVariant =
      nuri::DDGISurfaceGatherVariant::Candidates;
  EXPECT_NE(nuri::ddgiProductProfileFingerprint(gather), fingerprint);

  auto coveragePolicy = baseline;
  coveragePolicy.coverage.constraintPolicy =
      nuri::DDGICoverageConstraintPolicy::PreserveNearSpacing;
  EXPECT_NE(nuri::ddgiProductProfileFingerprint(coveragePolicy), fingerprint);

  auto coverageGeometry = baseline;
  coverageGeometry.coverage.requestedNearSpacing.x += 0.25f;
  EXPECT_NE(nuri::ddgiProductProfileFingerprint(coverageGeometry), fingerprint);

  auto authoredBounds = baseline;
  authoredBounds.coverage.authoredBounds.valid = true;
  authoredBounds.coverage.authoredBounds.bounds.max_.x = 42.0f;
  EXPECT_NE(nuri::ddgiProductProfileFingerprint(authoredBounds), fingerprint);
}

TEST(DDGISettingsTests, UniformLightSubsetHasExactPeriodCoverage) {
  // Pixels and timing cannot deterministically prove that every packed light
  // retains the same long-run inclusion probability. This pure selector owns
  // that radiometric sampling invariant.
  for (uint32_t total = 1u; total <= 32u; ++total) {
    for (uint32_t samples = 1u; samples <= total; ++samples) {
      std::vector<uint32_t> inclusions(total, 0u);
      for (uint64_t sequence = 0u; sequence < total; ++sequence) {
        std::vector<bool> selected(total, false);
        for (uint32_t sample = 0u; sample < samples; ++sample) {
          const uint32_t index =
              nuri::ddgiUniformSubsetIndex(total, samples, sequence, sample);
          ASSERT_LT(index, total);
          EXPECT_FALSE(selected[index]);
          selected[index] = true;
          ++inclusions[index];
        }
      }
      for (const uint32_t inclusionCount : inclusions) {
        EXPECT_EQ(inclusionCount, samples);
      }
    }
  }
}

} // namespace
