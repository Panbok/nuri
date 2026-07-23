#include "tests_pch.h"

#include "nuri/gfx/frame/render_frame_context.h"

#include <limits>

namespace {

TEST(DDGISettingsTests, NamedPresetsOverrideStaleOwnedFields) {
  nuri::RenderSettings settings{};
  settings.ddgi.preset = nuri::DDGIQualityPreset::High;
  settings.ddgi.raysPerProbe = 17u;
  settings.ddgi.maxProbeUpdatesPerFrame = 1u;
  settings.ddgi.maxRayQueriesPerFrame = 34u;
  settings.ddgi.irradianceHysteresis = 0.1f;
  settings.ddgi.distanceHysteresis = 0.2f;
  settings.ddgi.selfShadowBias = 0.75f;

  const nuri::RenderSettings resolved = nuri::resolveRenderSettings(settings);

  EXPECT_EQ(resolved.ddgi.preset, nuri::DDGIQualityPreset::High);
  EXPECT_EQ(resolved.ddgi.raysPerProbe, 256u);
  EXPECT_EQ(resolved.ddgi.maxProbeUpdatesPerFrame, 512u);
  EXPECT_EQ(resolved.ddgi.maxRayQueriesPerFrame, 131'072u);
  EXPECT_FLOAT_EQ(resolved.ddgi.irradianceHysteresis, 0.95f);
  EXPECT_FLOAT_EQ(resolved.ddgi.distanceHysteresis, 0.98f);
  EXPECT_FLOAT_EQ(resolved.ddgi.selfShadowBias, 0.75f);
}

TEST(DDGISettingsTests, CustomSettingsClampEveryNormativeRange) {
  nuri::RenderSettings::DDGISettings settings{};
  settings.preset = nuri::DDGIQualityPreset::Custom;
  settings.raysPerProbe = 0u;
  settings.maxProbeUpdatesPerFrame = 0u;
  settings.maxRayQueriesPerFrame = 0u;
  settings.maxLocalLightsPerHit = 99u;
  settings.maxCandidateIntersectionsPerRay = 1u;
  settings.irradianceHysteresis = std::numeric_limits<float>::infinity();
  settings.distanceHysteresis = -1.0f;
  settings.changeIrradianceHysteresisScale = 2.0f;
  settings.changeDistanceHysteresisScale = -1.0f;
  settings.selfShadowBias = 4.0f;
  settings.multiBounceLuminanceClamp = 0.0f;

  nuri::sanitizeDDGISettings(settings, 4'096u);

  EXPECT_EQ(settings.raysPerProbe, 16u);
  EXPECT_EQ(settings.maxProbeUpdatesPerFrame, 1u);
  EXPECT_EQ(settings.maxRayQueriesPerFrame, 32u);
  EXPECT_EQ(settings.maxLocalLightsPerHit, 16u);
  EXPECT_EQ(settings.maxCandidateIntersectionsPerRay, 8u);
  EXPECT_FLOAT_EQ(settings.irradianceHysteresis, 0.97f);
  EXPECT_FLOAT_EQ(settings.distanceHysteresis, 0.0f);
  EXPECT_FLOAT_EQ(settings.changeIrradianceHysteresisScale, 1.0f);
  EXPECT_FLOAT_EQ(settings.changeDistanceHysteresisScale, 0.0f);
  EXPECT_FLOAT_EQ(settings.selfShadowBias, 2.0f);
  EXPECT_FLOAT_EQ(settings.multiBounceLuminanceClamp, 32.0f);
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
  EXPECT_EQ(settings.raysPerProbe, 64u);
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

} // namespace
