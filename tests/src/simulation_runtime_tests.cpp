#include "tests_pch.h"

#include "nuri/gfx/sim/simulation_gpu_context.h"
#include "nuri/scene/render_scene.h"
#include "nuri/scene_runtime/scene_runtime_host.h"
#include "nuri/sim/simulation_scheduler.h"

#include <array>

namespace {

[[nodiscard]] nuri::SimulationDesc
makeNodeSimulationDesc(nuri::NodeId node,
                       std::span<const std::byte> initialParams = {}) {
  nuri::SimulationDesc desc;
  desc.kind = nuri::SimulationKind::Custom;
  desc.debugName = "TestSim";
  desc.binding.primaryTarget = nuri::SimulationBindingTarget::makeNode(node);
  desc.initialParams = initialParams;
  return desc;
}

[[nodiscard]] bool bytesEqual(std::span<const std::byte> lhs,
                              std::span<const std::byte> rhs) {
  return lhs.size() == rhs.size() &&
         std::equal(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
}

[[nodiscard]] glm::mat4 translate(const glm::vec3 &value) {
  return glm::translate(glm::mat4(1.0f), value);
}

} // namespace

TEST(SimulationRuntimeTests, ControllerReusesSlotsWithNewGeneration) {
  nuri::RenderScene scene;
  nuri::SceneRuntimeHost runtime;
  runtime.bindScene(&scene);

  auto firstResult = runtime.simulations().createSimulation(
      makeNodeSimulationDesc(scene.graph().rootNode()));
  ASSERT_FALSE(firstResult.hasError()) << firstResult.error();
  const nuri::SimulationHandle first = firstResult.value();

  EXPECT_TRUE(runtime.simulations().destroySimulation(first));
  EXPECT_FALSE(runtime.simulations().destroySimulation(first));

  auto secondResult = runtime.simulations().createSimulation(
      makeNodeSimulationDesc(scene.graph().rootNode()));
  ASSERT_FALSE(secondResult.hasError()) << secondResult.error();
  const nuri::SimulationHandle second = secondResult.value();

  EXPECT_EQ(nuri::indexOf(first), nuri::indexOf(second));
  EXPECT_NE(nuri::generationOf(first), nuri::generationOf(second));

  nuri::SimulationState state = nuri::SimulationState::Stopped;
  EXPECT_FALSE(runtime.simulations().getState(first, state));
  EXPECT_TRUE(runtime.simulations().getState(second, state));
  EXPECT_EQ(state, nuri::SimulationState::Running);
}

TEST(SimulationRuntimeTests, RebindingNonEmptySceneRebuildsBindingTable) {
  nuri::RenderScene firstScene;
  nuri::RenderScene secondScene;

  nuri::SceneRuntimeHost runtime;
  runtime.bindScene(&firstScene);
  auto firstSimulation = runtime.simulations().createSimulation(
      makeNodeSimulationDesc(firstScene.graph().rootNode()));
  ASSERT_FALSE(firstSimulation.hasError()) << firstSimulation.error();

  runtime.bindScene(&secondScene);
  auto secondSimulation = runtime.simulations().createSimulation(
      makeNodeSimulationDesc(secondScene.graph().rootNode()));
  ASSERT_FALSE(secondSimulation.hasError()) << secondSimulation.error();

  nuri::SimulationStats stats;
  ASSERT_TRUE(runtime.simulations().getStats(secondSimulation.value(), stats));
  EXPECT_FALSE(stats.faulted);
}

TEST(SimulationRuntimeTests, TickHonorsPauseResumeDisableAndSingleStep) {
  nuri::RenderScene scene;
  nuri::SceneRuntimeHost runtime;
  runtime.bindScene(&scene);

  auto createResult = runtime.simulations().createSimulation(
      makeNodeSimulationDesc(scene.graph().rootNode()));
  ASSERT_FALSE(createResult.hasError()) << createResult.error();
  const nuri::SimulationHandle handle = createResult.value();

  const double fixedDelta = 1.0 / 60.0;
  nuri::SimulationTickResult tickResult =
      runtime.tick({.frameDeltaSeconds = fixedDelta,
                    .absoluteTimeSeconds = fixedDelta,
                    .frameIndex = 1u});
  EXPECT_EQ(tickResult.executedSteps, 1u);
  EXPECT_TRUE(tickResult.anySimulationRan);

  nuri::SimulationStats stats;
  ASSERT_TRUE(runtime.simulations().getStats(handle, stats));
  EXPECT_EQ(stats.executedStepCount, 1u);
  EXPECT_EQ(stats.executedSubstepCount, 1u);
  EXPECT_EQ(stats.phaseExecutionCounts[0], 1u);
  EXPECT_EQ(stats.phaseExecutionCounts[1], 1u);
  EXPECT_EQ(stats.phaseExecutionCounts[2], 1u);
  EXPECT_EQ(stats.phaseExecutionCounts[3], 1u);
  EXPECT_EQ(stats.phaseExecutionCounts[4], 1u);

  EXPECT_TRUE(runtime.simulations().pause(handle));
  tickResult = runtime.tick({.frameDeltaSeconds = fixedDelta,
                             .absoluteTimeSeconds = fixedDelta * 2.0,
                             .frameIndex = 2u});
  EXPECT_EQ(tickResult.executedSteps, 1u);
  ASSERT_TRUE(runtime.simulations().getStats(handle, stats));
  EXPECT_EQ(stats.executedStepCount, 1u);

  EXPECT_TRUE(runtime.simulations().requestSingleStep(handle));
  tickResult = runtime.tick({.frameDeltaSeconds = fixedDelta,
                             .absoluteTimeSeconds = fixedDelta * 3.0,
                             .frameIndex = 3u});
  EXPECT_EQ(tickResult.executedSteps, 1u);
  ASSERT_TRUE(runtime.simulations().getStats(handle, stats));
  EXPECT_EQ(stats.executedStepCount, 2u);
  EXPECT_EQ(stats.executedSubstepCount, 2u);
  EXPECT_TRUE(stats.paused);

  nuri::SimulationState state = nuri::SimulationState::Stopped;
  ASSERT_TRUE(runtime.simulations().getState(handle, state));
  EXPECT_EQ(state, nuri::SimulationState::Paused);

  EXPECT_TRUE(runtime.simulations().setSubstepCount(handle, 2u));
  EXPECT_TRUE(runtime.simulations().setSolverIterationCount(handle, 3u));
  EXPECT_TRUE(runtime.simulations().resume(handle));
  tickResult = runtime.tick({.frameDeltaSeconds = fixedDelta,
                             .absoluteTimeSeconds = fixedDelta * 4.0,
                             .frameIndex = 4u});
  EXPECT_EQ(tickResult.executedSteps, 1u);
  ASSERT_TRUE(runtime.simulations().getStats(handle, stats));
  EXPECT_EQ(stats.executedStepCount, 3u);
  EXPECT_EQ(stats.executedSubstepCount, 4u);
  EXPECT_EQ(stats.phaseExecutionCounts[0], 3u);
  EXPECT_EQ(stats.phaseExecutionCounts[1], 4u);
  EXPECT_EQ(stats.phaseExecutionCounts[2], 8u);
  EXPECT_EQ(stats.phaseExecutionCounts[3], 4u);
  EXPECT_EQ(stats.phaseExecutionCounts[4], 3u);

  EXPECT_TRUE(runtime.simulations().setEnabled(handle, false));
  EXPECT_FALSE(runtime.simulations().requestSingleStep(handle));
  tickResult = runtime.tick({.frameDeltaSeconds = fixedDelta,
                             .absoluteTimeSeconds = fixedDelta * 5.0,
                             .frameIndex = 5u});
  EXPECT_EQ(tickResult.executedSteps, 1u);
  ASSERT_TRUE(runtime.simulations().getStats(handle, stats));
  EXPECT_EQ(stats.executedStepCount, 3u);
  EXPECT_FALSE(stats.paused);
  EXPECT_FALSE(stats.enabled);
}

TEST(SimulationRuntimeTests, ParamsRoundTripAndValidationBehaveAsExpected) {
  nuri::RenderScene scene;
  nuri::SceneRuntimeHost runtime;
  runtime.bindScene(&scene);

  constexpr std::array<std::byte, 3u> kInitialParams = {
      std::byte{0x01}, std::byte{0x02}, std::byte{0x03}};
  auto createResult = runtime.simulations().createSimulation(
      makeNodeSimulationDesc(scene.graph().rootNode(), kInitialParams));
  ASSERT_FALSE(createResult.hasError()) << createResult.error();
  const nuri::SimulationHandle handle = createResult.value();

  nuri::SimulationDesc desc;
  ASSERT_TRUE(runtime.simulations().getDesc(handle, desc));
  EXPECT_TRUE(bytesEqual(desc.initialParams, kInitialParams));

  constexpr std::array<std::byte, 2u> kUpdatedParams = {std::byte{0x0A},
                                                        std::byte{0x0B}};
  EXPECT_TRUE(runtime.simulations().setParams(handle, kUpdatedParams));
  ASSERT_TRUE(runtime.simulations().getDesc(handle, desc));
  EXPECT_TRUE(bytesEqual(desc.initialParams, kUpdatedParams));

  EXPECT_TRUE(
      runtime.simulations().setParams(handle, std::span<const std::byte>{}));
  ASSERT_TRUE(runtime.simulations().getDesc(handle, desc));
  EXPECT_TRUE(desc.initialParams.empty());

  nuri::SimulationStats stats;
  ASSERT_TRUE(runtime.simulations().getStats(handle, stats));
  EXPECT_EQ(stats.paramsSizeBytes, 0u);

  EXPECT_TRUE(runtime.simulations().setTimeScale(handle, 0.0f));
  EXPECT_FALSE(runtime.simulations().setTimeScale(handle, -1.0f));

  nuri::SimulationDesc invalidDesc;
  invalidDesc.kind = nuri::SimulationKind::Custom;
  invalidDesc.binding.primaryTarget =
      nuri::SimulationBindingTarget::makeNode(scene.graph().rootNode());
  invalidDesc.timeScale = -1.0f;
  const auto invalidResult =
      runtime.simulations().createSimulation(invalidDesc);
  EXPECT_TRUE(invalidResult.hasError());
}

TEST(SimulationRuntimeTests,
     SchedulerConfigHonorsMaxStepsAndAccumulationClamp) {
  nuri::RenderScene scene;
  nuri::SceneRuntimeHost runtime;
  runtime.bindScene(&scene);

  auto createResult = runtime.simulations().createSimulation(
      makeNodeSimulationDesc(scene.graph().rootNode()));
  ASSERT_FALSE(createResult.hasError()) << createResult.error();
  const nuri::SimulationHandle handle = createResult.value();

  nuri::SimulationScheduler scheduler;
  scheduler.setConfig({
      .fixedDeltaSeconds = 0.1,
      .maxStepsPerFrame = 2u,
      .maxAccumulatedSeconds = 1.0,
      .allowFrameDropping = true,
  });

  nuri::SimulationTickResult tickResult =
      scheduler.tick(runtime, {.frameDeltaSeconds = 0.35,
                               .absoluteTimeSeconds = 0.35,
                               .frameIndex = 1u});
  EXPECT_EQ(tickResult.executedSteps, 2u);
  EXPECT_NEAR(tickResult.consumedSeconds, 0.2, 1.0e-9);
  EXPECT_NEAR(tickResult.remainingAccumulatorSeconds, 0.05, 1.0e-9);
  EXPECT_TRUE(tickResult.clamped);
  EXPECT_TRUE(tickResult.anySimulationRan);

  nuri::SimulationStats stats;
  ASSERT_TRUE(runtime.simulations().getStats(handle, stats));
  EXPECT_EQ(stats.executedStepCount, 2u);

  nuri::RenderScene scene2;
  nuri::SceneRuntimeHost runtime2;
  runtime2.bindScene(&scene2);
  createResult = runtime2.simulations().createSimulation(
      makeNodeSimulationDesc(scene2.graph().rootNode()));
  ASSERT_FALSE(createResult.hasError()) << createResult.error();

  nuri::SimulationScheduler scheduler2;
  scheduler2.setConfig({
      .fixedDeltaSeconds = 0.1,
      .maxStepsPerFrame = 10u,
      .maxAccumulatedSeconds = 0.25,
      .allowFrameDropping = false,
  });

  tickResult = scheduler2.tick(
      runtime2,
      {.frameDeltaSeconds = 1.0, .absoluteTimeSeconds = 1.0, .frameIndex = 1u});
  EXPECT_EQ(tickResult.executedSteps, 2u);
  EXPECT_NEAR(tickResult.remainingAccumulatorSeconds, 0.05, 1.0e-9);
  EXPECT_TRUE(tickResult.clamped);
}

TEST(SimulationRuntimeTests, InvalidBindingFaultsAfterSceneTopologyChanges) {
  nuri::RenderScene scene;
  nuri::SceneRuntimeHost runtime;
  runtime.bindScene(&scene);

  auto nodeResult = scene.graph().createNode(scene.graph().rootNode(), "Bound");
  ASSERT_FALSE(nodeResult.hasError()) << nodeResult.error();
  const nuri::NodeId boundNode = nodeResult.value();

  auto createResult =
      runtime.simulations().createSimulation(makeNodeSimulationDesc(boundNode));
  ASSERT_FALSE(createResult.hasError()) << createResult.error();
  const nuri::SimulationHandle handle = createResult.value();

  EXPECT_TRUE(scene.graph().destroyNodeSubtree(boundNode));
  (void)runtime.tick(
      {.frameDeltaSeconds = 0.0, .absoluteTimeSeconds = 0.0, .frameIndex = 1u});

  nuri::SimulationStats stats;
  ASSERT_TRUE(runtime.simulations().getStats(handle, stats));
  EXPECT_TRUE(stats.faulted);
  EXPECT_FALSE(stats.lastFaultReason.empty());

  nuri::SimulationState state = nuri::SimulationState::Running;
  ASSERT_TRUE(runtime.simulations().getState(handle, state));
  EXPECT_EQ(state, nuri::SimulationState::Stopped);
}

TEST(SimulationRuntimeTests, BindingVersionTracksTopologyButNotTransforms) {
  nuri::RenderScene scene;
  nuri::SceneRuntimeHost runtime;
  runtime.bindScene(&scene);
  const uint64_t initialBindingVersion = runtime.bindingVersion();

  auto nodeResult = scene.graph().createNode(scene.graph().rootNode(), "Bound");
  ASSERT_FALSE(nodeResult.hasError()) << nodeResult.error();
  const nuri::NodeId node = nodeResult.value();

  (void)runtime.tick(
      {.frameDeltaSeconds = 0.0, .absoluteTimeSeconds = 0.0, .frameIndex = 1u});
  const uint64_t topologyBindingVersion = runtime.bindingVersion();
  EXPECT_GT(topologyBindingVersion, initialBindingVersion);

  ASSERT_TRUE(scene.graph().setNodeLocalTransform(
      node, translate(glm::vec3(1.0f, 2.0f, 3.0f))));
  (void)runtime.tick(
      {.frameDeltaSeconds = 0.0, .absoluteTimeSeconds = 0.0, .frameIndex = 2u});
  EXPECT_EQ(runtime.bindingVersion(), topologyBindingVersion);
}

TEST(SimulationRuntimeTests, GpuContextOnlyVersionsOnMeaningfulChanges) {
  nuri::SimulationGpuContext context;
  EXPECT_EQ(context.frameIndex(), 0u);
  EXPECT_EQ(context.resourceVersion(), 0u);

  context.beginFrame(7u);
  EXPECT_EQ(context.frameIndex(), 7u);
  EXPECT_EQ(context.resourceVersion(), 0u);

  nuri::SimulationFrameGpuResources resources{};
  resources.paramUpload.buffer = {.index = 1u, .generation = 1u};
  resources.paramUpload.sizeBytes = 64u;
  resources.writeback.buffer = {.index = 2u, .generation = 1u};
  resources.writeback.offsetBytes = 32u;
  resources.writeback.sizeBytes = 128u;

  context.publishFrameResources(resources);
  EXPECT_EQ(context.resourceVersion(), 1u);
  EXPECT_EQ(context.frameResources().frameGeneration, 7u);
  EXPECT_EQ(context.frameResources().resourceVersion, 1u);

  context.publishFrameResources(resources);
  EXPECT_EQ(context.resourceVersion(), 1u);

  resources.dispatchMetadata.buffer = {.index = 3u, .generation = 1u};
  resources.dispatchMetadata.sizeBytes = 16u;
  context.publishFrameResources(resources);
  EXPECT_EQ(context.resourceVersion(), 2u);
  EXPECT_EQ(context.frameResources().frameGeneration, 7u);
  EXPECT_EQ(context.frameResources().resourceVersion, 2u);
}
