#include "tests_pch.h"

#include "render_graph_test_support.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <span>
#include <string>

namespace {

using namespace nuri;
using namespace nuri::test_support;

using FakeGPUDevice = FakeExecutorGPUDevice;

class RenderGraphExecutorTest : public ::testing::Test {};

Result<RenderGraphCompileResult, std::string>
compileBuilder(RenderGraphBuilder &builder) {
  RenderGraphRuntime runtime;
  return builder.compile(runtime);
}

Result<bool, std::string>
executeCompiled(RenderGraphExecutor &executor, GPUDevice &gpu,
                const RenderGraphCompileResult &compiled) {
  auto beginResult = gpu.beginFrame(compiled.frameIndex);
  if (beginResult.hasError()) {
    return Result<bool, std::string>::makeError(beginResult.error());
  }
  RenderGraphRuntime runtime;
  auto result = executor.execute(runtime, gpu, compiled);
  if (result.hasError()) {
    return Result<bool, std::string>::makeError(result.error());
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<RenderGraphExecutionMetadata, std::string>
executeCompiledWithConfig(RenderGraphExecutor &executor, GPUDevice &gpu,
                          const RenderGraphCompileResult &compiled,
                          const RenderGraphRuntimeConfig &config) {
  auto beginResult = gpu.beginFrame(compiled.frameIndex);
  if (beginResult.hasError()) {
    return Result<RenderGraphExecutionMetadata, std::string>::makeError(
        beginResult.error());
  }
  RenderGraphRuntime runtime(config);
  return executor.execute(runtime, gpu, compiled);
}

bool hasExecutionFailureStage(const std::string &error,
                              RenderGraphExecutionFailureStage stage) {
  const std::string expectedTag =
      "[stage=" + std::string(toString(stage)) + "]";
  return error.find(expectedTag) != std::string::npos;
}

Result<RenderGraphCompileResult, std::string>
buildExecutorCompiledFrame(uint64_t frameIndex) {
  RenderGraphBuilder builder;
  builder.beginFrame(frameIndex);

  auto colorTexResult = builder.createTransientTexture(
      makeTransientTextureDesc(Format::RGBA8_UNORM, 16u, 16u), "exec_color");
  auto depthTexResult = builder.createTransientTexture(
      makeTransientTextureDesc(Format::D32_FLOAT, 16u, 16u), "exec_depth");
  auto transientBufResult =
      builder.createTransientBuffer(makeTransientBufferDesc(64u), "exec_buf");
  if (colorTexResult.hasError()) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        "buildExecutorCompiledFrame: createTransientTexture(color) failed: " +
        colorTexResult.error());
  }
  if (depthTexResult.hasError()) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        "buildExecutorCompiledFrame: createTransientTexture(depth) failed: " +
        depthTexResult.error());
  }
  if (transientBufResult.hasError()) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        "buildExecutorCompiledFrame: createTransientBuffer failed: " +
        transientBufResult.error());
  }

  std::array<BufferHandle, 1> passDeps = {BufferHandle{}};
  std::array<BufferHandle, 1> dispatchDeps = {BufferHandle{}};
  ComputeDispatchItem dispatch{};
  dispatch.dependencyBuffers =
      std::span<const BufferHandle>(dispatchDeps.data(), dispatchDeps.size());
  std::array<ComputeDispatchItem, 1> preDispatches = {dispatch};
  std::array<DrawItem, 1> draws = {DrawItem{}};

  RenderPass pass{};
  pass.dependencyBuffers =
      std::span<const BufferHandle>(passDeps.data(), passDeps.size());
  pass.preDispatches = std::span<const ComputeDispatchItem>(
      preDispatches.data(), preDispatches.size());
  pass.draws = std::span<const DrawItem>(draws.data(), draws.size());
  pass.debugLabel = "exec_pass";

  auto addPassResult = addTestGraphicsPass(builder, pass, pass.debugLabel);
  if (addPassResult.hasError()) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        "buildExecutorCompiledFrame: addLegacyRenderPass failed: " +
        addPassResult.error());
  }

  auto bindResult = builder.bindPassColorTexture(addPassResult.value(),
                                                 colorTexResult.value());
  if (bindResult.hasError()) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        "buildExecutorCompiledFrame: bindPassColorTexture failed: " +
        bindResult.error());
  }
  bindResult = builder.bindPassDepthTexture(addPassResult.value(),
                                            depthTexResult.value());
  if (bindResult.hasError()) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        "buildExecutorCompiledFrame: bindPassDepthTexture failed: " +
        bindResult.error());
  }
  bindResult = builder.bindPassDependencyBuffer(addPassResult.value(), 0u,
                                                transientBufResult.value());
  if (bindResult.hasError()) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        "buildExecutorCompiledFrame: bindPassDependencyBuffer failed: " +
        bindResult.error());
  }
  bindResult = builder.bindPreDispatchDependencyBuffer(
      addPassResult.value(), 0u, 0u, transientBufResult.value());
  if (bindResult.hasError()) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        "buildExecutorCompiledFrame: bindPreDispatchDependencyBuffer failed: " +
        bindResult.error());
  }
  bindResult = builder.bindDrawBuffer(
      addPassResult.value(), 0u,
      RenderGraphCompileResult::DrawBufferBindingTarget::Vertex,
      transientBufResult.value(), RenderGraphAccessMode::Read);
  if (bindResult.hasError()) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        "buildExecutorCompiledFrame: bindDrawBuffer failed: " +
        bindResult.error());
  }

  return compileBuilder(builder);
}

Result<RenderGraphCompileResult, std::string>
buildEmptyCompiledFrame(uint64_t frameIndex) {
  RenderGraphBuilder builder;
  builder.beginFrame(frameIndex);
  return compileBuilder(builder);
}

Result<RenderGraphCompileResult, std::string>
buildTwoPassCompiledFrameWithDependency(uint64_t frameIndex) {
  RenderGraphBuilder builder;
  builder.beginFrame(frameIndex);

  RenderPass passA{};
  passA.debugLabel = "edge_pass_a";
  RenderPass passB{};
  passB.debugLabel = "edge_pass_b";

  auto passAResult = addTestGraphicsPass(builder, passA, passA.debugLabel);
  auto passBResult = addTestGraphicsPass(builder, passB, passB.debugLabel);
  if (passAResult.hasError() || passBResult.hasError()) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        "buildTwoPassCompiledFrameWithDependency: addLegacyRenderPass failed");
  }

  auto depResult =
      builder.addDependency(passAResult.value(), passBResult.value());
  if (depResult.hasError()) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        "buildTwoPassCompiledFrameWithDependency: addDependency failed: " +
        depResult.error());
  }

  return compileBuilder(builder);
}

Result<RenderGraphCompileResult, std::string>
buildIndependentParallelCompiledFrame(uint64_t frameIndex, uint32_t passCount) {
  RenderGraphBuilder builder;
  builder.beginFrame(frameIndex);

  for (uint32_t passIndex = 0u; passIndex < passCount; ++passIndex) {
    RenderPass pass{};
    const std::string label =
        "parallel_pass_" + std::to_string(static_cast<unsigned>(passIndex));
    pass.debugLabel = label;
    auto addResult = addTestGraphicsPass(builder, pass, pass.debugLabel);
    if (addResult.hasError()) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "buildIndependentParallelCompiledFrame: addLegacyRenderPass "
          "failed: " +
          addResult.error());
    }
    auto sideEffectResult = builder.markPassSideEffect(addResult.value());
    if (sideEffectResult.hasError()) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "buildIndependentParallelCompiledFrame: markPassSideEffect failed: " +
          sideEffectResult.error());
    }
  }

  return compileBuilder(builder);
}

Result<RenderGraphCompileResult, std::string>
buildBarrierTrackedCompiledFrame(uint64_t frameIndex) {
  RenderGraphBuilder builder;
  builder.beginFrame(frameIndex);

  auto bufferResult = builder.createTransientBuffer(
      makeTransientBufferDesc(64u), "barrier_buf");
  if (bufferResult.hasError()) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        "buildBarrierTrackedCompiledFrame: createTransientBuffer failed: " +
        bufferResult.error());
  }

  RenderPass passA{};
  passA.debugLabel = "barrier_pass_a";
  RenderPass passB{};
  passB.debugLabel = "barrier_pass_b";

  auto passAResult = addTestGraphicsPass(builder, passA, passA.debugLabel);
  auto passBResult = addTestGraphicsPass(builder, passB, passB.debugLabel);
  if (passAResult.hasError() || passBResult.hasError()) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        "buildBarrierTrackedCompiledFrame: addLegacyRenderPass failed");
  }

  auto accessResult =
      builder.addBufferWrite(passAResult.value(), bufferResult.value());
  if (accessResult.hasError()) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        "buildBarrierTrackedCompiledFrame: addBufferWrite failed: " +
        accessResult.error());
  }
  accessResult =
      builder.addBufferRead(passBResult.value(), bufferResult.value());
  if (accessResult.hasError()) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        "buildBarrierTrackedCompiledFrame: addBufferRead failed: " +
        accessResult.error());
  }
  auto sideEffectResult = builder.markPassSideEffect(passBResult.value());
  if (sideEffectResult.hasError()) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        "buildBarrierTrackedCompiledFrame: markPassSideEffect failed: " +
        sideEffectResult.error());
  }

  return compileBuilder(builder);
}

Result<RenderGraphCompileResult, std::string>
buildFrameOutputCompiledFrame(uint64_t frameIndex) {
  RenderGraphBuilder builder;
  builder.beginFrame(frameIndex);

  const TextureHandle outputTexture{.index = 701u, .generation = 1u};
  auto importResult = builder.importTexture(outputTexture, "frame_output");
  if (importResult.hasError()) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        "buildFrameOutputCompiledFrame: importTexture failed: " +
        importResult.error());
  }

  RenderPass pass{};
  pass.colorTexture = outputTexture;
  pass.debugLabel = "frame_output_pass";
  auto passResult = addTestGraphicsPass(builder, pass, pass.debugLabel);
  if (passResult.hasError()) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        "buildFrameOutputCompiledFrame: addLegacyRenderPass failed: " +
        passResult.error());
  }

  auto bindResult =
      builder.bindPassColorTexture(passResult.value(), importResult.value());
  if (bindResult.hasError()) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        "buildFrameOutputCompiledFrame: bindPassColorTexture failed: " +
        bindResult.error());
  }
  auto outputResult = builder.markTextureAsFrameOutput(importResult.value());
  if (outputResult.hasError()) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        "buildFrameOutputCompiledFrame: markTextureAsFrameOutput failed: " +
        outputResult.error());
  }

  return compileBuilder(builder);
}

TEST_F(RenderGraphExecutorTest,
       ExecutorMaterializesRewritesAndRetiresTransients) {
  auto compileResult = buildExecutorCompiledFrame(100u);
  ASSERT_FALSE(compileResult.hasError());
  const RenderGraphCompileResult &compiled = compileResult.value();
  ASSERT_EQ(compiled.unresolvedTextureBindings.size(), 2u)
      << "expected unresolved transient color/depth bindings";
  ASSERT_EQ(compiled.unresolvedDependencyBufferBindings.size(), 1u)
      << "expected unresolved transient buffer dependency bindings";
  ASSERT_EQ(compiled.unresolvedPreDispatchDependencyBufferBindings.size(), 1u)
      << "expected unresolved transient pre-dispatch bindings";
  ASSERT_EQ(compiled.unresolvedDrawBufferBindings.size(), 1u)
      << "expected unresolved transient draw bindings";

  FakeExecutorGPUDevice gpu;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, compiled);
  ASSERT_FALSE(executeResult.hasError());
  ASSERT_TRUE(executeResult.value())
      << "executor should succeed for transient rewrite graph";

  ASSERT_EQ(gpu.submitCount, 1u);
  ASSERT_EQ(gpu.lastSubmitPassCount, 1u) << "executor should submit one pass";
  ASSERT_TRUE(nuri::isValid(gpu.lastColorTexture) &&
              nuri::isValid(gpu.lastDepthTexture))
      << "submitted pass should have materialized transient textures";
  ASSERT_NE(sameTexture(gpu.lastColorTexture, gpu.lastDepthTexture), true)
      << "color/depth transient textures should be distinct in this graph";
  ASSERT_TRUE(nuri::isValid(gpu.lastDependencyBuffer) &&
              nuri::isValid(gpu.lastPreDispatchDependencyBuffer) &&
              nuri::isValid(gpu.lastDrawVertexBuffer))
      << "submitted pass should have materialized transient buffers";
  ASSERT_TRUE(sameBuffer(gpu.lastDependencyBuffer,
                         gpu.lastPreDispatchDependencyBuffer) &&
              sameBuffer(gpu.lastDependencyBuffer, gpu.lastDrawVertexBuffer))
      << "all rewritten buffer slots should resolve to same transient "
         "allocation";
  ASSERT_EQ(gpu.createdTextureCount, compiled.transientTexturePhysicalCount)
      << "materialized transient texture allocation counts should match "
         "compile metadata";
  ASSERT_EQ(gpu.createdBufferCount, compiled.transientBufferPhysicalCount)
      << "materialized transient buffer allocation counts should match "
         "compile metadata";
  ASSERT_EQ(gpu.destroyedTextureCount, 0u)
      << "newly materialized transient textures should not retire "
         "immediately";
  ASSERT_EQ(gpu.destroyedBufferCount, 0u)
      << "newly materialized transient buffers should not retire immediately";
  ASSERT_EQ(gpu.waitIdleCallCount, 0u)
      << "executor retirement should not block on waitIdle after submit";

  auto compile102 = buildEmptyCompiledFrame(102u);
  ASSERT_FALSE(compile102.hasError());
  executeResult = executeCompiled(executor, gpu, compile102.value());
  ASSERT_FALSE(executeResult.hasError());
  ASSERT_TRUE(executeResult.value())
      << "executor should succeed for empty frame 102";
  ASSERT_EQ(gpu.destroyedTextureCount, 0u)
      << "retirement should not occur before retire frame";
  ASSERT_EQ(gpu.destroyedBufferCount, 0u)
      << "retirement should not occur before retire frame";
  ASSERT_EQ(gpu.waitIdleCallCount, 0u)
      << "executor retirement polling should remain non-blocking";

  auto compile103 = buildEmptyCompiledFrame(103u);
  ASSERT_FALSE(compile103.hasError());
  executeResult = executeCompiled(executor, gpu, compile103.value());
  ASSERT_FALSE(executeResult.hasError());
  ASSERT_TRUE(executeResult.value())
      << "executor should succeed for empty frame 103";
  EXPECT_LE(gpu.destroyedTextureCount, gpu.createdTextureCount)
      << "retirement should not over-destroy transient textures";
  EXPECT_LE(gpu.destroyedBufferCount, gpu.createdBufferCount)
      << "retirement should not over-destroy transient buffers";
  EXPECT_EQ(gpu.waitIdleCallCount, 0u)
      << "executor retirement should not fall back to waitIdle";

  const uint32_t createdTextureCountBeforeReuse = gpu.createdTextureCount;
  const uint32_t createdBufferCountBeforeReuse = gpu.createdBufferCount;
  auto compile104 = buildExecutorCompiledFrame(104u);
  ASSERT_FALSE(compile104.hasError());
  executeResult = executeCompiled(executor, gpu, compile104.value());
  ASSERT_FALSE(executeResult.hasError());
  ASSERT_TRUE(executeResult.value())
      << "executor should succeed for frame 104 reuse check";
  EXPECT_EQ(gpu.createdTextureCount, createdTextureCountBeforeReuse)
      << "retired transient textures should be reused without re-creation";
  EXPECT_EQ(gpu.createdBufferCount, createdBufferCountBeforeReuse)
      << "retired transient buffers should be reused without re-creation";
}

TEST_F(RenderGraphExecutorTest,
       ExecutorCleansUpOnPartialMaterializationFailure) {
  auto compileResult = buildExecutorCompiledFrame(110u);
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed";
    return;
  }
  const RenderGraphCompileResult &compiled = compileResult.value();

  FakeExecutorGPUDevice gpu;
  gpu.failCreateBufferAtCall = 1u;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, compiled);
  if (!(executeResult.hasError())) {
    ADD_FAILURE() << "executor should fail when buffer materialization fails";
    return;
  }
  if (((executeResult.error()).find("failed to create transient buffer") ==
       std::string_view::npos)) {
    ADD_FAILURE() << "expected transient buffer creation failure message";
    return;
  }
  if (!hasExecutionFailureStage(
          executeResult.error(),
          RenderGraphExecutionFailureStage::MaterializeTransients)) {
    ADD_FAILURE() << "expected materialize-transients failure stage tag";
    return;
  }
  if (!(gpu.submitCount == 0u)) {
    ADD_FAILURE() << "submit should not be called on materialization failure";
    return;
  }
  if (!(gpu.createdTextureCount == compiled.transientTexturePhysicalCount)) {
    ADD_FAILURE() << "textures should be created before buffer failure";
    return;
  }
  if (!(gpu.createdBufferCount == 0u)) {
    ADD_FAILURE()
        << "no transient buffer should be created on first-call failure";
    return;
  }
  if (!(gpu.destroyedTextureCount == gpu.createdTextureCount &&
        gpu.destroyedBufferCount == 0u)) {
    ADD_FAILURE() << "materialized textures should be cleaned up on failure";
    return;
  }
}

TEST_F(RenderGraphExecutorTest,
       ExecutorSubmitFailureDefersThenRetiresResources) {
  auto compileResult = buildExecutorCompiledFrame(120u);
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed";
    return;
  }
  const RenderGraphCompileResult &compiled = compileResult.value();

  FakeExecutorGPUDevice gpu;
  gpu.failSubmitFrame = true;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, compiled);
  if (!(executeResult.hasError())) {
    ADD_FAILURE() << "executor should propagate submit failure";
    return;
  }
  if (((executeResult.error()).find("fake submitFrame failure") ==
       std::string_view::npos)) {
    ADD_FAILURE() << "submit failure reason should propagate";
    return;
  }
  if (!hasExecutionFailureStage(
          executeResult.error(),
          RenderGraphExecutionFailureStage::SubmitRecordedFrame)) {
    ADD_FAILURE() << "expected submit-recorded-frame failure stage tag";
    return;
  }
  if (!(gpu.submitCount == 1u)) {
    ADD_FAILURE() << "submit should be attempted once";
    return;
  }
  if (!(gpu.createdTextureCount == compiled.transientTexturePhysicalCount &&
        gpu.createdBufferCount == compiled.transientBufferPhysicalCount)) {
    ADD_FAILURE() << "resources should still be materialized before submit";
    return;
  }
  if (!(gpu.destroyedTextureCount == 0u && gpu.destroyedBufferCount == 0u)) {
    ADD_FAILURE() << "submit failure keeps resources pending for deferred "
                     "retirement";
    return;
  }

  gpu.failSubmitFrame = false;
  auto compile123 = buildEmptyCompiledFrame(123u);
  if (!(!compile123.hasError())) {
    ADD_FAILURE() << "empty frame 123 compile should succeed";
    return;
  }
  executeResult = executeCompiled(executor, gpu, compile123.value());
  if (!(!executeResult.hasError() && executeResult.value())) {
    ADD_FAILURE() << "executor should succeed for retirement frame";
    return;
  }
  if (!(gpu.destroyedTextureCount <= gpu.createdTextureCount &&
        gpu.destroyedBufferCount <= gpu.createdBufferCount)) {
    ADD_FAILURE() << "retirement should not over-destroy deferred resources";
    return;
  }

  const uint32_t createdTextureCountBeforeReuse = gpu.createdTextureCount;
  const uint32_t createdBufferCountBeforeReuse = gpu.createdBufferCount;
  auto compile124 = buildExecutorCompiledFrame(124u);
  if (!(!compile124.hasError())) {
    ADD_FAILURE() << "non-empty frame 124 compile should succeed";
    return;
  }
  executeResult = executeCompiled(executor, gpu, compile124.value());
  if (!(!executeResult.hasError() && executeResult.value())) {
    ADD_FAILURE() << "executor should succeed for frame 124 reuse check";
    return;
  }
  if (!(gpu.createdTextureCount == createdTextureCountBeforeReuse &&
        gpu.createdBufferCount == createdBufferCountBeforeReuse)) {
    ADD_FAILURE() << "retired deferred resources should be reused without "
                     "re-creation";
    return;
  }
}

TEST_F(RenderGraphExecutorTest, ExecutorResolvesAndRecordsPerPassBarriers) {
  auto compileResult = buildBarrierTrackedCompiledFrame(140u);
  ASSERT_FALSE(compileResult.hasError());
  const RenderGraphCompileResult &compiled = compileResult.value();
  ASSERT_EQ(compiled.passBarrierPlans.size(), 2u);
  ASSERT_EQ(compiled.passBarrierRecords.size(), 2u);

  FakeExecutorGPUDevice gpu;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, compiled);
  ASSERT_FALSE(executeResult.hasError());
  ASSERT_TRUE(executeResult.value());

  ASSERT_EQ(gpu.recordedBarrierBatchCounts.size(), 2u);
  EXPECT_EQ(gpu.recordedBarrierBatchCounts[0], 1u);
  EXPECT_EQ(gpu.recordedBarrierBatchCounts[1], 1u);
}

TEST_F(RenderGraphExecutorTest,
       ExecutorRecordsFinalPresentBarrierAfterLastPass) {
  auto compileResult = buildFrameOutputCompiledFrame(142u);
  ASSERT_FALSE(compileResult.hasError());
  const RenderGraphCompileResult &compiled = compileResult.value();
  ASSERT_EQ(compiled.passBarrierPlans.size(), 1u);
  ASSERT_EQ(compiled.finalBarrierPlan.barrierCount, 1u);

  FakeExecutorGPUDevice gpu;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, compiled);
  ASSERT_FALSE(executeResult.hasError());
  ASSERT_TRUE(executeResult.value());

  ASSERT_EQ(gpu.recordedBarrierBatchCounts.size(), 2u);
  EXPECT_EQ(gpu.recordedBarrierBatchCounts[0], 1u);
  EXPECT_EQ(gpu.recordedBarrierBatchCounts[1], 1u);
}

TEST_F(RenderGraphExecutorTest, ExecutorTagsBarrierResolutionFailuresByStage) {
  auto compileResult = buildBarrierTrackedCompiledFrame(141u);
  ASSERT_FALSE(compileResult.hasError());

  RenderGraphCompileResult invalid = compileResult.value();
  ASSERT_FALSE(invalid.passBarrierRecords.empty());
  invalid.passBarrierRecords[0].resourceKind =
      RenderGraphBarrierResourceKind::Buffer;
  invalid.passBarrierRecords[0].resourceIndex =
      static_cast<uint32_t>(invalid.bufferHandlesByResource.size());

  FakeExecutorGPUDevice gpu;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, invalid);
  ASSERT_TRUE(executeResult.hasError());
  EXPECT_TRUE(hasExecutionFailureStage(
      executeResult.error(),
      RenderGraphExecutionFailureStage::ResolveBarriers));
  EXPECT_NE(executeResult.error().find(
                "buffer barrier resource index is out of range"),
            std::string_view::npos);
  EXPECT_EQ(gpu.submitCount, 0u);
}

TEST_F(RenderGraphExecutorTest,
       ExecutorParallelAcquireFailureDiscardsAllContextsAndSubmitsNothing) {
  auto compileResult = buildIndependentParallelCompiledFrame(125u, 8u);
  ASSERT_FALSE(compileResult.hasError());

  FakeExecutorGPUDevice gpu;
  gpu.maxRecordingContexts = 4u;
  gpu.failAcquireWorkerIndex = 0;
  RenderGraphExecutor executor;
  const RenderGraphRuntimeConfig config{
      .workerCount = 4u,
      .parallelCompile = true,
      .parallelGraphicsRecording = true,
  };

  auto executeResult =
      executeCompiledWithConfig(executor, gpu, compileResult.value(), config);
  ASSERT_TRUE(executeResult.hasError());
  EXPECT_TRUE(hasExecutionFailureStage(
      executeResult.error(),
      RenderGraphExecutionFailureStage::AcquireRecordingContext));
  EXPECT_NE(executeResult.error().find(
                "failed to acquire graphics recording context"),
            std::string_view::npos);
  EXPECT_EQ(gpu.submitCount, 0u);
  EXPECT_EQ(gpu.finishedRecordingContextCount, 0u);
  EXPECT_EQ(gpu.discardedRecordedCommandBufferCount, 0u);
  EXPECT_EQ(gpu.discardedRecordingContextCount,
            gpu.acquiredRecordingContextCount);
}

TEST_F(RenderGraphExecutorTest,
       ExecutorParallelRecordFailureDiscardsAllContextsAndSubmitsNothing) {
  auto compileResult = buildIndependentParallelCompiledFrame(126u, 8u);
  ASSERT_FALSE(compileResult.hasError());

  FakeExecutorGPUDevice gpu;
  gpu.maxRecordingContexts = 2u;
  gpu.failRecordPassLabel = "parallel_pass_2";
  RenderGraphExecutor executor;
  const RenderGraphRuntimeConfig config{
      .workerCount = 2u,
      .parallelCompile = true,
      .parallelGraphicsRecording = true,
  };

  auto executeResult =
      executeCompiledWithConfig(executor, gpu, compileResult.value(), config);
  ASSERT_TRUE(executeResult.hasError());
  EXPECT_TRUE(hasExecutionFailureStage(
      executeResult.error(),
      RenderGraphExecutionFailureStage::RecordGraphicsPasses));
  EXPECT_NE(executeResult.error().find("failed to record graphics pass"),
            std::string_view::npos);
  EXPECT_EQ(gpu.submitCount, 0u);
  EXPECT_EQ(gpu.finishedRecordingContextCount, 0u);
  EXPECT_EQ(gpu.discardedRecordedCommandBufferCount, 0u);
  EXPECT_EQ(gpu.discardedRecordingContextCount,
            gpu.acquiredRecordingContextCount);
}

TEST_F(RenderGraphExecutorTest,
       ExecutorParallelFinishFailureDiscardsRecordedBuffersAndSubmitsNothing) {
  auto compileResult = buildIndependentParallelCompiledFrame(127u, 8u);
  ASSERT_FALSE(compileResult.hasError());

  FakeExecutorGPUDevice gpu;
  gpu.maxRecordingContexts = 2u;
  gpu.failFinishAtCall = 2u;
  RenderGraphExecutor executor;
  const RenderGraphRuntimeConfig config{
      .workerCount = 2u,
      .parallelCompile = true,
      .parallelGraphicsRecording = true,
  };

  auto executeResult =
      executeCompiledWithConfig(executor, gpu, compileResult.value(), config);
  ASSERT_TRUE(executeResult.hasError());
  EXPECT_TRUE(hasExecutionFailureStage(
      executeResult.error(),
      RenderGraphExecutionFailureStage::FinishRecordingContext));
  EXPECT_NE(
      executeResult.error().find("failed to finish graphics recording context"),
      std::string_view::npos);
  EXPECT_EQ(gpu.submitCount, 0u);
  EXPECT_EQ(gpu.acquiredRecordingContextCount, 2u);
  EXPECT_EQ(gpu.finishedRecordingContextCount, 1u);
  EXPECT_EQ(gpu.discardedRecordedCommandBufferCount, 1u);
  EXPECT_EQ(gpu.discardedRecordingContextCount, 1u);
}

TEST_F(RenderGraphExecutorTest,
       ExecutorRejectsInvalidAllocationMetadataBeforeCreate) {
  auto compileResult = buildExecutorCompiledFrame(130u);
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed";
    return;
  }

  RenderGraphCompileResult invalid = compileResult.value();
  invalid.transientTexturePhysicalCount += 1u;

  FakeExecutorGPUDevice gpu;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, invalid);
  if (!(executeResult.hasError())) {
    ADD_FAILURE() << "executor should reject invalid allocation metadata";
    return;
  }
  if (((executeResult.error()).find("allocation metadata count mismatch") ==
       std::string_view::npos)) {
    ADD_FAILURE() << "expected allocation metadata mismatch error";
    return;
  }
  if (!(gpu.createdTextureCount == 0u && gpu.createdBufferCount == 0u &&
        gpu.destroyedTextureCount == 0u && gpu.destroyedBufferCount == 0u &&
        gpu.submitCount == 0u)) {
    ADD_FAILURE()
        << "executor should fail before resource creation/submission on "
           "invalid metadata";
    return;
  }
}

TEST_F(RenderGraphExecutorTest,
       ExecutorRejectsOrderedPassIndexMetadataCountMismatch) {
  auto compileResult = buildExecutorCompiledFrame(132u);
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed";
    return;
  }

  RenderGraphCompileResult invalid = compileResult.value();
  if (!(!invalid.orderedPassIndices.empty())) {
    ADD_FAILURE() << "expected ordered pass indices";
    return;
  }
  invalid.orderedPassIndices.pop_back();

  FakeGPUDevice gpu;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, invalid);
  if (!(executeResult.hasError())) {
    ADD_FAILURE() << "executor should reject ordered pass index metadata count "
                     "mismatch";
    return;
  }
  if (((executeResult.error())
           .find("ordered pass index metadata count mismatch") ==
       std::string_view::npos)) {
    ADD_FAILURE() << "expected ordered pass index metadata mismatch error";
    return;
  }
  if (!(gpu.createdTextureCount == 0u && gpu.createdBufferCount == 0u &&
        gpu.destroyedTextureCount == 0u && gpu.destroyedBufferCount == 0u &&
        gpu.submitCount == 0u)) {
    ADD_FAILURE()
        << "executor should fail before resource creation/submission on "
           "ordered pass index metadata mismatch";
    return;
  }
}

TEST_F(RenderGraphExecutorTest,
       ExecutorRejectsDeclaredOrderedCulledPassCountInconsistency) {
  auto compileResult = buildExecutorCompiledFrame(133u);
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed";
    return;
  }

  RenderGraphCompileResult invalid = compileResult.value();
  invalid.culledPassCount += 1u;

  FakeGPUDevice gpu;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, invalid);
  if (!(executeResult.hasError())) {
    ADD_FAILURE()
        << "executor should reject declared/ordered/culled pass count "
           "inconsistency";
    return;
  }
  if (((executeResult.error())
           .find("declared/ordered/culled pass counts are inconsistent") ==
       std::string_view::npos)) {
    ADD_FAILURE() << "expected declared/ordered/culled pass count "
                     "inconsistency error";
    return;
  }
  if (!(gpu.createdTextureCount == 0u && gpu.createdBufferCount == 0u &&
        gpu.destroyedTextureCount == 0u && gpu.destroyedBufferCount == 0u &&
        gpu.submitCount == 0u)) {
    ADD_FAILURE()
        << "executor should fail before resource creation/submission on "
           "declared/ordered/culled pass count inconsistency";
    return;
  }
}

TEST_F(RenderGraphExecutorTest,
       ExecutorRejectsPassDebugNameMetadataCountMismatch) {
  auto compileResult = buildExecutorCompiledFrame(134u);
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed";
    return;
  }

  RenderGraphCompileResult invalid = compileResult.value();
  if (!(!invalid.passDebugNames.empty())) {
    ADD_FAILURE() << "expected pass debug names";
    return;
  }
  invalid.passDebugNames.pop_back();

  FakeGPUDevice gpu;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, invalid);
  if (!(executeResult.hasError())) {
    ADD_FAILURE() << "executor should reject pass debug-name metadata count "
                     "mismatch";
    return;
  }
  if (((executeResult.error())
           .find("pass debug-name metadata count mismatch") ==
       std::string_view::npos)) {
    ADD_FAILURE() << "expected pass debug-name metadata mismatch error";
    return;
  }
  if (!(gpu.createdTextureCount == 0u && gpu.createdBufferCount == 0u &&
        gpu.destroyedTextureCount == 0u && gpu.destroyedBufferCount == 0u &&
        gpu.submitCount == 0u)) {
    ADD_FAILURE()
        << "executor should fail before resource creation/submission on "
           "pass debug-name metadata mismatch";
    return;
  }
}

TEST_F(RenderGraphExecutorTest, ExecutorRejectsRootPassCountOverDeclared) {
  auto compileResult = buildExecutorCompiledFrame(1341u);
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed";
    return;
  }

  RenderGraphCompileResult invalid = compileResult.value();
  invalid.rootPassCount = invalid.declaredPassCount + 1u;

  FakeGPUDevice gpu;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, invalid);
  if (!(executeResult.hasError())) {
    ADD_FAILURE()
        << "executor should reject root pass count over declared pass count";
    return;
  }
  if (((executeResult.error())
           .find("root pass count exceeds declared pass count") ==
       std::string_view::npos)) {
    ADD_FAILURE() << "expected root pass count bounds error";
    return;
  }
  if (!(gpu.createdTextureCount == 0u && gpu.createdBufferCount == 0u &&
        gpu.destroyedTextureCount == 0u && gpu.destroyedBufferCount == 0u &&
        gpu.submitCount == 0u)) {
    ADD_FAILURE()
        << "executor should fail before resource creation/submission on "
           "root pass count bounds error";
    return;
  }
}

TEST_F(RenderGraphExecutorTest,
       ExecutorRejectsDependencyEdgePassIndexOutOfRange) {
  auto compileResult = buildTwoPassCompiledFrameWithDependency(1342u);
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed";
    return;
  }

  RenderGraphCompileResult invalid = compileResult.value();
  if (!(!invalid.edges.empty())) {
    ADD_FAILURE() << "expected dependency edges";
    return;
  }
  invalid.edges[0u].after = invalid.declaredPassCount;

  FakeGPUDevice gpu;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, invalid);
  if (!(executeResult.hasError())) {
    ADD_FAILURE() << "executor should reject dependency edge pass index out of "
                     "range";
    return;
  }
  if (((executeResult.error())
           .find("dependency edge pass index is out of range") ==
       std::string_view::npos)) {
    ADD_FAILURE() << "expected dependency edge pass-index bounds error";
    return;
  }
  if (!(gpu.createdTextureCount == 0u && gpu.createdBufferCount == 0u &&
        gpu.destroyedTextureCount == 0u && gpu.destroyedBufferCount == 0u &&
        gpu.submitCount == 0u)) {
    ADD_FAILURE()
        << "executor should fail before resource creation/submission on "
           "dependency edge pass-index bounds error";
    return;
  }
}

TEST_F(RenderGraphExecutorTest, ExecutorRejectsDependencyEdgeSelfCycle) {
  auto compileResult = buildTwoPassCompiledFrameWithDependency(1343u);
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed";
    return;
  }

  RenderGraphCompileResult invalid = compileResult.value();
  if (!(!invalid.edges.empty())) {
    ADD_FAILURE() << "expected dependency edges";
    return;
  }
  invalid.edges[0u].after = invalid.edges[0u].before;

  FakeGPUDevice gpu;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, invalid);
  if (!(executeResult.hasError())) {
    ADD_FAILURE() << "executor should reject dependency edge self-cycle";
    return;
  }
  if (((executeResult.error()).find("dependency edge self-cycle is invalid") ==
       std::string_view::npos)) {
    ADD_FAILURE() << "expected dependency edge self-cycle error";
    return;
  }
  if (!(gpu.createdTextureCount == 0u && gpu.createdBufferCount == 0u &&
        gpu.destroyedTextureCount == 0u && gpu.destroyedBufferCount == 0u &&
        gpu.submitCount == 0u)) {
    ADD_FAILURE()
        << "executor should fail before resource creation/submission on "
           "dependency edge self-cycle";
    return;
  }
}

TEST_F(RenderGraphExecutorTest,
       ExecutorRejectsDependencyEdgeReferencingCulledPass) {
  auto compileResult = buildTwoPassCompiledFrameWithDependency(1344u);
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed";
    return;
  }

  RenderGraphCompileResult invalid = compileResult.value();
  invalid.declaredPassCount = 3u;
  invalid.culledPassCount = 1u;
  invalid.passDebugNames.push_back("edge_pass_culled");
  if (!(!invalid.edges.empty())) {
    ADD_FAILURE() << "expected dependency edges";
    return;
  }
  invalid.edges[0u].after = 2u;

  FakeGPUDevice gpu;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, invalid);
  if (!(executeResult.hasError())) {
    ADD_FAILURE()
        << "executor should reject dependency edge referencing culled "
           "pass";
    return;
  }
  if (((executeResult.error()).find("dependency edge references culled pass") ==
       std::string_view::npos)) {
    ADD_FAILURE() << "expected dependency edge references culled pass error";
    return;
  }
  if (!(gpu.createdTextureCount == 0u && gpu.createdBufferCount == 0u &&
        gpu.destroyedTextureCount == 0u && gpu.destroyedBufferCount == 0u &&
        gpu.submitCount == 0u)) {
    ADD_FAILURE()
        << "executor should fail before resource creation/submission on "
           "dependency edge culled-pass error";
    return;
  }
}

TEST_F(RenderGraphExecutorTest, ExecutorRejectsDuplicatedDependencyEdge) {
  auto compileResult = buildTwoPassCompiledFrameWithDependency(1345u);
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed";
    return;
  }

  RenderGraphCompileResult invalid = compileResult.value();
  if (!(!invalid.edges.empty())) {
    ADD_FAILURE() << "expected dependency edges";
    return;
  }
  invalid.edges.push_back(invalid.edges[0u]);

  FakeGPUDevice gpu;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, invalid);
  if (!(executeResult.hasError())) {
    ADD_FAILURE() << "executor should reject duplicated dependency edge";
    return;
  }
  if (((executeResult.error()).find("dependency edge is duplicated") ==
       std::string_view::npos)) {
    ADD_FAILURE() << "expected duplicated dependency edge error";
    return;
  }
  if (!(gpu.createdTextureCount == 0u && gpu.createdBufferCount == 0u &&
        gpu.destroyedTextureCount == 0u && gpu.destroyedBufferCount == 0u &&
        gpu.submitCount == 0u)) {
    ADD_FAILURE()
        << "executor should fail before resource creation/submission on "
           "duplicated dependency edge";
    return;
  }
}

TEST_F(RenderGraphExecutorTest,
       ExecutorRejectsDependencyEdgeTopologyViolation) {
  auto compileResult = buildTwoPassCompiledFrameWithDependency(1346u);
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed";
    return;
  }

  RenderGraphCompileResult invalid = compileResult.value();
  if (!(!invalid.edges.empty())) {
    ADD_FAILURE() << "expected dependency edges";
    return;
  }
  invalid.edges[0u].before = invalid.orderedPassIndices[1u];
  invalid.edges[0u].after = invalid.orderedPassIndices[0u];

  FakeGPUDevice gpu;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, invalid);
  if (!(executeResult.hasError())) {
    ADD_FAILURE() << "executor should reject dependency edge topology "
                     "violation";
    return;
  }
  if (((executeResult.error())
           .find("dependency edge violates ordered pass topology") ==
       std::string_view::npos)) {
    ADD_FAILURE() << "expected dependency edge topology violation error";
    return;
  }
  if (!(gpu.createdTextureCount == 0u && gpu.createdBufferCount == 0u &&
        gpu.destroyedTextureCount == 0u && gpu.destroyedBufferCount == 0u &&
        gpu.submitCount == 0u)) {
    ADD_FAILURE()
        << "executor should fail before resource creation/submission on "
           "dependency edge topology violation";
    return;
  }
}

TEST_F(RenderGraphExecutorTest,
       ExecutorRejectsOutOfRangeTransientTextureAllocationIndex) {
  auto compileResult = buildExecutorCompiledFrame(135u);
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed";
    return;
  }

  RenderGraphCompileResult invalid = compileResult.value();
  if (!(!invalid.transientTexturePhysicalAllocations.empty())) {
    ADD_FAILURE() << "expected transient texture allocations";
    return;
  }
  invalid.transientTexturePhysicalAllocations[0u].allocationIndex = UINT32_MAX;

  FakeGPUDevice gpu;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, invalid);
  if (!(executeResult.hasError())) {
    ADD_FAILURE() << "executor should reject out-of-range transient texture "
                     "allocation index";
    return;
  }
  if (((executeResult.error())
           .find("transient texture allocation index is out of range") ==
       std::string_view::npos)) {
    ADD_FAILURE() << "expected transient texture allocation index bounds "
                     "error";
    return;
  }
  if (!(gpu.submitCount == 0u)) {
    ADD_FAILURE() << "submit should not run on transient texture allocation "
                     "index failure";
    return;
  }
  if (!(gpu.createdTextureCount == 0u && gpu.createdBufferCount == 0u)) {
    ADD_FAILURE() << "executor should fail before creating transient resources "
                     "on texture allocation index failure";
    return;
  }
}

TEST_F(RenderGraphExecutorTest,
       ExecutorRejectsOutOfRangeTransientBufferAllocationIndex) {
  auto compileResult = buildExecutorCompiledFrame(136u);
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed";
    return;
  }

  RenderGraphCompileResult invalid = compileResult.value();
  if (!(!invalid.transientBufferPhysicalAllocations.empty())) {
    ADD_FAILURE() << "expected transient buffer allocations";
    return;
  }
  invalid.transientBufferPhysicalAllocations[0u].allocationIndex = UINT32_MAX;

  FakeGPUDevice gpu;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, invalid);
  if (!(executeResult.hasError())) {
    ADD_FAILURE() << "executor should reject out-of-range transient buffer "
                     "allocation index";
    return;
  }
  if (((executeResult.error())
           .find("transient buffer allocation index is out of range") ==
       std::string_view::npos)) {
    ADD_FAILURE() << "expected transient buffer allocation index bounds "
                     "error";
    return;
  }
  if (!(gpu.submitCount == 0u)) {
    ADD_FAILURE() << "submit should not run on transient buffer allocation "
                     "index failure";
    return;
  }
  if (!(gpu.createdTextureCount == invalid.transientTexturePhysicalCount &&
        gpu.createdBufferCount == 0u)) {
    ADD_FAILURE() << "executor should only create textures before failing on "
                     "buffer allocation index";
    return;
  }
  if (!(gpu.destroyedTextureCount == gpu.createdTextureCount &&
        gpu.destroyedBufferCount == 0u)) {
    ADD_FAILURE() << "executor should clean up created textures on buffer "
                     "allocation index failure";
    return;
  }
}

TEST_F(RenderGraphExecutorTest, ExecutorRejectsOrderedPassIndexOutOfRange) {
  auto compileResult = buildExecutorCompiledFrame(137u);
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed";
    return;
  }

  RenderGraphCompileResult invalid = compileResult.value();
  if (!(!invalid.orderedPassIndices.empty())) {
    ADD_FAILURE() << "expected ordered pass indices";
    return;
  }
  invalid.orderedPassIndices[0u] = invalid.declaredPassCount;

  FakeGPUDevice gpu;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, invalid);
  if (!(executeResult.hasError())) {
    ADD_FAILURE() << "executor should reject ordered pass index out of range";
    return;
  }
  if (((executeResult.error()).find("ordered pass index is out of range") ==
       std::string_view::npos)) {
    ADD_FAILURE() << "expected ordered pass index bounds error";
    return;
  }
  if (!(gpu.createdTextureCount == 0u && gpu.createdBufferCount == 0u &&
        gpu.destroyedTextureCount == 0u && gpu.destroyedBufferCount == 0u &&
        gpu.submitCount == 0u)) {
    ADD_FAILURE()
        << "executor should fail before resource creation/submission on "
           "ordered pass index bounds error";
    return;
  }
}

TEST_F(RenderGraphExecutorTest, ExecutorRejectsDuplicatedOrderedPassIndex) {
  auto compileResult = buildExecutorCompiledFrame(138u);
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed";
    return;
  }

  RenderGraphCompileResult invalid = compileResult.value();
  if (!(!invalid.orderedPasses.empty() &&
        !invalid.orderedPassIndices.empty())) {
    ADD_FAILURE() << "expected ordered pass metadata";
    return;
  }
  invalid.declaredPassCount = 2u;
  invalid.culledPassCount = 0u;
  invalid.orderedPasses.push_back(invalid.orderedPasses[0u]);
  invalid.orderedPassIndices.push_back(invalid.orderedPassIndices[0u]);
  invalid.passDebugNames.push_back("exec_pass_dup");

  FakeGPUDevice gpu;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, invalid);
  if (!(executeResult.hasError())) {
    ADD_FAILURE() << "executor should reject duplicated ordered pass index";
    return;
  }
  if (((executeResult.error()).find("ordered pass index is duplicated") ==
       std::string_view::npos)) {
    ADD_FAILURE() << "expected duplicated ordered pass index error";
    return;
  }
  if (!(gpu.createdTextureCount == 0u && gpu.createdBufferCount == 0u &&
        gpu.destroyedTextureCount == 0u && gpu.destroyedBufferCount == 0u &&
        gpu.submitCount == 0u)) {
    ADD_FAILURE()
        << "executor should fail before resource creation/submission on "
           "duplicated ordered pass index";
    return;
  }
}

TEST_F(RenderGraphExecutorTest,
       ExecutorRejectsDuplicatedTransientTextureAllocationIndex) {
  auto compileResult = buildExecutorCompiledFrame(137u);
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed";
    return;
  }

  RenderGraphCompileResult invalid = compileResult.value();
  if (!(!invalid.transientTexturePhysicalAllocations.empty())) {
    ADD_FAILURE() << "expected transient texture allocations";
    return;
  }
  if (invalid.transientTexturePhysicalAllocations.size() == 1u) {
    invalid.transientTexturePhysicalAllocations.push_back(
        invalid.transientTexturePhysicalAllocations.front());
    invalid.transientTexturePhysicalCount += 1u;
  }
  invalid.transientTexturePhysicalAllocations[1u].allocationIndex =
      invalid.transientTexturePhysicalAllocations[0u].allocationIndex;

  FakeGPUDevice gpu;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, invalid);
  if (!(executeResult.hasError())) {
    ADD_FAILURE() << "executor should reject duplicated transient texture "
                     "allocation index";
    return;
  }
  if (((executeResult.error())
           .find("transient texture allocation index is duplicated") ==
       std::string_view::npos)) {
    ADD_FAILURE() << "expected duplicated transient texture allocation index "
                     "error";
    return;
  }
  if (!(gpu.submitCount == 0u)) {
    ADD_FAILURE() << "submit should not run on duplicated transient texture "
                     "allocation index failure";
    return;
  }
  if (!(gpu.createdTextureCount > 0u && gpu.createdBufferCount == 0u)) {
    ADD_FAILURE()
        << "executor should fail during texture materialization before "
           "buffer creation";
    return;
  }
  if (!(gpu.destroyedTextureCount == gpu.createdTextureCount &&
        gpu.destroyedBufferCount == 0u)) {
    ADD_FAILURE() << "executor should clean up created textures on duplicated "
                     "texture allocation index failure";
    return;
  }
}

TEST_F(RenderGraphExecutorTest,
       ExecutorRejectsDuplicatedTransientBufferAllocationIndex) {
  auto compileResult = buildExecutorCompiledFrame(138u);
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed";
    return;
  }

  RenderGraphCompileResult invalid = compileResult.value();
  if (!(!invalid.transientBufferPhysicalAllocations.empty())) {
    ADD_FAILURE() << "expected transient buffer allocations";
    return;
  }
  if (invalid.transientBufferPhysicalAllocations.size() == 1u) {
    invalid.transientBufferPhysicalAllocations.push_back(
        invalid.transientBufferPhysicalAllocations.front());
    invalid.transientBufferPhysicalCount += 1u;
  }
  invalid.transientBufferPhysicalAllocations[1u].allocationIndex =
      invalid.transientBufferPhysicalAllocations[0u].allocationIndex;

  FakeGPUDevice gpu;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, invalid);
  if (!(executeResult.hasError())) {
    ADD_FAILURE() << "executor should reject duplicated transient buffer "
                     "allocation index";
    return;
  }
  if (((executeResult.error())
           .find("transient buffer allocation index is duplicated") ==
       std::string_view::npos)) {
    ADD_FAILURE() << "expected duplicated transient buffer allocation index "
                     "error";
    return;
  }
  if (!(gpu.submitCount == 0u)) {
    ADD_FAILURE() << "submit should not run on duplicated transient buffer "
                     "allocation index failure";
    return;
  }
  if (!(gpu.createdTextureCount == invalid.transientTexturePhysicalCount &&
        gpu.createdBufferCount > 0u)) {
    ADD_FAILURE()
        << "executor should fully materialize textures and fail during "
           "buffer materialization";
    return;
  }
  if (!(gpu.destroyedTextureCount == gpu.createdTextureCount &&
        gpu.destroyedBufferCount == gpu.createdBufferCount)) {
    ADD_FAILURE() << "executor should clean up created resources on duplicated "
                     "buffer allocation index failure";
    return;
  }
}

TEST_F(RenderGraphExecutorTest,
       ExecutorRejectsOutOfRangeUnresolvedTextureBindingResource) {
  auto compileResult = buildExecutorCompiledFrame(140u);
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed";
    return;
  }
  RenderGraphCompileResult invalid = compileResult.value();
  if (!(!invalid.unresolvedTextureBindings.empty())) {
    ADD_FAILURE() << "expected unresolved texture bindings";
    return;
  }
  invalid.unresolvedTextureBindings[0u].textureResourceIndex = UINT32_MAX;

  FakeGPUDevice gpu;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, invalid);
  if (!(executeResult.hasError())) {
    ADD_FAILURE() << "executor should reject out-of-range unresolved texture "
                     "binding resource index";
    return;
  }
  if (((executeResult.error())
           .find("unresolved texture binding resource index is out of "
                 "range") == std::string_view::npos)) {
    ADD_FAILURE() << "expected unresolved texture binding range error";
    return;
  }
  if (!(gpu.submitCount == 0u)) {
    ADD_FAILURE() << "submit should not run on unresolved texture binding "
                     "metadata failure";
    return;
  }
  if (!(gpu.createdTextureCount > 0u)) {
    ADD_FAILURE()
        << "transient resources should be materialized before binding "
           "validation";
    return;
  }
  if (!(gpu.destroyedTextureCount == gpu.createdTextureCount &&
        gpu.destroyedBufferCount == gpu.createdBufferCount)) {
    ADD_FAILURE()
        << "materialized resources should be cleaned up on unresolved "
           "texture binding failure";
    return;
  }
}

TEST_F(RenderGraphExecutorTest,
       ExecutorRejectsOutOfRangeUnresolvedDrawBindingIndex) {
  auto compileResult = buildExecutorCompiledFrame(150u);
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed";
    return;
  }
  RenderGraphCompileResult invalid = compileResult.value();
  if (!(!invalid.unresolvedDrawBufferBindings.empty())) {
    ADD_FAILURE() << "expected unresolved draw buffer bindings";
    return;
  }
  invalid.unresolvedDrawBufferBindings[0u].drawIndex = UINT32_MAX;

  FakeGPUDevice gpu;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, invalid);
  if (!(executeResult.hasError())) {
    ADD_FAILURE() << "executor should reject out-of-range unresolved draw "
                     "binding index";
    return;
  }
  if (((executeResult.error())
           .find("unresolved draw buffer binding draw index is out of "
                 "range") == std::string_view::npos)) {
    ADD_FAILURE() << "expected unresolved draw index range error";
    return;
  }
  if (!(gpu.submitCount == 0u)) {
    ADD_FAILURE() << "submit should not run on unresolved draw binding "
                     "metadata failure";
    return;
  }
  if (!(gpu.createdTextureCount > 0u || gpu.createdBufferCount > 0u)) {
    ADD_FAILURE() << "transient resources should be materialized before draw "
                     "binding validation";
    return;
  }
  if (!(gpu.destroyedTextureCount == gpu.createdTextureCount &&
        gpu.destroyedBufferCount == gpu.createdBufferCount)) {
    ADD_FAILURE()
        << "materialized resources should be cleaned up on unresolved "
           "draw binding failure";
    return;
  }
}

TEST_F(RenderGraphExecutorTest,
       ExecutorRejectsPreDispatchDependencyRangeCountMismatch) {
  auto compileResult = buildExecutorCompiledFrame(160u);
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed";
    return;
  }
  RenderGraphCompileResult invalid = compileResult.value();
  invalid.preDispatchDependencyRanges.clear();

  FakeGPUDevice gpu;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, invalid);
  if (!(executeResult.hasError())) {
    ADD_FAILURE() << "executor should reject pre-dispatch dependency range "
                     "metadata count mismatch";
    return;
  }
  if (((executeResult.error())
           .find("pre-dispatch dependency range metadata count mismatch") ==
       std::string_view::npos)) {
    ADD_FAILURE()
        << "expected pre-dispatch dependency range metadata mismatch error";
    return;
  }
  if (!(gpu.submitCount == 0u)) {
    ADD_FAILURE() << "submit should not run on pre-dispatch range metadata "
                     "failure";
    return;
  }
  if (!(gpu.createdTextureCount > 0u || gpu.createdBufferCount > 0u)) {
    ADD_FAILURE() << "transient resources should be materialized before "
                     "pre-dispatch range validation";
    return;
  }
  if (!(gpu.destroyedTextureCount == gpu.createdTextureCount &&
        gpu.destroyedBufferCount == gpu.createdBufferCount)) {
    ADD_FAILURE() << "materialized resources should be cleaned up on "
                     "pre-dispatch range metadata failure";
    return;
  }
}

TEST_F(RenderGraphExecutorTest, ExecutorRejectsPassDrawRangeOutOfBounds) {
  auto compileResult = buildExecutorCompiledFrame(170u);
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed";
    return;
  }
  RenderGraphCompileResult invalid = compileResult.value();
  if (!(!invalid.drawRangesByPass.empty())) {
    ADD_FAILURE() << "expected draw ranges in compiled graph";
    return;
  }
  invalid.drawRangesByPass[0u].count = UINT32_MAX;

  FakeGPUDevice gpu;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, invalid);
  if (!(executeResult.hasError())) {
    ADD_FAILURE() << "executor should reject out-of-bounds pass draw range";
    return;
  }
  if (((executeResult.error()).find("pass draw range is out of bounds") ==
       std::string_view::npos)) {
    ADD_FAILURE() << "expected pass draw range bounds error";
    return;
  }
  if (!(gpu.submitCount == 0u)) {
    ADD_FAILURE() << "submit should not run on pass draw range failure";
    return;
  }
  if (!(gpu.destroyedTextureCount == gpu.createdTextureCount &&
        gpu.destroyedBufferCount == gpu.createdBufferCount)) {
    ADD_FAILURE() << "materialized resources should be cleaned up on pass draw "
                     "range failure";
    return;
  }
}

TEST_F(RenderGraphExecutorTest,
       ExecutorRejectsUnresolvedPreDispatchBindingIndexOutOfRange) {
  auto compileResult = buildExecutorCompiledFrame(180u);
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed";
    return;
  }
  RenderGraphCompileResult invalid = compileResult.value();
  if (!(!invalid.unresolvedPreDispatchDependencyBufferBindings.empty())) {
    ADD_FAILURE() << "expected unresolved pre-dispatch bindings";
    return;
  }
  invalid.unresolvedPreDispatchDependencyBufferBindings[0u].preDispatchIndex =
      UINT32_MAX;

  FakeGPUDevice gpu;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, invalid);
  if (!(executeResult.hasError())) {
    ADD_FAILURE() << "executor should reject unresolved pre-dispatch binding "
                     "index out of range";
    return;
  }
  if (((executeResult.error())
           .find("unresolved pre-dispatch dependency binding dispatch index is "
                 "out of "
                 "range") == std::string_view::npos)) {
    ADD_FAILURE()
        << "expected unresolved pre-dispatch dispatch-index bounds error";
    return;
  }
  if (!(gpu.submitCount == 0u)) {
    ADD_FAILURE() << "submit should not run on unresolved pre-dispatch binding "
                     "index failure";
    return;
  }
  if (!(gpu.destroyedTextureCount == gpu.createdTextureCount &&
        gpu.destroyedBufferCount == gpu.createdBufferCount)) {
    ADD_FAILURE()
        << "materialized resources should be cleaned up on unresolved "
           "pre-dispatch binding failure";
    return;
  }
}

TEST_F(RenderGraphExecutorTest,
       ExecutorRejectsUnresolvedDependencyBindingSlotOutOfRange) {
  auto compileResult = buildExecutorCompiledFrame(190u);
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed";
    return;
  }
  RenderGraphCompileResult invalid = compileResult.value();
  if (!(!invalid.unresolvedDependencyBufferBindings.empty())) {
    ADD_FAILURE() << "expected unresolved dependency bindings";
    return;
  }
  invalid.unresolvedDependencyBufferBindings[0u].dependencyBufferIndex =
      UINT32_MAX;

  FakeGPUDevice gpu;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, invalid);
  if (!(executeResult.hasError())) {
    ADD_FAILURE()
        << "executor should reject unresolved dependency binding slot "
           "index out of range";
    return;
  }
  if (((executeResult.error())
           .find("unresolved dependency buffer binding slot index is out of "
                 "range") == std::string_view::npos)) {
    ADD_FAILURE() << "expected unresolved dependency slot-index bounds error";
    return;
  }
  if (!(gpu.submitCount == 0u)) {
    ADD_FAILURE() << "submit should not run on unresolved dependency binding "
                     "slot failure";
    return;
  }
  if (!(gpu.destroyedTextureCount == gpu.createdTextureCount &&
        gpu.destroyedBufferCount == gpu.createdBufferCount)) {
    ADD_FAILURE()
        << "materialized resources should be cleaned up on unresolved "
           "dependency binding failure";
    return;
  }
}

TEST_F(RenderGraphExecutorTest,
       ExecutorRejectsUnresolvedDependencyBindingResourceOutOfRange) {
  auto compileResult = buildExecutorCompiledFrame(200u);
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed";
    return;
  }
  RenderGraphCompileResult invalid = compileResult.value();
  if (!(!invalid.unresolvedDependencyBufferBindings.empty())) {
    ADD_FAILURE() << "expected unresolved dependency bindings";
    return;
  }
  invalid.unresolvedDependencyBufferBindings[0u].bufferResourceIndex =
      UINT32_MAX;

  FakeGPUDevice gpu;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, invalid);
  if (!(executeResult.hasError())) {
    ADD_FAILURE() << "executor should reject unresolved dependency binding "
                     "resource index out of range";
    return;
  }
  if (((executeResult.error())
           .find("unresolved dependency buffer binding resource index is out "
                 "of range") == std::string_view::npos)) {
    ADD_FAILURE()
        << "expected unresolved dependency resource-index bounds error";
    return;
  }
  if (!(gpu.submitCount == 0u)) {
    ADD_FAILURE() << "submit should not run on unresolved dependency binding "
                     "resource failure";
    return;
  }
  if (!(gpu.destroyedTextureCount == gpu.createdTextureCount &&
        gpu.destroyedBufferCount == gpu.createdBufferCount)) {
    ADD_FAILURE()
        << "materialized resources should be cleaned up on unresolved "
           "dependency binding resource failure";
    return;
  }
}

TEST_F(RenderGraphExecutorTest,
       ExecutorRejectsPreDispatchDependencyRangeOutOfBounds) {
  auto compileResult = buildExecutorCompiledFrame(210u);
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed";
    return;
  }
  RenderGraphCompileResult invalid = compileResult.value();
  if (!(!invalid.preDispatchDependencyRanges.empty())) {
    ADD_FAILURE() << "expected pre-dispatch dependency ranges";
    return;
  }
  invalid.preDispatchDependencyRanges[0u].offset = static_cast<uint32_t>(
      invalid.resolvedPreDispatchDependencyBuffers.size());
  invalid.preDispatchDependencyRanges[0u].count = 1u;

  FakeGPUDevice gpu;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, invalid);
  if (!(executeResult.hasError())) {
    ADD_FAILURE() << "executor should reject out-of-bounds pre-dispatch "
                     "dependency range";
    return;
  }
  if (((executeResult.error())
           .find("pre-dispatch dependency range is out of bounds") ==
       std::string_view::npos)) {
    ADD_FAILURE() << "expected pre-dispatch dependency range bounds error";
    return;
  }
  if (!(gpu.submitCount == 0u)) {
    ADD_FAILURE() << "submit should not run on pre-dispatch dependency range "
                     "failure";
    return;
  }
  if (!(gpu.destroyedTextureCount == gpu.createdTextureCount &&
        gpu.destroyedBufferCount == gpu.createdBufferCount)) {
    ADD_FAILURE() << "materialized resources should be cleaned up on "
                     "pre-dispatch dependency range failure";
    return;
  }
}

TEST_F(RenderGraphExecutorTest,
       ExecutorRejectsUnresolvedDrawBindingInvalidTarget) {
  auto compileResult = buildExecutorCompiledFrame(220u);
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed";
    return;
  }
  RenderGraphCompileResult invalid = compileResult.value();
  if (!(!invalid.unresolvedDrawBufferBindings.empty())) {
    ADD_FAILURE() << "expected unresolved draw buffer bindings";
    return;
  }
  invalid.unresolvedDrawBufferBindings[0u].target =
      static_cast<RenderGraphCompileResult::DrawBufferBindingTarget>(255u);

  FakeGPUDevice gpu;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, invalid);
  if (!(executeResult.hasError())) {
    ADD_FAILURE() << "executor should reject unresolved draw binding invalid "
                     "target";
    return;
  }
  if (((executeResult.error())
           .find("unresolved draw buffer binding target is invalid") ==
       std::string_view::npos)) {
    ADD_FAILURE() << "expected unresolved draw target validation error";
    return;
  }
  if (!(gpu.submitCount == 0u)) {
    ADD_FAILURE() << "submit should not run on unresolved draw binding target "
                     "failure";
    return;
  }
  if (!(gpu.destroyedTextureCount == gpu.createdTextureCount &&
        gpu.destroyedBufferCount == gpu.createdBufferCount)) {
    ADD_FAILURE()
        << "materialized resources should be cleaned up on unresolved "
           "draw binding target failure";
    return;
  }
}

TEST_F(RenderGraphExecutorTest,
       ExecutorRejectsUnresolvedTextureBindingInvalidTarget) {
  auto compileResult = buildExecutorCompiledFrame(225u);
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed";
    return;
  }
  RenderGraphCompileResult invalid = compileResult.value();
  if (!(!invalid.unresolvedTextureBindings.empty())) {
    ADD_FAILURE() << "expected unresolved texture bindings";
    return;
  }
  invalid.unresolvedTextureBindings[0u].target =
      static_cast<RenderGraphCompileResult::PassTextureBindingTarget>(255u);

  FakeGPUDevice gpu;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, invalid);
  if (!(executeResult.hasError())) {
    ADD_FAILURE()
        << "executor should reject unresolved texture binding invalid "
           "target";
    return;
  }
  if (((executeResult.error())
           .find("unresolved texture binding target is invalid") ==
       std::string_view::npos)) {
    ADD_FAILURE() << "expected unresolved texture target validation error";
    return;
  }
  if (!(gpu.submitCount == 0u)) {
    ADD_FAILURE() << "submit should not run on unresolved texture binding "
                     "target failure";
    return;
  }
  if (!(gpu.destroyedTextureCount == gpu.createdTextureCount &&
        gpu.destroyedBufferCount == gpu.createdBufferCount)) {
    ADD_FAILURE()
        << "materialized resources should be cleaned up on unresolved "
           "texture binding target failure";
    return;
  }
}

TEST_F(RenderGraphExecutorTest,
       ExecutorRejectsUnresolvedTextureBindingWithoutAllocation) {
  auto compileResult = buildExecutorCompiledFrame(230u);
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed";
    return;
  }
  RenderGraphCompileResult invalid = compileResult.value();
  if (!(!invalid.unresolvedTextureBindings.empty())) {
    ADD_FAILURE() << "expected unresolved texture bindings";
    return;
  }
  const uint32_t resourceIndex =
      invalid.unresolvedTextureBindings[0u].textureResourceIndex;
  if (!(resourceIndex < invalid.transientTextureAllocationByResource.size())) {
    ADD_FAILURE()
        << "unresolved texture resource index must be in allocation map "
           "range";
    return;
  }
  invalid.transientTextureAllocationByResource[resourceIndex] = UINT32_MAX;

  FakeGPUDevice gpu;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, invalid);
  if (!(executeResult.hasError())) {
    ADD_FAILURE()
        << "executor should reject unresolved texture binding without "
           "materialized allocation";
    return;
  }
  if (((executeResult.error())
           .find("unresolved texture binding has no materialized "
                 "allocation") == std::string_view::npos)) {
    ADD_FAILURE() << "expected unresolved texture no-allocation error";
    return;
  }
  if (!(gpu.submitCount == 0u)) {
    ADD_FAILURE()
        << "submit should not run on unresolved texture no-allocation "
           "failure";
    return;
  }
  if (!(gpu.destroyedTextureCount == gpu.createdTextureCount &&
        gpu.destroyedBufferCount == gpu.createdBufferCount)) {
    ADD_FAILURE()
        << "materialized resources should be cleaned up on unresolved "
           "texture no-allocation failure";
    return;
  }
}

TEST_F(RenderGraphExecutorTest,
       ExecutorRejectsUnresolvedDependencyBindingWithoutAllocation) {
  auto compileResult = buildExecutorCompiledFrame(240u);
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed";
    return;
  }
  RenderGraphCompileResult invalid = compileResult.value();
  if (!(!invalid.unresolvedDependencyBufferBindings.empty())) {
    ADD_FAILURE() << "expected unresolved dependency bindings";
    return;
  }
  const uint32_t resourceIndex =
      invalid.unresolvedDependencyBufferBindings[0u].bufferResourceIndex;
  if (!(resourceIndex < invalid.transientBufferAllocationByResource.size())) {
    ADD_FAILURE()
        << "unresolved dependency resource index must be in allocation "
           "map range";
    return;
  }
  invalid.transientBufferAllocationByResource[resourceIndex] = UINT32_MAX;

  FakeGPUDevice gpu;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, invalid);
  if (!(executeResult.hasError())) {
    ADD_FAILURE() << "executor should reject unresolved dependency binding "
                     "without materialized allocation";
    return;
  }
  if (((executeResult.error())
           .find("unresolved dependency buffer binding has no "
                 "materialized allocation") == std::string_view::npos)) {
    ADD_FAILURE() << "expected unresolved dependency no-allocation error";
    return;
  }
  if (!(gpu.submitCount == 0u)) {
    ADD_FAILURE() << "submit should not run on unresolved dependency "
                     "no-allocation failure";
    return;
  }
  if (!(gpu.destroyedTextureCount == gpu.createdTextureCount &&
        gpu.destroyedBufferCount == gpu.createdBufferCount)) {
    ADD_FAILURE()
        << "materialized resources should be cleaned up on unresolved "
           "dependency no-allocation failure";
    return;
  }
}

TEST_F(RenderGraphExecutorTest,
       ExecutorRejectsUnresolvedPreDispatchBindingWithoutAllocation) {
  auto compileResult = buildExecutorCompiledFrame(250u);
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed";
    return;
  }
  RenderGraphCompileResult invalid = compileResult.value();
  if (!(!invalid.unresolvedPreDispatchDependencyBufferBindings.empty())) {
    ADD_FAILURE() << "expected unresolved pre-dispatch bindings";
    return;
  }
  invalid.unresolvedDependencyBufferBindings.clear();
  invalid.unresolvedDrawBufferBindings.clear();
  const uint32_t resourceIndex =
      invalid.unresolvedPreDispatchDependencyBufferBindings[0u]
          .bufferResourceIndex;
  if (!(resourceIndex < invalid.transientBufferAllocationByResource.size())) {
    ADD_FAILURE() << "unresolved pre-dispatch resource index must be in "
                     "allocation map range";
    return;
  }
  invalid.transientBufferAllocationByResource[resourceIndex] = UINT32_MAX;

  FakeGPUDevice gpu;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, invalid);
  if (!(executeResult.hasError())) {
    ADD_FAILURE() << "executor should reject unresolved pre-dispatch binding "
                     "without materialized allocation";
    return;
  }
  if (((executeResult.error())
           .find(
               "unresolved pre-dispatch dependency binding has no materialized "
               "allocation") == std::string_view::npos)) {
    ADD_FAILURE() << "expected unresolved pre-dispatch no-allocation error";
    return;
  }
  if (!(gpu.submitCount == 0u)) {
    ADD_FAILURE() << "submit should not run on unresolved pre-dispatch "
                     "no-allocation failure";
    return;
  }
  if (!(gpu.destroyedTextureCount == gpu.createdTextureCount &&
        gpu.destroyedBufferCount == gpu.createdBufferCount)) {
    ADD_FAILURE()
        << "materialized resources should be cleaned up on unresolved "
           "pre-dispatch no-allocation failure";
    return;
  }
}

TEST_F(RenderGraphExecutorTest,
       ExecutorRejectsUnresolvedDrawBindingWithoutAllocation) {
  auto compileResult = buildExecutorCompiledFrame(260u);
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed";
    return;
  }
  RenderGraphCompileResult invalid = compileResult.value();
  if (!(!invalid.unresolvedDrawBufferBindings.empty())) {
    ADD_FAILURE() << "expected unresolved draw bindings";
    return;
  }
  invalid.unresolvedDependencyBufferBindings.clear();
  invalid.unresolvedPreDispatchDependencyBufferBindings.clear();
  const uint32_t resourceIndex =
      invalid.unresolvedDrawBufferBindings[0u].bufferResourceIndex;
  if (!(resourceIndex < invalid.transientBufferAllocationByResource.size())) {
    ADD_FAILURE() << "unresolved draw resource index must be in allocation map "
                     "range";
    return;
  }
  invalid.transientBufferAllocationByResource[resourceIndex] = UINT32_MAX;

  FakeGPUDevice gpu;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, invalid);
  if (!(executeResult.hasError())) {
    ADD_FAILURE() << "executor should reject unresolved draw binding without "
                     "materialized allocation";
    return;
  }
  if (((executeResult.error())
           .find("unresolved draw buffer binding has no materialized "
                 "allocation") == std::string_view::npos)) {
    ADD_FAILURE() << "expected unresolved draw no-allocation error";
    return;
  }
  if (!(gpu.submitCount == 0u)) {
    ADD_FAILURE() << "submit should not run on unresolved draw no-allocation "
                     "failure";
    return;
  }
  if (!(gpu.destroyedTextureCount == gpu.createdTextureCount &&
        gpu.destroyedBufferCount == gpu.createdBufferCount)) {
    ADD_FAILURE()
        << "materialized resources should be cleaned up on unresolved "
           "draw no-allocation failure";
    return;
  }
}

TEST_F(RenderGraphExecutorTest,
       ExecutorRejectsPassDependencyRangeCountMismatch) {
  auto compileResult = buildExecutorCompiledFrame(270u);
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed";
    return;
  }
  RenderGraphCompileResult invalid = compileResult.value();
  invalid.dependencyBufferRangesByPass.clear();

  FakeGPUDevice gpu;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, invalid);
  if (!(executeResult.hasError())) {
    ADD_FAILURE() << "executor should reject pass dependency range metadata "
                     "count mismatch";
    return;
  }
  if (((executeResult.error())
           .find("pass dependency buffer range metadata count mismatch") ==
       std::string_view::npos)) {
    ADD_FAILURE() << "expected pass dependency range metadata mismatch error";
    return;
  }
  if (!(gpu.submitCount == 0u)) {
    ADD_FAILURE() << "submit should not run on pass dependency range count "
                     "mismatch";
    return;
  }
  if (!(gpu.destroyedTextureCount == gpu.createdTextureCount &&
        gpu.destroyedBufferCount == gpu.createdBufferCount)) {
    ADD_FAILURE() << "materialized resources should be cleaned up on pass "
                     "dependency range count mismatch";
    return;
  }
}

TEST_F(RenderGraphExecutorTest,
       ExecutorRejectsPassDependencyRangeOverContractLimit) {
  auto compileResult = buildExecutorCompiledFrame(275u);
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed";
    return;
  }

  RenderGraphCompileResult invalid = compileResult.value();
  if (!(!invalid.dependencyBufferRangesByPass.empty())) {
    ADD_FAILURE() << "expected pass dependency ranges";
    return;
  }

  invalid.dependencyBufferRangesByPass[0u].count =
      static_cast<uint32_t>(kMaxDependencyBuffers + 1u);
  invalid.resolvedDependencyBuffers.resize(kMaxDependencyBuffers + 1u,
                                           BufferHandle{});

  FakeGPUDevice gpu;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, invalid);
  if (!(executeResult.hasError())) {
    ADD_FAILURE() << "executor should reject pass dependency range over "
                     "kMaxDependencyBuffers";
    return;
  }
  if (((executeResult.error())
           .find("pass dependency buffer range exceeds "
                 "kMaxDependencyBuffers") == std::string_view::npos)) {
    ADD_FAILURE() << "expected pass dependency range contract-limit "
                     "error";
    return;
  }
  if (!(gpu.submitCount == 0u)) {
    ADD_FAILURE() << "submit should not run on pass dependency contract-limit "
                     "failure";
    return;
  }
  if (!(gpu.destroyedTextureCount == gpu.createdTextureCount &&
        gpu.destroyedBufferCount == gpu.createdBufferCount)) {
    ADD_FAILURE() << "materialized resources should be cleaned up on pass "
                     "dependency contract-limit failure";
    return;
  }
}

TEST_F(RenderGraphExecutorTest,
       ExecutorRejectsPassPreDispatchRangeCountMismatch) {
  auto compileResult = buildExecutorCompiledFrame(280u);
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed";
    return;
  }
  RenderGraphCompileResult invalid = compileResult.value();
  invalid.preDispatchRangesByPass.clear();

  FakeGPUDevice gpu;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, invalid);
  if (!(executeResult.hasError())) {
    ADD_FAILURE() << "executor should reject pass pre-dispatch range metadata "
                     "count mismatch";
    return;
  }
  if (((executeResult.error())
           .find("pass pre-dispatch range metadata count mismatch") ==
       std::string_view::npos)) {
    ADD_FAILURE() << "expected pass pre-dispatch range metadata mismatch error";
    return;
  }
  if (!(gpu.submitCount == 0u)) {
    ADD_FAILURE() << "submit should not run on pass pre-dispatch range count "
                     "mismatch";
    return;
  }
  if (!(gpu.destroyedTextureCount == gpu.createdTextureCount &&
        gpu.destroyedBufferCount == gpu.createdBufferCount)) {
    ADD_FAILURE() << "materialized resources should be cleaned up on pass "
                     "pre-dispatch range count mismatch";
    return;
  }
}

TEST_F(RenderGraphExecutorTest,
       ExecutorRejectsPreDispatchDependencyRangeOverContractLimit) {
  auto compileResult = buildExecutorCompiledFrame(285u);
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed";
    return;
  }

  RenderGraphCompileResult invalid = compileResult.value();
  if (!(!invalid.preDispatchDependencyRanges.empty())) {
    ADD_FAILURE() << "expected pre-dispatch dependency ranges";
    return;
  }

  invalid.preDispatchDependencyRanges[0u].count =
      static_cast<uint32_t>(kMaxDependencyBuffers + 1u);
  invalid.resolvedPreDispatchDependencyBuffers.resize(
      kMaxDependencyBuffers + 1u, BufferHandle{});

  FakeGPUDevice gpu;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, invalid);
  if (!(executeResult.hasError())) {
    ADD_FAILURE()
        << "executor should reject pre-dispatch dependency range over "
           "kMaxDependencyBuffers";
    return;
  }
  if (((executeResult.error())
           .find("pre-dispatch dependency range exceeds "
                 "kMaxDependencyBuffers") == std::string_view::npos)) {
    ADD_FAILURE() << "expected pre-dispatch dependency range "
                     "contract-limit error";
    return;
  }
  if (!(gpu.submitCount == 0u)) {
    ADD_FAILURE() << "submit should not run on pre-dispatch dependency "
                     "contract-limit failure";
    return;
  }
  if (!(gpu.destroyedTextureCount == gpu.createdTextureCount &&
        gpu.destroyedBufferCount == gpu.createdBufferCount)) {
    ADD_FAILURE() << "materialized resources should be cleaned up on "
                     "pre-dispatch dependency contract-limit failure";
    return;
  }
}

TEST_F(RenderGraphExecutorTest, ExecutorRejectsPassDrawRangeCountMismatch) {
  auto compileResult = buildExecutorCompiledFrame(290u);
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed";
    return;
  }
  RenderGraphCompileResult invalid = compileResult.value();
  invalid.drawRangesByPass.clear();

  FakeGPUDevice gpu;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, invalid);
  if (!(executeResult.hasError())) {
    ADD_FAILURE() << "executor should reject pass draw range metadata count "
                     "mismatch";
    return;
  }
  if (((executeResult.error())
           .find("pass draw range metadata count mismatch") ==
       std::string_view::npos)) {
    ADD_FAILURE() << "expected pass draw range metadata mismatch error";
    return;
  }
  if (!(gpu.submitCount == 0u)) {
    ADD_FAILURE() << "submit should not run on pass draw range count mismatch";
    return;
  }
  if (!(gpu.destroyedTextureCount == gpu.createdTextureCount &&
        gpu.destroyedBufferCount == gpu.createdBufferCount)) {
    ADD_FAILURE() << "materialized resources should be cleaned up on pass draw "
                     "range count mismatch";
    return;
  }
}

TEST_F(RenderGraphExecutorTest, ExecutorRejectsPassDependencyRangeOutOfBounds) {
  auto compileResult = buildExecutorCompiledFrame(300u);
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed";
    return;
  }
  RenderGraphCompileResult invalid = compileResult.value();
  if (!(!invalid.dependencyBufferRangesByPass.empty())) {
    ADD_FAILURE() << "expected dependency buffer ranges";
    return;
  }
  invalid.dependencyBufferRangesByPass[0u].offset =
      static_cast<uint32_t>(invalid.resolvedDependencyBuffers.size());
  invalid.dependencyBufferRangesByPass[0u].count = 1u;

  FakeGPUDevice gpu;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, invalid);
  if (!(executeResult.hasError())) {
    ADD_FAILURE() << "executor should reject out-of-bounds pass dependency "
                     "buffer range";
    return;
  }
  if (((executeResult.error())
           .find("pass dependency buffer range is out of bounds") ==
       std::string_view::npos)) {
    ADD_FAILURE() << "expected pass dependency range bounds error";
    return;
  }
  if (!(gpu.submitCount == 0u)) {
    ADD_FAILURE() << "submit should not run on pass dependency range bounds "
                     "failure";
    return;
  }
  if (!(gpu.destroyedTextureCount == gpu.createdTextureCount &&
        gpu.destroyedBufferCount == gpu.createdBufferCount)) {
    ADD_FAILURE() << "materialized resources should be cleaned up on pass "
                     "dependency range bounds failure";
    return;
  }
}

TEST_F(RenderGraphExecutorTest,
       ExecutorRejectsPassPreDispatchRangeOutOfBounds) {
  auto compileResult = buildExecutorCompiledFrame(310u);
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed";
    return;
  }
  RenderGraphCompileResult invalid = compileResult.value();
  if (!(!invalid.preDispatchRangesByPass.empty())) {
    ADD_FAILURE() << "expected pre-dispatch ranges";
    return;
  }
  invalid.preDispatchRangesByPass[0u].count = UINT32_MAX;

  FakeGPUDevice gpu;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, invalid);
  if (!(executeResult.hasError())) {
    ADD_FAILURE() << "executor should reject out-of-bounds pass pre-dispatch "
                     "range";
    return;
  }
  if (((executeResult.error())
           .find("pass pre-dispatch range is out of bounds") ==
       std::string_view::npos)) {
    ADD_FAILURE() << "expected pass pre-dispatch range bounds error";
    return;
  }
  if (!(gpu.submitCount == 0u)) {
    ADD_FAILURE() << "submit should not run on pass pre-dispatch range bounds "
                     "failure";
    return;
  }
  if (!(gpu.destroyedTextureCount == gpu.createdTextureCount &&
        gpu.destroyedBufferCount == gpu.createdBufferCount)) {
    ADD_FAILURE() << "materialized resources should be cleaned up on pass "
                     "pre-dispatch range bounds failure";
    return;
  }
}

TEST_F(RenderGraphExecutorTest,
       ExecutorRejectsUnresolvedTextureBindingPassIndexOutOfRange) {
  auto compileResult = buildExecutorCompiledFrame(320u);
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed";
    return;
  }
  RenderGraphCompileResult invalid = compileResult.value();
  if (!(!invalid.unresolvedTextureBindings.empty())) {
    ADD_FAILURE() << "expected unresolved texture bindings";
    return;
  }
  invalid.unresolvedTextureBindings[0u].orderedPassIndex = UINT32_MAX;

  FakeGPUDevice gpu;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, invalid);
  if (!(executeResult.hasError())) {
    ADD_FAILURE() << "executor should reject unresolved texture binding pass "
                     "index out of range";
    return;
  }
  if (((executeResult.error())
           .find("unresolved texture binding pass index is out of range") ==
       std::string_view::npos)) {
    ADD_FAILURE() << "expected unresolved texture pass-index bounds error";
    return;
  }
  if (!(gpu.submitCount == 0u)) {
    ADD_FAILURE() << "submit should not run on unresolved texture pass-index "
                     "failure";
    return;
  }
  if (!(gpu.destroyedTextureCount == gpu.createdTextureCount &&
        gpu.destroyedBufferCount == gpu.createdBufferCount)) {
    ADD_FAILURE()
        << "materialized resources should be cleaned up on unresolved "
           "texture pass-index failure";
    return;
  }
}

TEST_F(RenderGraphExecutorTest,
       ExecutorRejectsUnresolvedDependencyBindingPassIndexOutOfRange) {
  auto compileResult = buildExecutorCompiledFrame(330u);
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed";
    return;
  }
  RenderGraphCompileResult invalid = compileResult.value();
  if (!(!invalid.unresolvedDependencyBufferBindings.empty())) {
    ADD_FAILURE() << "expected unresolved dependency bindings";
    return;
  }
  invalid.unresolvedDependencyBufferBindings[0u].orderedPassIndex = UINT32_MAX;

  FakeGPUDevice gpu;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, invalid);
  if (!(executeResult.hasError())) {
    ADD_FAILURE()
        << "executor should reject unresolved dependency binding pass "
           "index out of range";
    return;
  }
  if (((executeResult.error())
           .find("unresolved dependency buffer binding pass index is out of "
                 "range") == std::string_view::npos)) {
    ADD_FAILURE() << "expected unresolved dependency pass-index bounds error";
    return;
  }
  if (!(gpu.submitCount == 0u)) {
    ADD_FAILURE()
        << "submit should not run on unresolved dependency pass-index "
           "failure";
    return;
  }
  if (!(gpu.destroyedTextureCount == gpu.createdTextureCount &&
        gpu.destroyedBufferCount == gpu.createdBufferCount)) {
    ADD_FAILURE()
        << "materialized resources should be cleaned up on unresolved "
           "dependency pass-index failure";
    return;
  }
}

TEST_F(RenderGraphExecutorTest,
       ExecutorRejectsUnresolvedPreDispatchBindingPassIndexOutOfRange) {
  auto compileResult = buildExecutorCompiledFrame(340u);
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed";
    return;
  }
  RenderGraphCompileResult invalid = compileResult.value();
  if (!(!invalid.unresolvedPreDispatchDependencyBufferBindings.empty())) {
    ADD_FAILURE() << "expected unresolved pre-dispatch bindings";
    return;
  }
  invalid.unresolvedPreDispatchDependencyBufferBindings[0u].orderedPassIndex =
      UINT32_MAX;

  FakeGPUDevice gpu;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, invalid);
  if (!(executeResult.hasError())) {
    ADD_FAILURE() << "executor should reject unresolved pre-dispatch binding "
                     "pass index out of range";
    return;
  }
  if (((executeResult.error())
           .find("unresolved pre-dispatch dependency binding pass index is out "
                 "of "
                 "range") == std::string_view::npos)) {
    ADD_FAILURE() << "expected unresolved pre-dispatch pass-index bounds error";
    return;
  }
  if (!(gpu.submitCount == 0u)) {
    ADD_FAILURE() << "submit should not run on unresolved pre-dispatch "
                     "pass-index failure";
    return;
  }
  if (!(gpu.destroyedTextureCount == gpu.createdTextureCount &&
        gpu.destroyedBufferCount == gpu.createdBufferCount)) {
    ADD_FAILURE()
        << "materialized resources should be cleaned up on unresolved "
           "pre-dispatch pass-index failure";
    return;
  }
}

TEST_F(RenderGraphExecutorTest,
       ExecutorRejectsUnresolvedDrawBindingPassIndexOutOfRange) {
  auto compileResult = buildExecutorCompiledFrame(350u);
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed";
    return;
  }
  RenderGraphCompileResult invalid = compileResult.value();
  if (!(!invalid.unresolvedDrawBufferBindings.empty())) {
    ADD_FAILURE() << "expected unresolved draw bindings";
    return;
  }
  invalid.unresolvedDrawBufferBindings[0u].orderedPassIndex = UINT32_MAX;

  FakeGPUDevice gpu;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, invalid);
  if (!(executeResult.hasError())) {
    ADD_FAILURE()
        << "executor should reject unresolved draw binding pass index "
           "out of range";
    return;
  }
  if (((executeResult.error())
           .find("unresolved draw buffer binding pass index is out of "
                 "range") == std::string_view::npos)) {
    ADD_FAILURE() << "expected unresolved draw pass-index bounds error";
    return;
  }
  if (!(gpu.submitCount == 0u)) {
    ADD_FAILURE() << "submit should not run on unresolved draw pass-index "
                     "failure";
    return;
  }
  if (!(gpu.destroyedTextureCount == gpu.createdTextureCount &&
        gpu.destroyedBufferCount == gpu.createdBufferCount)) {
    ADD_FAILURE()
        << "materialized resources should be cleaned up on unresolved "
           "draw pass-index failure";
    return;
  }
}

TEST_F(RenderGraphExecutorTest,
       ExecutorRejectsUnresolvedPreDispatchBindingResourceOutOfRange) {
  auto compileResult = buildExecutorCompiledFrame(360u);
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed";
    return;
  }
  RenderGraphCompileResult invalid = compileResult.value();
  if (!(!invalid.unresolvedPreDispatchDependencyBufferBindings.empty())) {
    ADD_FAILURE() << "expected unresolved pre-dispatch bindings";
    return;
  }
  invalid.unresolvedPreDispatchDependencyBufferBindings[0u]
      .bufferResourceIndex = UINT32_MAX;

  FakeGPUDevice gpu;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, invalid);
  if (!(executeResult.hasError())) {
    ADD_FAILURE() << "executor should reject unresolved pre-dispatch binding "
                     "resource index out of range";
    return;
  }
  if (((executeResult.error())
           .find("unresolved pre-dispatch dependency binding resource index is "
                 "out of "
                 "range") == std::string_view::npos)) {
    ADD_FAILURE()
        << "expected unresolved pre-dispatch resource-index bounds error";
    return;
  }
  if (!(gpu.submitCount == 0u)) {
    ADD_FAILURE() << "submit should not run on unresolved pre-dispatch "
                     "resource-index failure";
    return;
  }
  if (!(gpu.destroyedTextureCount == gpu.createdTextureCount &&
        gpu.destroyedBufferCount == gpu.createdBufferCount)) {
    ADD_FAILURE()
        << "materialized resources should be cleaned up on unresolved "
           "pre-dispatch resource-index failure";
    return;
  }
}

TEST_F(RenderGraphExecutorTest,
       ExecutorRejectsUnresolvedPreDispatchBindingSlotOutOfRange) {
  auto compileResult = buildExecutorCompiledFrame(370u);
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed";
    return;
  }
  RenderGraphCompileResult invalid = compileResult.value();
  if (!(!invalid.unresolvedPreDispatchDependencyBufferBindings.empty())) {
    ADD_FAILURE() << "expected unresolved pre-dispatch bindings";
    return;
  }
  invalid.unresolvedPreDispatchDependencyBufferBindings[0u]
      .dependencyBufferIndex = UINT32_MAX;

  FakeGPUDevice gpu;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, invalid);
  if (!(executeResult.hasError())) {
    ADD_FAILURE() << "executor should reject unresolved pre-dispatch binding "
                     "slot index out of range";
    return;
  }
  if (((executeResult.error())
           .find("unresolved pre-dispatch dependency binding slot index is out "
                 "of "
                 "range") == std::string_view::npos)) {
    ADD_FAILURE() << "expected unresolved pre-dispatch slot-index bounds error";
    return;
  }
  if (!(gpu.submitCount == 0u)) {
    ADD_FAILURE() << "submit should not run on unresolved pre-dispatch "
                     "slot-index failure";
    return;
  }
  if (!(gpu.destroyedTextureCount == gpu.createdTextureCount &&
        gpu.destroyedBufferCount == gpu.createdBufferCount)) {
    ADD_FAILURE()
        << "materialized resources should be cleaned up on unresolved "
           "pre-dispatch slot-index failure";
    return;
  }
}

TEST_F(RenderGraphExecutorTest,
       ExecutorRejectsUnresolvedDrawBindingResourceOutOfRange) {
  auto compileResult = buildExecutorCompiledFrame(380u);
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed";
    return;
  }
  RenderGraphCompileResult invalid = compileResult.value();
  if (!(!invalid.unresolvedDrawBufferBindings.empty())) {
    ADD_FAILURE() << "expected unresolved draw bindings";
    return;
  }
  invalid.unresolvedDrawBufferBindings[0u].bufferResourceIndex = UINT32_MAX;

  FakeGPUDevice gpu;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, invalid);
  if (!(executeResult.hasError())) {
    ADD_FAILURE() << "executor should reject unresolved draw binding resource "
                     "index out of range";
    return;
  }
  if (((executeResult.error())
           .find("unresolved draw buffer binding resource index is out "
                 "of range") == std::string_view::npos)) {
    ADD_FAILURE() << "expected unresolved draw resource-index bounds error";
    return;
  }
  if (!(gpu.submitCount == 0u)) {
    ADD_FAILURE() << "submit should not run on unresolved draw resource-index "
                     "failure";
    return;
  }
  if (!(gpu.destroyedTextureCount == gpu.createdTextureCount &&
        gpu.destroyedBufferCount == gpu.createdBufferCount)) {
    ADD_FAILURE()
        << "materialized resources should be cleaned up on unresolved "
           "draw resource-index failure";
    return;
  }
}

} // namespace
