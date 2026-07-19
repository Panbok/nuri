#include "tests_pch.h"

#include "render_graph_test_support.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <vector>

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
                          const RenderGraphRuntimeConfig &config,
                          RenderGraphExecutionOptions options = {}) {
  auto beginResult = gpu.beginFrame(compiled.frameIndex);
  if (beginResult.hasError()) {
    return Result<RenderGraphExecutionMetadata, std::string>::makeError(
        beginResult.error());
  }
  RenderGraphRuntime runtime(config);
  return executor.execute(runtime, gpu, compiled, options);
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
buildComputeOnlyCompiledFrame(uint64_t frameIndex) {
  RenderGraphBuilder builder;
  builder.beginFrame(frameIndex);

  auto sourceTextureResult = builder.createTransientTexture(
      makeTransientTextureDesc(Format::RG32_FLOAT, 1u, 1u),
      "exec_compute_only_source");
  auto resultBufferResult = builder.createTransientBuffer(
      makeTransientBufferDesc(32u), "exec_compute_only_result");
  if (sourceTextureResult.hasError()) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        sourceTextureResult.error());
  }
  if (resultBufferResult.hasError()) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        resultBufferResult.error());
  }

  const std::array<BufferHandle, 1u> dispatchDependencies = {BufferHandle{}};
  ComputeDispatchItem dispatch{};
  dispatch.pipeline = ComputePipelineHandle{.index = 91u, .generation = 1u};
  dispatch.dispatch = {.x = 1u, .y = 1u, .z = 1u};
  dispatch.dependencyBuffers = std::span<const BufferHandle>(
      dispatchDependencies.data(), dispatchDependencies.size());
  dispatch.debugLabel = "exec_compute_dispatch";
  const std::array<ComputeDispatchItem, 1u> preDispatches = {dispatch};

  RenderGraphGraphicsPassDesc passDesc{};
  passDesc.executionMode = RenderPassExecutionMode::ComputeOnly;
  passDesc.hasColorAttachment = false;
  passDesc.preDispatches = std::span<const ComputeDispatchItem>(
      preDispatches.data(), preDispatches.size());
  passDesc.debugLabel = "exec_compute_only_pass";
  passDesc.markImplicitOutputSideEffect = true;

  auto passResult = builder.addGraphicsPass(passDesc);
  if (passResult.hasError()) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        passResult.error());
  }
  auto bindResult =
      builder.addTextureRead(passResult.value(), sourceTextureResult.value());
  if (bindResult.hasError()) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        bindResult.error());
  }
  bindResult = builder.bindPreDispatchDependencyBuffer(
      passResult.value(), 0u, 0u, resultBufferResult.value(),
      RenderGraphAccessMode::Write);
  if (bindResult.hasError()) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        bindResult.error());
  }

  return compileBuilder(builder);
}

Result<RenderGraphCompileResult, std::string>
buildMeshDispatchCompiledFrame(uint64_t frameIndex) {
  RenderGraphBuilder builder;
  builder.beginFrame(frameIndex);

  auto colorResult = builder.createTransientTexture(
      makeTransientTextureDesc(Format::RGBA8_UNORM, 32u, 32u),
      "exec_mesh_dispatch_color");
  if (colorResult.hasError()) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        colorResult.error());
  }

  const std::array<std::byte, 4u> pushConstants{
      std::byte{0x41}, std::byte{0x42}, std::byte{0x43}, std::byte{0x44}};
  const std::array<BufferHandle, 1u> dependencyBuffers{
      BufferHandle{.index = 70u, .generation = 1u}};

  MeshDispatchItem dispatch{};
  dispatch.pipeline = MeshletPipelineHandle{.index = 17u, .generation = 1u};
  dispatch.groupsX = 5u;
  dispatch.groupsY = 1u;
  dispatch.groupsZ = 1u;
  dispatch.pushConstants =
      std::span<const std::byte>(pushConstants.data(), pushConstants.size());
  dispatch.dependencyBuffers = std::span<const BufferHandle>(
      dependencyBuffers.data(), dependencyBuffers.size());
  dispatch.debugLabel = "exec_mesh_dispatch";
  const std::array<MeshDispatchItem, 1u> dispatches{dispatch};

  RenderGraphGraphicsPassDesc passDesc{};
  passDesc.colorTexture = colorResult.value();
  passDesc.meshDispatches =
      std::span<const MeshDispatchItem>(dispatches.data(), dispatches.size());
  passDesc.debugLabel = "exec_mesh_dispatch_pass";

  auto passResult = builder.addGraphicsPass(passDesc);
  if (passResult.hasError()) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        passResult.error());
  }

  return compileBuilder(builder);
}

Result<RenderGraphCompileResult, std::string>
buildTransientMeshDispatchIndirectCompiledFrame(uint64_t frameIndex) {
  RenderGraphBuilder builder;
  builder.beginFrame(frameIndex);

  auto colorResult = builder.createTransientTexture(
      makeTransientTextureDesc(Format::RGBA8_UNORM, 32u, 32u),
      "exec_mesh_indirect_color");
  if (colorResult.hasError()) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        colorResult.error());
  }

  BufferDesc indirectDesc = makeTransientBufferDesc(64u);
  indirectDesc.usage = BufferUsage::Storage | BufferUsage::Indirect;
  auto indirectResult =
      builder.createTransientBuffer(indirectDesc, "exec_mesh_indirect_args");
  auto countResult =
      builder.createTransientBuffer(indirectDesc, "exec_mesh_indirect_count");
  if (indirectResult.hasError()) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        indirectResult.error());
  }
  if (countResult.hasError()) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        countResult.error());
  }

  MeshDispatchItem dispatch{};
  dispatch.command = MeshDispatchCommandType::IndirectCount;
  dispatch.pipeline = MeshletPipelineHandle{.index = 18u, .generation = 1u};
  dispatch.indirectBufferOffset = 16u;
  dispatch.indirectCountBufferOffset = 4u;
  dispatch.indirectDispatchCount = 9u;
  dispatch.debugLabel = "exec_mesh_indirect_dispatch";
  const std::array<MeshDispatchItem, 1u> dispatches{dispatch};

  const std::array<RenderGraphPreparedMeshDispatchBufferBinding, 2u> bindings{{
      {.meshDispatchIndex = 0u,
       .target = RenderGraphMeshDispatchBufferBindingTarget::Indirect,
       .buffer = indirectResult.value()},
      {.meshDispatchIndex = 0u,
       .target = RenderGraphMeshDispatchBufferBindingTarget::IndirectCount,
       .buffer = countResult.value()},
  }};

  RenderGraphGraphicsPassDesc passDesc{};
  passDesc.colorTexture = colorResult.value();
  passDesc.meshDispatches =
      std::span<const MeshDispatchItem>(dispatches.data(), dispatches.size());
  passDesc.meshDispatchBufferBindings =
      std::span<const RenderGraphPreparedMeshDispatchBufferBinding>(
          bindings.data(), bindings.size());
  passDesc.debugLabel = "exec_mesh_indirect_pass";

  auto passResult = builder.addGraphicsPass(passDesc);
  if (passResult.hasError()) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        passResult.error());
  }

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

TEST_F(RenderGraphExecutorTest, TextureCopyPassPatchesTransientDestination) {
  FakeGPUDevice gpu;
  const TextureDesc textureDesc =
      makeTransientTextureDesc(Format::RGBA8_UNORM, 4u, 4u);
  auto sourceResult = gpu.createTexture(textureDesc, "copy_source");
  ASSERT_FALSE(sourceResult.hasError());
  const TextureHandle sourceTexture = sourceResult.value();

  std::vector<std::byte> sourceBytes(4u * 4u * 4u);
  for (size_t i = 0u; i < sourceBytes.size(); ++i) {
    sourceBytes[i] = static_cast<std::byte>(0x80u + i);
  }
  ASSERT_FALSE(gpu.seedTextureBytes(sourceTexture, sourceBytes).hasError());

  RenderGraphBuilder builder;
  builder.beginFrame(151u);
  auto sourceImport = builder.importTexture(sourceTexture, "copy_source");
  auto destinationTextureResult =
      builder.createTransientTexture(textureDesc, "copy_transient_destination");
  ASSERT_FALSE(sourceImport.hasError());
  ASSERT_FALSE(destinationTextureResult.hasError());

  const std::array<RenderGraphTextureCopyItem, 1u> copies{{
      {.sourceTexture = sourceImport.value(),
       .destinationTexture = destinationTextureResult.value(),
       .sourceX = 0u,
       .sourceY = 0u,
       .destinationX = 1u,
       .destinationY = 1u,
       .width = 2u,
       .height = 2u},
  }};
  RenderGraphTextureCopyPassDesc desc{};
  desc.copies =
      std::span<const RenderGraphTextureCopyItem>(copies.data(), copies.size());
  desc.debugLabel = "exec_transient_copy_pass";
  auto passResult = builder.addTextureCopyPass(desc);
  ASSERT_FALSE(passResult.hasError()) << passResult.error();
  ASSERT_FALSE(builder.markPassSideEffect(passResult.value()).hasError());

  auto compileResult = compileBuilder(builder);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  ASSERT_EQ(compileResult.value().unresolvedTextureCopyBindings.size(), 1u);

  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, compileResult.value());
  ASSERT_FALSE(executeResult.hasError()) << executeResult.error();
  ASSERT_EQ(gpu.recordedPasses.size(), 1u);
  ASSERT_EQ(gpu.recordedPasses[0u].textureCopies.size(), 1u);
  const TextureHandle materializedDestination =
      gpu.recordedPasses[0u].textureCopies[0u].destinationTexture;
  ASSERT_TRUE(nuri::isValid(materializedDestination));
  EXPECT_FALSE(sameTexture(materializedDestination, sourceTexture));

  std::vector<std::byte> destinationBytes(4u * 4u * 4u);
  auto readResult = gpu.readTexture(
      materializedDestination,
      TextureReadbackRegion{.x = 0u, .y = 0u, .width = 4u, .height = 4u},
      destinationBytes);
  ASSERT_FALSE(readResult.hasError()) << readResult.error();

  std::vector<std::byte> expected(4u * 4u * 4u, std::byte{0});
  constexpr size_t kPixelBytes = 4u;
  for (uint32_t row = 0u; row < 2u; ++row) {
    const size_t sourceOffset = static_cast<size_t>(row) * 4u * kPixelBytes;
    const size_t destinationOffset =
        (static_cast<size_t>(1u + row) * 4u + 1u) * kPixelBytes;
    std::copy_n(sourceBytes.begin() + static_cast<std::ptrdiff_t>(sourceOffset),
                2u * kPixelBytes,
                expected.begin() +
                    static_cast<std::ptrdiff_t>(destinationOffset));
  }
  EXPECT_EQ(destinationBytes, expected);
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
       ComputeOnlyPassExecutesAndPreservesDispatchDependencies) {
  auto compileResult = buildComputeOnlyCompiledFrame(304u);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  const RenderGraphCompileResult &compiled = compileResult.value();
  ASSERT_EQ(compiled.orderedPasses.size(), 1u);
  EXPECT_EQ(compiled.orderedPasses[0].executionMode,
            RenderPassExecutionMode::ComputeOnly);
  ASSERT_EQ(compiled.orderedPasses[0].preDispatches.size(), 1u);
  ASSERT_EQ(compiled.orderedPasses[0].preDispatches[0].dependencyBuffers.size(),
            1u);

  FakeExecutorGPUDevice gpu;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, compiled);
  ASSERT_FALSE(executeResult.hasError()) << executeResult.error();
  ASSERT_TRUE(executeResult.value());

  ASSERT_EQ(gpu.recordedPasses.size(), 1u);
  EXPECT_EQ(gpu.recordedPasses[0].executionMode,
            RenderPassExecutionMode::ComputeOnly);
  EXPECT_FALSE(gpu.recordedPasses[0].hasColorAttachment);
  EXPECT_TRUE(gpu.recordedPasses[0].draws.empty());
  EXPECT_TRUE(nuri::isValid(gpu.lastPreDispatchDependencyBuffer));
}

TEST_F(RenderGraphExecutorTest,
       MeshDispatchIndirectTransientBuffersArePatchedBeforeRecording) {
  auto compileResult = buildTransientMeshDispatchIndirectCompiledFrame(306u);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  const RenderGraphCompileResult &compiled = compileResult.value();
  ASSERT_EQ(compiled.unresolvedMeshDispatchBufferBindings.size(), 2u);

  FakeExecutorGPUDevice gpu;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, compiled);
  ASSERT_FALSE(executeResult.hasError()) << executeResult.error();
  ASSERT_TRUE(executeResult.value());

  ASSERT_EQ(gpu.recordedPasses.size(), 1u);
  const RenderPass &recordedPass = gpu.recordedPasses[0];
  ASSERT_EQ(recordedPass.meshDispatches.size(), 1u);
  const MeshDispatchItem &recordedDispatch = recordedPass.meshDispatches[0];
  EXPECT_EQ(recordedDispatch.command, MeshDispatchCommandType::IndirectCount);
  EXPECT_TRUE(nuri::isValid(recordedDispatch.indirectBuffer));
  EXPECT_TRUE(nuri::isValid(recordedDispatch.indirectCountBuffer));
  EXPECT_FALSE(sameBuffer(recordedDispatch.indirectBuffer,
                          recordedDispatch.indirectCountBuffer));
  EXPECT_EQ(recordedDispatch.indirectBufferOffset, 16u);
  EXPECT_EQ(recordedDispatch.indirectCountBufferOffset, 4u);
  EXPECT_EQ(recordedDispatch.indirectDispatchCount, 9u);
}

TEST_F(RenderGraphExecutorTest,
       ExecutorCleansUpOnPartialMaterializationFailure) {
  auto compileResult = buildExecutorCompiledFrame(110u);
  if (compileResult.hasError()) {
    ADD_FAILURE() << "compile should succeed";
    return;
  }
  const RenderGraphCompileResult &compiled = compileResult.value();

  FakeExecutorGPUDevice gpu;
  gpu.failCreateBufferAtCall = 1u;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, compiled);
  if (!executeResult.hasError()) {
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
  if (compileResult.hasError()) {
    ADD_FAILURE() << "compile should succeed";
    return;
  }
  const RenderGraphCompileResult &compiled = compileResult.value();

  FakeExecutorGPUDevice gpu;
  gpu.failSubmitFrame = true;
  RenderGraphExecutor executor;
  auto executeResult = executeCompiled(executor, gpu, compiled);
  if (!executeResult.hasError()) {
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
  if (compile123.hasError()) {
    ADD_FAILURE() << "empty frame 123 compile should succeed";
    return;
  }
  executeResult = executeCompiled(executor, gpu, compile123.value());
  if (executeResult.hasError() || !executeResult.value()) {
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
  if (compile124.hasError()) {
    ADD_FAILURE() << "non-empty frame 124 compile should succeed";
    return;
  }
  executeResult = executeCompiled(executor, gpu, compile124.value());
  if (executeResult.hasError() || !executeResult.value()) {
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

TEST_F(RenderGraphExecutorTest,
       PresentationFailureRetainsTheExecutedSubmissionToken) {
  auto compiled = buildExecutorCompiledFrame(130u);
  ASSERT_FALSE(compiled.hasError()) << compiled.error();

  FakeExecutorGPUDevice gpu;
  gpu.failPresentFrame = true;
  RenderGraphExecutor executor;
  auto result = executeCompiled(executor, gpu, compiled.value());
  ASSERT_TRUE(result.hasError());
  EXPECT_TRUE(hasExecutionFailureStage(
      result.error(), RenderGraphExecutionFailureStage::PresentFrameOutput));
  EXPECT_TRUE(nuri::isValid(gpu.lastSubmittedFrameHandle));
  EXPECT_EQ(gpu.destroyedTextureCount, 0u);
  EXPECT_EQ(gpu.destroyedBufferCount, 0u);

  gpu.failPresentFrame = false;
  const uint32_t texturesBeforeCompletion = gpu.createdTextureCount;
  const uint32_t buffersBeforeCompletion = gpu.createdBufferCount;
  auto beforeCompletion = buildExecutorCompiledFrame(131u);
  ASSERT_FALSE(beforeCompletion.hasError()) << beforeCompletion.error();
  result = executeCompiled(executor, gpu, beforeCompletion.value());
  ASSERT_FALSE(result.hasError()) << result.error();
  EXPECT_EQ(gpu.createdTextureCount, texturesBeforeCompletion * 2u);
  EXPECT_EQ(gpu.createdBufferCount, buffersBeforeCompletion * 2u);
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
       ExecutorCapturesTelemetryAndPassTimingOnlyWhenRequested) {
  const RenderGraphRuntimeConfig config{};

  {
    auto compileResult = buildExecutorCompiledFrame(142u);
    ASSERT_FALSE(compileResult.hasError());
    FakeExecutorGPUDevice gpu;
    RenderGraphExecutor executor;
    auto executeResult =
        executeCompiledWithConfig(executor, gpu, compileResult.value(), config);
    ASSERT_FALSE(executeResult.hasError());
    EXPECT_TRUE(isValid(executeResult.value().submission));
    EXPECT_TRUE(executeResult.value().recordedCommandBuffers.empty());
    EXPECT_TRUE(executeResult.value().submitBatches.empty());
    EXPECT_TRUE(executeResult.value().passRanges.empty());
    EXPECT_TRUE(executeResult.value().passTimings.empty());
  }

  {
    auto compileResult = buildExecutorCompiledFrame(143u);
    ASSERT_FALSE(compileResult.hasError());
    FakeExecutorGPUDevice gpu;
    RenderGraphExecutor executor;
    auto executeResult = executeCompiledWithConfig(
        executor, gpu, compileResult.value(), config,
        {.telemetry = RenderGraphTelemetryLevel::Metadata});
    ASSERT_FALSE(executeResult.hasError());
    EXPECT_EQ(executeResult.value().recordedCommandBuffers.size(), 1u);
    EXPECT_EQ(executeResult.value().submitBatches.size(), 1u);
    EXPECT_EQ(executeResult.value().passRanges.size(), 1u);
    EXPECT_TRUE(executeResult.value().passTimings.empty());
  }

  {
    auto compileResult = buildExecutorCompiledFrame(144u);
    ASSERT_FALSE(compileResult.hasError());
    FakeExecutorGPUDevice gpu;
    RenderGraphExecutor executor;
    auto executeResult = executeCompiledWithConfig(
        executor, gpu, compileResult.value(), config,
        {.telemetry = RenderGraphTelemetryLevel::PassTimings});
    ASSERT_FALSE(executeResult.hasError());
    ASSERT_EQ(executeResult.value().passTimings.size(), 1u);
    EXPECT_EQ(executeResult.value().passTimings[0].orderedPassIndex, 0u);
  }
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

} // namespace
