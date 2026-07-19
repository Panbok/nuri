#include "tests_pch.h"

#include "render_graph_test_support.h"

#include <gtest/gtest.h>

#include "nuri/gfx/frame/render_frame_context.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>

namespace {

using namespace nuri;
using namespace nuri::test_support;

Result<RenderGraphCompileResult, std::string>
compileBuilder(RenderGraphBuilder &builder) {
  RenderGraphRuntime runtime;
  return builder.compile(runtime);
}

Result<RenderGraphCompileResult, std::string>
compileBuilderWithConfig(RenderGraphBuilder &builder,
                         const RenderGraphRuntimeConfig &config) {
  RenderGraphRuntime runtime(config);
  return builder.compile(runtime);
}

TEST(RenderGraphCompileBehaviorTest, CompileDeterminismAndTieBreak) {
  RenderGraphBuilder builder;
  builder.beginFrame(201u);

  auto pass0Result =
      addTestGraphicsPass(builder, makeTestPass("det_p0"), "det_p0");
  auto pass1Result =
      addTestGraphicsPass(builder, makeTestPass("det_p1"), "det_p1");
  auto pass2Result =
      addTestGraphicsPass(builder, makeTestPass("det_p2"), "det_p2");
  ASSERT_FALSE(pass0Result.hasError());
  ASSERT_FALSE(pass1Result.hasError());
  ASSERT_FALSE(pass2Result.hasError());

  auto depResult =
      builder.addDependency(pass1Result.value(), pass2Result.value());
  if (!(!depResult.hasError())) {
    ADD_FAILURE() << "addDependency p1->p2 should succeed";
    return;
  }
  depResult = builder.addDependency(pass0Result.value(), pass2Result.value());
  if (!(!depResult.hasError())) {
    ADD_FAILURE() << "addDependency p0->p2 should succeed";
    return;
  }

  auto compileAResult = compileBuilder(builder);
  auto compileBResult = compileBuilder(builder);
  if (!(!compileAResult.hasError() && !compileBResult.hasError())) {
    ADD_FAILURE() << "compile should succeed for determinism graph";
    if (compileAResult.hasError()) {
      std::cerr << compileAResult.error() << "\n";
    }
    if (compileBResult.hasError()) {
      std::cerr << compileBResult.error() << "\n";
    }
    return;
  }

  const RenderGraphCompileResult &compiledA = compileAResult.value();
  const RenderGraphCompileResult &compiledB = compileBResult.value();

  if (!(compiledA.orderedPassIndices.size() == 3u)) {
    ADD_FAILURE() << "determinism graph should schedule 3 passes";
    return;
  }
  if (!(compiledA.orderedPassIndices[0u] == 0u &&
        compiledA.orderedPassIndices[1u] == 1u &&
        compiledA.orderedPassIndices[2u] == 2u)) {
    ADD_FAILURE() << "tie-break ordering should follow pass declaration index";
    return;
  }
  if (!(compiledA.orderedPassIndices == compiledB.orderedPassIndices)) {
    ADD_FAILURE()
        << "ordered pass indices should be stable across compile calls";
    return;
  }
  if (!(compiledA.edges.size() == compiledB.edges.size())) {
    ADD_FAILURE() << "edge count should be stable across compile calls";
    return;
  }
  for (size_t i = 0; i < compiledA.edges.size(); ++i) {
    if (!(compiledA.edges[i].before == compiledB.edges[i].before &&
          compiledA.edges[i].after == compiledB.edges[i].after)) {
      ADD_FAILURE()
          << "edge ordering/content should be stable across compile calls";
      return;
    }
  }
}

TEST(RenderGraphCompileBehaviorTest,
     ParallelCompilePreservesDeterministicLifetimesAndBarrierPlans) {
  RenderGraphBuilder builder;
  builder.beginFrame(202u);
  constexpr uint32_t kParallelBufferChainLength = 64u;
  std::vector<RenderGraphBufferId> buffers;
  std::vector<RenderGraphPassId> passes;
  buffers.reserve(kParallelBufferChainLength);
  passes.reserve(kParallelBufferChainLength);

  for (uint32_t i = 0u; i < kParallelBufferChainLength; ++i) {
    const std::string bufferName = "pc_buf_" + std::to_string(i);
    auto bufferResult =
        builder.createTransientBuffer(makeTransientBufferDesc(64u), bufferName);
    ASSERT_FALSE(bufferResult.hasError());
    buffers.push_back(bufferResult.value());

    const std::string passName = "pc_p" + std::to_string(i);
    auto passResult =
        addTestGraphicsPass(builder, makeTestPass(passName), passName);
    ASSERT_FALSE(passResult.hasError());
    passes.push_back(passResult.value());
  }

  ASSERT_EQ(buffers.size(), kParallelBufferChainLength);
  ASSERT_EQ(passes.size(), kParallelBufferChainLength);

  ASSERT_FALSE(
      builder.addBufferWrite(passes.front(), buffers.front()).hasError());
  for (uint32_t i = 1u; i < kParallelBufferChainLength; ++i) {
    ASSERT_FALSE(builder.addBufferRead(passes[i], buffers[i - 1u]).hasError());
    ASSERT_FALSE(builder.addBufferWrite(passes[i], buffers[i]).hasError());
  }
  ASSERT_FALSE(builder.markPassSideEffect(passes.back()).hasError());

  const RenderGraphRuntimeConfig serialConfig{
      .workerCount = 1u,
      .parallelCompile = true,
      .parallelGraphicsRecording = false,
  };
  const RenderGraphRuntimeConfig parallelConfig{
      .workerCount = 4u,
      .parallelCompile = true,
      .parallelGraphicsRecording = false,
  };

  auto serialCompile = compileBuilderWithConfig(builder, serialConfig);
  auto parallelCompile = compileBuilderWithConfig(builder, parallelConfig);
  ASSERT_FALSE(serialCompile.hasError());
  ASSERT_FALSE(parallelCompile.hasError());

  const RenderGraphCompileResult &serial = serialCompile.value();
  const RenderGraphCompileResult &parallel = parallelCompile.value();

  EXPECT_FALSE(serial.usedParallelCompile);
  EXPECT_TRUE(parallel.usedParallelCompile);
  EXPECT_EQ(serial.orderedPassIndices, parallel.orderedPassIndices);
  EXPECT_EQ(serial.edges.size(), parallel.edges.size());
  EXPECT_EQ(serial.transientBufferLifetimes.size(),
            parallel.transientBufferLifetimes.size());
  EXPECT_EQ(serial.passBarrierPlans.size(), parallel.passBarrierPlans.size());
  EXPECT_EQ(serial.passBarrierRecords.size(),
            parallel.passBarrierRecords.size());

  for (size_t i = 0; i < serial.edges.size(); ++i) {
    const auto &lhs = serial.edges[i];
    const auto &rhs = parallel.edges[i];
    EXPECT_EQ(lhs.before, rhs.before);
    EXPECT_EQ(lhs.after, rhs.after);
  }

  for (size_t i = 0; i < serial.transientBufferLifetimes.size(); ++i) {
    const auto &lhs = serial.transientBufferLifetimes[i];
    const auto &rhs = parallel.transientBufferLifetimes[i];
    EXPECT_EQ(lhs.resourceIndex, rhs.resourceIndex);
    EXPECT_EQ(lhs.firstExecutionIndex, rhs.firstExecutionIndex);
    EXPECT_EQ(lhs.lastExecutionIndex, rhs.lastExecutionIndex);
  }

  for (size_t i = 0; i < serial.passBarrierPlans.size(); ++i) {
    const auto &lhs = serial.passBarrierPlans[i];
    const auto &rhs = parallel.passBarrierPlans[i];
    EXPECT_EQ(lhs.orderedPassIndex, rhs.orderedPassIndex);
    EXPECT_EQ(lhs.barrierCount, rhs.barrierCount);
  }

  for (size_t i = 0; i < serial.passBarrierRecords.size(); ++i) {
    const auto &lhs = serial.passBarrierRecords[i];
    const auto &rhs = parallel.passBarrierRecords[i];
    EXPECT_EQ(lhs.resourceKind, rhs.resourceKind);
    EXPECT_EQ(lhs.resourceIndex, rhs.resourceIndex);
    EXPECT_EQ(lhs.beforeAccess, rhs.beforeAccess);
    EXPECT_EQ(lhs.afterAccess, rhs.afterAccess);
    EXPECT_EQ(lhs.beforeState, rhs.beforeState);
    EXPECT_EQ(lhs.afterState, rhs.afterState);
  }
}

TEST(RenderGraphCompileBehaviorTest,
     MeshDispatchIndirectCountAcceptsTransientBufferBindings) {
  RenderGraphBuilder builder;
  builder.beginFrame(232u);

  BufferDesc indirectDesc = makeTransientBufferDesc(64u);
  indirectDesc.usage = BufferUsage::Storage | BufferUsage::Indirect;
  auto indirectResult =
      builder.createTransientBuffer(indirectDesc, "mesh_indirect_args");
  auto countResult =
      builder.createTransientBuffer(indirectDesc, "mesh_indirect_count");
  ASSERT_FALSE(indirectResult.hasError()) << indirectResult.error();
  ASSERT_FALSE(countResult.hasError()) << countResult.error();

  MeshDispatchItem dispatch{};
  dispatch.command = MeshDispatchCommandType::IndirectCount;
  dispatch.pipeline = MeshletPipelineHandle{.index = 4u, .generation = 1u};
  dispatch.indirectDispatchCount = 16u;
  const std::array<MeshDispatchItem, 1u> dispatches{dispatch};

  const std::array<RenderGraphPreparedMeshDispatchBufferBinding, 2u> bindings{{
      {.meshDispatchIndex = 0u,
       .target = RenderGraphMeshDispatchBufferBindingTarget::Indirect,
       .buffer = indirectResult.value()},
      {.meshDispatchIndex = 0u,
       .target = RenderGraphMeshDispatchBufferBindingTarget::IndirectCount,
       .buffer = countResult.value()},
  }};

  RenderGraphGraphicsPassDesc desc{};
  desc.meshDispatches =
      std::span<const MeshDispatchItem>(dispatches.data(), dispatches.size());
  desc.meshDispatchBufferBindings =
      std::span<const RenderGraphPreparedMeshDispatchBufferBinding>(
          bindings.data(), bindings.size());
  desc.debugLabel = "mesh_indirect_transient";

  auto passResult = builder.addGraphicsPass(desc);
  ASSERT_FALSE(passResult.hasError()) << passResult.error();

  auto compileResult = compileBuilder(builder);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  const RenderGraphCompileResult &compiled = compileResult.value();
  ASSERT_EQ(compiled.unresolvedMeshDispatchBufferBindings.size(), 2u);
  EXPECT_EQ(compiled.unresolvedMeshDispatchBufferBindings[0].meshDispatchIndex,
            0u);
  EXPECT_EQ(
      compiled.unresolvedMeshDispatchBufferBindings[0].target,
      RenderGraphCompileResult::MeshDispatchBufferBindingTarget::Indirect);
  EXPECT_EQ(
      compiled.unresolvedMeshDispatchBufferBindings[1].target,
      RenderGraphCompileResult::MeshDispatchBufferBindingTarget::IndirectCount);
}

TEST(RenderGraphCompileBehaviorTest,
     GraphFingerprintTracksPreDispatchLayoutButNotContent) {
  RenderGraphBuilder builder;

  auto recordFrame = [&](uint64_t frameIndex,
                         std::span<const ComputeDispatchItem> preDispatches)
      -> RenderGraphBuilder::GraphFingerprint {
    builder.beginFrame(frameIndex);
    auto colorResult = builder.createTransientTexture(
        makeTransientTextureDesc(Format::RGBA8_UNORM, 32u, 32u),
        "payload_layout_color");
    EXPECT_FALSE(colorResult.hasError());
    if (colorResult.hasError()) {
      return {};
    }

    RenderGraphGraphicsPassDesc passDesc{};
    passDesc.colorTexture = colorResult.value();
    passDesc.preDispatches = preDispatches;
    passDesc.debugLabel = "payload_layout_pass";

    auto passResult = builder.addGraphicsPass(passDesc);
    EXPECT_FALSE(passResult.hasError());
    if (passResult.hasError()) {
      return {};
    }
    EXPECT_FALSE(builder.markPassSideEffect(passResult.value()).hasError());
    return builder.computeGraphFingerprint();
  };

  const std::array<std::byte, 4u> pushBytesA = {
      std::byte{0x01}, std::byte{0x02}, std::byte{0x03}, std::byte{0x04}};
  const std::array<std::byte, 4u> pushBytesB = {
      std::byte{0x05}, std::byte{0x06}, std::byte{0x07}, std::byte{0x08}};
  const std::array<BufferHandle, 1u> dependencyBuffers = {
      BufferHandle{.index = 61u, .generation = 1u}};

  ComputeDispatchItem dispatchA{};
  dispatchA.pipeline = ComputePipelineHandle{.index = 77u, .generation = 1u};
  dispatchA.dispatch = {.x = 1u, .y = 1u, .z = 1u};
  dispatchA.pushConstants =
      std::span<const std::byte>(pushBytesA.data(), pushBytesA.size());
  dispatchA.dependencyBuffers = std::span<const BufferHandle>(
      dependencyBuffers.data(), dependencyBuffers.size());
  dispatchA.debugLabel = "payload_layout_dispatch_a";

  ComputeDispatchItem dispatchB = dispatchA;
  dispatchB.pushConstants =
      std::span<const std::byte>(pushBytesB.data(), pushBytesB.size());
  dispatchB.debugLabel = "payload_layout_dispatch_b";

  const std::array<ComputeDispatchItem, 1u> oneDispatch = {dispatchA};
  const std::array<ComputeDispatchItem, 2u> twoDispatches = {dispatchA,
                                                             dispatchB};

  const auto fingerprintA =
      recordFrame(240u, std::span<const ComputeDispatchItem>(
                            oneDispatch.data(), oneDispatch.size()));
  const std::array<ComputeDispatchItem, 1u> changedContent = {dispatchB};
  const auto sameLayoutFingerprint =
      recordFrame(241u, std::span<const ComputeDispatchItem>(
                            changedContent.data(), changedContent.size()));
  const auto fingerprintB =
      recordFrame(242u, std::span<const ComputeDispatchItem>(
                            twoDispatches.data(), twoDispatches.size()));

  EXPECT_TRUE(fingerprintA == sameLayoutFingerprint);
  EXPECT_FALSE(fingerprintA == fingerprintB);
}

TEST(RenderGraphCompileBehaviorTest,
     GraphFingerprintRejectsImportedTransientBufferSlotRewire) {
  RenderGraphBuilder builder;

  const auto recordFrame =
      [&](uint64_t frameIndex, bool importFirst,
          BufferHandle importedHandle) -> RenderGraphBuilder::GraphFingerprint {
    builder.beginFrame(frameIndex);

    Result<RenderGraphBufferId, std::string> importedResult =
        Result<RenderGraphBufferId, std::string>::makeError("not imported");
    Result<RenderGraphBufferId, std::string> transientResult =
        Result<RenderGraphBufferId, std::string>::makeError("not created");
    if (importFirst) {
      importedResult =
          builder.importBuffer(importedHandle, "fingerprint_imported_buffer");
      transientResult = builder.createTransientBuffer(
          makeTransientBufferDesc(64u), "fingerprint_transient_buffer");
    } else {
      transientResult = builder.createTransientBuffer(
          makeTransientBufferDesc(64u), "fingerprint_transient_buffer");
      importedResult =
          builder.importBuffer(importedHandle, "fingerprint_imported_buffer");
    }
    EXPECT_FALSE(importedResult.hasError());
    EXPECT_FALSE(transientResult.hasError());
    if (importedResult.hasError() || transientResult.hasError()) {
      return {};
    }

    auto passResult =
        addTestGraphicsPass(builder, makeTestPass("fingerprint_buffer_rewire"),
                            "fingerprint_buffer_rewire");
    EXPECT_FALSE(passResult.hasError());
    if (passResult.hasError()) {
      return {};
    }
    EXPECT_FALSE(builder
                     .addBufferAccess(passResult.value(),
                                      importedResult.value(),
                                      RenderGraphAccessMode::Read)
                     .hasError());
    EXPECT_FALSE(builder
                     .addBufferAccess(passResult.value(),
                                      transientResult.value(),
                                      RenderGraphAccessMode::Write)
                     .hasError());
    EXPECT_FALSE(builder.markPassSideEffect(passResult.value()).hasError());
    return builder.computeGraphFingerprint();
  };

  const auto importedFirstFingerprint =
      recordFrame(243u, true, BufferHandle{.index = 401u, .generation = 1u});
  const auto transientFirstFingerprint =
      recordFrame(244u, false, BufferHandle{.index = 402u, .generation = 2u});

  EXPECT_FALSE(importedFirstFingerprint == transientFirstFingerprint);
}

TEST(RenderGraphCompileBehaviorTest,
     RefreshHandlesUpdatesBorrowedPreResolvedDrawSpan) {
  RenderGraphBuilder builder;

  auto recordFrame = [&](uint64_t frameIndex, std::span<const DrawItem> draws)
      -> RenderGraphBuilder::GraphFingerprint {
    builder.beginFrame(frameIndex);
    auto colorResult = builder.createTransientTexture(
        makeTransientTextureDesc(Format::RGBA8_UNORM, 32u, 32u),
        "refresh_borrowed_preresolved_draw_color");
    EXPECT_FALSE(colorResult.hasError());
    if (colorResult.hasError()) {
      return {};
    }

    const std::array<BufferHandle, 1u> preResolvedDrawBuffers = {
        BufferHandle{.index = 150u, .generation = 1u}};

    RenderGraphGraphicsPassDesc passDesc{};
    passDesc.colorTexture = colorResult.value();
    passDesc.draws = draws;
    passDesc.drawBuffersPreResolved = true;
    passDesc.preResolvedDrawBuffers = std::span<const BufferHandle>(
        preResolvedDrawBuffers.data(), preResolvedDrawBuffers.size());
    passDesc.borrowPayload = true;
    passDesc.debugLabel = "refresh_borrowed_preresolved_draw_pass";

    auto passResult = builder.addGraphicsPass(passDesc);
    EXPECT_FALSE(passResult.hasError());
    if (passResult.hasError()) {
      return {};
    }
    EXPECT_FALSE(builder.markPassSideEffect(passResult.value()).hasError());
    return builder.computeGraphFingerprint();
  };

  std::array<DrawItem, 4u> draws{};
  for (uint32_t i = 0u; i < draws.size(); ++i) {
    draws[i].vertexBuffer = BufferHandle{.index = 160u + i, .generation = 1u};
  }

  const auto fingerprintA =
      recordFrame(268u, std::span<const DrawItem>(draws.data(), 1u));
  auto compileResult = compileBuilder(builder);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  RenderGraphCompileResult compiled = std::move(compileResult.value());
  ASSERT_EQ(compiled.orderedPasses.size(), 1u);
  ASSERT_EQ(compiled.orderedPasses[0u].draws.size(), 1u);

  const auto fingerprintB =
      recordFrame(269u, std::span<const DrawItem>(draws.data(), draws.size()));
  ASSERT_TRUE(fingerprintA == fingerprintB);

  builder.refreshHandlesInCompileResult(compiled);
  ASSERT_EQ(compiled.orderedPasses.size(), 1u);
  ASSERT_EQ(compiled.orderedPasses[0u].draws.size(), draws.size());
  EXPECT_EQ(compiled.orderedPasses[0u].draws[3u].vertexBuffer.index, 163u);
}

TEST(RenderGraphCompileBehaviorTest, TextureCopyPassCompilesNativePayload) {
  RenderGraphBuilder builder;
  builder.beginFrame(270u);

  const TextureHandle sourceTexture{.index = 301u, .generation = 1u};
  const TextureHandle destinationTexture{.index = 302u, .generation = 1u};
  auto sourceResult = builder.importTexture(sourceTexture, "copy_source");
  auto destinationResult =
      builder.importTexture(destinationTexture, "copy_destination");
  ASSERT_FALSE(sourceResult.hasError());
  ASSERT_FALSE(destinationResult.hasError());

  const std::array<RenderGraphTextureCopyItem, 1u> copies{{
      {.sourceTexture = sourceResult.value(),
       .destinationTexture = destinationResult.value(),
       .sourceX = 1u,
       .sourceY = 2u,
       .destinationX = 3u,
       .destinationY = 4u,
       .width = 8u,
       .height = 9u},
  }};
  RenderGraphTextureCopyPassDesc desc{};
  desc.copies =
      std::span<const RenderGraphTextureCopyItem>(copies.data(), copies.size());
  desc.debugLabel = "copy_pass";

  auto passResult = builder.addTextureCopyPass(desc);
  ASSERT_FALSE(passResult.hasError()) << passResult.error();

  auto compileResult = compileBuilder(builder);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  const RenderGraphCompileResult &compiled = compileResult.value();
  ASSERT_EQ(compiled.orderedPasses.size(), 1u);
  const RenderPass &pass = compiled.orderedPasses[0u];
  EXPECT_EQ(pass.executionMode, RenderPassExecutionMode::CopyOnly);
  EXPECT_EQ(pass.debugLabel, "copy_pass");
  ASSERT_EQ(pass.textureCopies.size(), 1u);
  EXPECT_TRUE(sameTexture(pass.textureCopies[0u].sourceTexture, sourceTexture));
  EXPECT_TRUE(sameTexture(pass.textureCopies[0u].destinationTexture,
                          destinationTexture));
  EXPECT_EQ(pass.textureCopies[0u].sourceX, 1u);
  EXPECT_EQ(pass.textureCopies[0u].destinationY, 4u);
  ASSERT_EQ(compiled.textureCopyRangesByPass.size(), 1u);
  EXPECT_EQ(compiled.textureCopyRangesByPass[0u].count, 1u);
  EXPECT_TRUE(compiled.unresolvedTextureCopyBindings.empty());
}

TEST(RenderGraphCompileBehaviorTest,
     RefreshHandlesUpdatesTextureCopyImportedHandles) {
  RenderGraphBuilder builder;

  auto recordFrame = [&](uint64_t frameIndex, TextureHandle sourceTexture,
                         TextureHandle destinationTexture) {
    builder.beginFrame(frameIndex);
    auto sourceResult = builder.importTexture(sourceTexture, "copy_source");
    auto destinationResult =
        builder.importTexture(destinationTexture, "copy_destination");
    EXPECT_FALSE(sourceResult.hasError());
    EXPECT_FALSE(destinationResult.hasError());
    if (sourceResult.hasError() || destinationResult.hasError()) {
      return RenderGraphBuilder::GraphFingerprint{};
    }

    const std::array<RenderGraphTextureCopyItem, 1u> copies{{
        {.sourceTexture = sourceResult.value(),
         .destinationTexture = destinationResult.value(),
         .sourceX = 2u,
         .sourceY = 3u,
         .destinationX = 4u,
         .destinationY = 5u,
         .width = 6u,
         .height = 7u},
    }};
    RenderGraphTextureCopyPassDesc desc{};
    desc.copies = std::span<const RenderGraphTextureCopyItem>(copies.data(),
                                                              copies.size());
    desc.debugLabel = "refresh_copy_pass";
    auto passResult = builder.addTextureCopyPass(desc);
    EXPECT_FALSE(passResult.hasError());
    return builder.computeGraphFingerprint();
  };

  const TextureHandle sourceA{.index = 311u, .generation = 1u};
  const TextureHandle destinationA{.index = 312u, .generation = 1u};
  const TextureHandle sourceB{.index = 313u, .generation = 2u};
  const TextureHandle destinationB{.index = 314u, .generation = 2u};

  const auto fingerprintA = recordFrame(271u, sourceA, destinationA);
  auto compileResult = compileBuilder(builder);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  RenderGraphCompileResult compiled = std::move(compileResult.value());
  ASSERT_EQ(compiled.orderedPasses.size(), 1u);
  ASSERT_EQ(compiled.orderedPasses[0u].textureCopies.size(), 1u);
  EXPECT_TRUE(sameTexture(
      compiled.orderedPasses[0u].textureCopies[0u].sourceTexture, sourceA));

  const auto fingerprintB = recordFrame(272u, sourceB, destinationB);
  ASSERT_TRUE(fingerprintA == fingerprintB);
  builder.refreshHandlesInCompileResult(compiled);

  ASSERT_EQ(compiled.orderedPasses.size(), 1u);
  ASSERT_EQ(compiled.orderedPasses[0u].textureCopies.size(), 1u);
  EXPECT_TRUE(sameTexture(
      compiled.orderedPasses[0u].textureCopies[0u].sourceTexture, sourceB));
  EXPECT_TRUE(sameTexture(
      compiled.orderedPasses[0u].textureCopies[0u].destinationTexture,
      destinationB));
}

TEST(RenderGraphCompileBehaviorTest,
     ComputeOnlyPassCompilesWithTextureReadAndPreDispatchBufferWrite) {
  RenderGraphBuilder builder;
  builder.beginFrame(244u);

  auto sourceTextureResult = builder.createTransientTexture(
      makeTransientTextureDesc(Format::RG32_FLOAT, 1u, 1u),
      "compute_only_source");
  auto resultBufferResult = builder.createTransientBuffer(
      makeTransientBufferDesc(32u), "compute_only_result");
  ASSERT_FALSE(sourceTextureResult.hasError());
  ASSERT_FALSE(resultBufferResult.hasError());

  const std::array<BufferHandle, 1u> dispatchDependencies = {BufferHandle{}};
  ComputeDispatchItem dispatch{};
  dispatch.pipeline = ComputePipelineHandle{.index = 88u, .generation = 1u};
  dispatch.dispatch = {.x = 1u, .y = 1u, .z = 1u};
  dispatch.dependencyBuffers = std::span<const BufferHandle>(
      dispatchDependencies.data(), dispatchDependencies.size());
  dispatch.debugLabel = "compute_only_dispatch";
  const std::array<ComputeDispatchItem, 1u> preDispatches = {dispatch};

  RenderGraphGraphicsPassDesc passDesc{};
  passDesc.executionMode = RenderPassExecutionMode::ComputeOnly;
  passDesc.hasColorAttachment = false;
  passDesc.preDispatches = std::span<const ComputeDispatchItem>(
      preDispatches.data(), preDispatches.size());
  passDesc.debugLabel = "compute_only_pass";
  passDesc.markImplicitOutputSideEffect = true;

  auto passResult = builder.addGraphicsPass(passDesc);
  ASSERT_FALSE(passResult.hasError()) << passResult.error();
  ASSERT_FALSE(
      builder.addTextureRead(passResult.value(), sourceTextureResult.value())
          .hasError());
  ASSERT_FALSE(builder
                   .bindPreDispatchDependencyBuffer(
                       passResult.value(), 0u, 0u, resultBufferResult.value(),
                       RenderGraphAccessMode::Write)
                   .hasError());

  auto compileResult = compileBuilder(builder);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  ASSERT_EQ(compileResult.value().orderedPasses.size(), 1u);
  EXPECT_EQ(compileResult.value().orderedPasses[0].executionMode,
            RenderPassExecutionMode::ComputeOnly);
  ASSERT_EQ(compileResult.value().orderedPasses[0].preDispatches.size(), 1u);
  ASSERT_EQ(compileResult.value()
                .orderedPasses[0]
                .preDispatches[0]
                .dependencyBuffers.size(),
            1u);
}

TEST(RenderGraphCompileBehaviorTest,
     ImplicitPreDispatchDependenciesPreserveExactAccessModes) {
  RenderGraphBuilder builder;
  builder.beginFrame(245u);

  const std::array<BufferHandle, 2u> dependencies = {
      BufferHandle{.index = 501u, .generation = 1u},
      BufferHandle{.index = 502u, .generation = 1u},
  };
  const std::array<RenderGraphAccessMode, 2u> accessModes = {
      RenderGraphAccessMode::Read,
      RenderGraphAccessMode::Write,
  };
  ComputeDispatchItem dispatch{};
  dispatch.pipeline = ComputePipelineHandle{.index = 91u, .generation = 1u};
  dispatch.dispatch = {.x = 1u, .y = 1u, .z = 1u};
  dispatch.dependencyBuffers = dependencies;
  dispatch.dependencyBufferAccessModes = accessModes;
  dispatch.debugLabel = "exact_access_dispatch";
  const std::array<ComputeDispatchItem, 1u> preDispatches = {dispatch};

  RenderGraphGraphicsPassDesc passDesc{};
  passDesc.executionMode = RenderPassExecutionMode::ComputeOnly;
  passDesc.hasColorAttachment = false;
  passDesc.preDispatches = preDispatches;
  passDesc.debugLabel = "exact_access_pass";
  passDesc.markImplicitOutputSideEffect = true;

  auto passResult = builder.addGraphicsPass(passDesc);
  ASSERT_FALSE(passResult.hasError()) << passResult.error();

  auto compileResult = compileBuilder(builder);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  const RenderGraphCompileResult &compiled = compileResult.value();
  ASSERT_EQ(compiled.passBarrierRecords.size(), 2u);
  EXPECT_EQ(compiled.passBarrierRecords[0].afterAccess,
            RenderGraphAccessMode::Read);
  EXPECT_EQ(compiled.passBarrierRecords[1].afterAccess,
            RenderGraphAccessMode::Write);
}

TEST(RenderGraphCompileBehaviorTest,
     ComputeOnlyPassRejectsAttachmentsDrawsAndEmptyDispatches) {
  {
    RenderGraphBuilder builder;
    builder.beginFrame(245u);
    auto colorResult = builder.createTransientTexture(
        makeTransientTextureDesc(Format::RGBA8_UNORM, 16u, 16u),
        "compute_only_invalid_color");
    ASSERT_FALSE(colorResult.hasError());

    RenderGraphGraphicsPassDesc passDesc{};
    passDesc.executionMode = RenderPassExecutionMode::ComputeOnly;
    passDesc.colorTexture = colorResult.value();
    passDesc.debugLabel = "compute_only_invalid_color";
    auto passResult = builder.addGraphicsPass(passDesc);
    ASSERT_TRUE(passResult.hasError());
    EXPECT_NE(passResult.error().find("color"), std::string::npos);
  }

  {
    RenderGraphBuilder builder;
    builder.beginFrame(246u);
    DrawItem draw{};
    const std::array<DrawItem, 1u> draws = {draw};
    ComputeDispatchItem dispatch{};
    dispatch.pipeline = ComputePipelineHandle{.index = 89u, .generation = 1u};
    dispatch.dispatch = {.x = 1u, .y = 1u, .z = 1u};
    const std::array<ComputeDispatchItem, 1u> preDispatches = {dispatch};

    RenderGraphGraphicsPassDesc passDesc{};
    passDesc.executionMode = RenderPassExecutionMode::ComputeOnly;
    passDesc.hasColorAttachment = false;
    passDesc.preDispatches = std::span<const ComputeDispatchItem>(
        preDispatches.data(), preDispatches.size());
    passDesc.draws = std::span<const DrawItem>(draws.data(), draws.size());
    passDesc.debugLabel = "compute_only_invalid_draws";
    auto passResult = builder.addGraphicsPass(passDesc);
    ASSERT_TRUE(passResult.hasError());
    EXPECT_NE(passResult.error().find("draws"), std::string::npos);
  }

  {
    RenderGraphBuilder builder;
    builder.beginFrame(247u);
    RenderGraphGraphicsPassDesc passDesc{};
    passDesc.executionMode = RenderPassExecutionMode::ComputeOnly;
    passDesc.hasColorAttachment = false;
    passDesc.debugLabel = "compute_only_empty";
    auto passResult = builder.addGraphicsPass(passDesc);
    ASSERT_TRUE(passResult.hasError());
    EXPECT_NE(passResult.error().find("dispatch"), std::string::npos);
  }
}

TEST(RenderGraphCompileBehaviorTest,
     ExplicitAndHazardDependencyOverlapDeduplicatesToSingleEdge) {
  RenderGraphBuilder builder;
  builder.beginFrame(217u);

  auto pass0Result = addTestGraphicsPass(
      builder, makeTestPass("dedup_overlap_p0"), "dedup_overlap_p0");
  auto pass1Result = addTestGraphicsPass(
      builder, makeTestPass("dedup_overlap_p1"), "dedup_overlap_p1");
  if (!(!pass0Result.hasError() && !pass1Result.hasError())) {
    ADD_FAILURE() << "addLegacyRenderPass should succeed for overlap dedup "
                     "graph";
    return;
  }

  auto transientBufferResult = builder.createTransientBuffer(
      makeTransientBufferDesc(64u), "dedup_overlap_buffer");
  if (!(!transientBufferResult.hasError())) {
    ADD_FAILURE() << "createTransientBuffer should succeed for overlap dedup "
                     "graph";
    return;
  }

  auto accessResult = builder.addBufferWrite(pass0Result.value(),
                                             transientBufferResult.value());
  if (!(!accessResult.hasError())) {
    ADD_FAILURE() << "addBufferWrite should succeed for overlap dedup graph";
    return;
  }
  accessResult =
      builder.addBufferRead(pass1Result.value(), transientBufferResult.value());
  if (!(!accessResult.hasError())) {
    ADD_FAILURE() << "addBufferRead should succeed for overlap dedup graph";
    return;
  }

  auto depResult =
      builder.addDependency(pass0Result.value(), pass1Result.value());
  if (!(!depResult.hasError())) {
    ADD_FAILURE() << "explicit addDependency should succeed for overlap dedup "
                     "graph";
    return;
  }

  auto compileResult = compileBuilder(builder);
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed for overlap dedup graph";
    if (compileResult.hasError()) {
      std::cerr << compileResult.error() << "\n";
    }
    return;
  }
  const RenderGraphCompileResult &compiled = compileResult.value();

  if (!(compiled.edges.size() == 1u)) {
    ADD_FAILURE()
        << "overlapping explicit+hazard dependency should collapse to "
           "one edge";
    return;
  }
  if (!(compiled.edges[0u].before == pass0Result.value().value &&
        compiled.edges[0u].after == pass1Result.value().value)) {
    ADD_FAILURE()
        << "deduped overlap edge should preserve dependency direction";
    return;
  }
}

TEST(RenderGraphCompileBehaviorTest,
     BeginFrameResetsDependencyAndFrameOutputDedupState) {
  RenderGraphBuilder builder;
  builder.beginFrame(218u);
  const TextureHandle outputTexture{.index = 901u, .generation = 7u};

  auto frameAPass0 = addTestGraphicsPass(
      builder, makeTestPass("reset_a_p0", outputTexture), "reset_a_p0");
  auto frameAPass1 =
      addTestGraphicsPass(builder, makeTestPass("reset_a_p1"), "reset_a_p1");
  if (!(!frameAPass0.hasError() && !frameAPass1.hasError())) {
    ADD_FAILURE() << "frame A addLegacyRenderPass should succeed";
    return;
  }
  auto sideEffectResult = builder.markPassSideEffect(frameAPass1.value());
  if (!(!sideEffectResult.hasError())) {
    ADD_FAILURE() << "frame A markPassSideEffect should succeed";
    return;
  }

  auto depResult =
      builder.addDependency(frameAPass0.value(), frameAPass1.value());
  if (!(!depResult.hasError())) {
    ADD_FAILURE() << "frame A addDependency should succeed";
    return;
  }
  depResult = builder.addDependency(frameAPass0.value(), frameAPass1.value());
  if (!(!depResult.hasError())) {
    ADD_FAILURE() << "frame A duplicate addDependency should succeed";
    return;
  }

  auto outputImportA = builder.importTexture(outputTexture, "reset_a_out_tex");
  if (!(!outputImportA.hasError())) {
    ADD_FAILURE() << "frame A importTexture should succeed";
    return;
  }
  auto outputMarkResult =
      builder.markTextureAsFrameOutput(outputImportA.value());
  if (!(!outputMarkResult.hasError())) {
    ADD_FAILURE() << "frame A markTextureAsFrameOutput should succeed";
    return;
  }
  outputMarkResult = builder.markTextureAsFrameOutput(outputImportA.value());
  if (!(!outputMarkResult.hasError())) {
    ADD_FAILURE() << "frame A duplicate markTextureAsFrameOutput should "
                     "succeed";
    return;
  }

  auto compileA = compileBuilder(builder);
  if (!(!compileA.hasError())) {
    ADD_FAILURE() << "frame A compile should succeed";
    if (compileA.hasError()) {
      std::cerr << compileA.error() << "\n";
    }
    return;
  }

  builder.beginFrame(219u);

  auto frameBPass0 = addTestGraphicsPass(
      builder, makeTestPass("reset_b_p0", outputTexture), "reset_b_p0");
  auto frameBPass1 =
      addTestGraphicsPass(builder, makeTestPass("reset_b_p1"), "reset_b_p1");
  if (!(!frameBPass0.hasError() && !frameBPass1.hasError())) {
    ADD_FAILURE() << "frame B addLegacyRenderPass should succeed";
    return;
  }
  sideEffectResult = builder.markPassSideEffect(frameBPass1.value());
  if (!(!sideEffectResult.hasError())) {
    ADD_FAILURE() << "frame B markPassSideEffect should succeed";
    return;
  }

  depResult = builder.addDependency(frameBPass0.value(), frameBPass1.value());
  if (!(!depResult.hasError())) {
    ADD_FAILURE() << "frame B addDependency should succeed after beginFrame";
    return;
  }

  auto outputImportB = builder.importTexture(outputTexture, "reset_b_out_tex");
  if (!(!outputImportB.hasError())) {
    ADD_FAILURE() << "frame B importTexture should succeed";
    return;
  }
  outputMarkResult = builder.markTextureAsFrameOutput(outputImportB.value());
  if (!(!outputMarkResult.hasError())) {
    ADD_FAILURE() << "frame B markTextureAsFrameOutput should succeed after "
                     "beginFrame";
    return;
  }

  auto compileB = compileBuilder(builder);
  if (!(!compileB.hasError())) {
    ADD_FAILURE() << "frame B compile should succeed";
    if (compileB.hasError()) {
      std::cerr << compileB.error() << "\n";
    }
    return;
  }
  const RenderGraphCompileResult &compiledB = compileB.value();
  if (!(compiledB.edges.size() == 1u)) {
    ADD_FAILURE() << "frame B explicit dependency should be present after "
                     "beginFrame reset";
    return;
  }
  if (!(compiledB.rootPassCount >= 1u)) {
    ADD_FAILURE()
        << "frame B explicit frame output root should be present after "
           "beginFrame reset";
    return;
  }
}

TEST(RenderGraphCompileBehaviorTest, DeadPassCullingFromFrameOutputRoots) {
  RenderGraphBuilder builder;
  builder.beginFrame(202u);

  const TextureHandle texA{.index = 1u, .generation = 1u};
  const TextureHandle texB{.index = 2u, .generation = 1u};
  const TextureHandle texDead{.index = 3u, .generation = 1u};

  auto pass0Result =
      addTestGraphicsPass(builder, makeTestPass("cull_p0", texA), "cull_p0");
  auto pass1Result =
      addTestGraphicsPass(builder, makeTestPass("cull_p1", texB), "cull_p1");
  auto pass2Result = addTestGraphicsPass(
      builder, makeTestPass("cull_dead", texDead), "cull_dead");
  if (!(!pass0Result.hasError() && !pass1Result.hasError() &&
        !pass2Result.hasError())) {
    ADD_FAILURE() << "addLegacyRenderPass should succeed for culling graph";
    return;
  }

  auto sharedBufferResult = builder.createTransientBuffer(
      makeTransientBufferDesc(64u), "cull_shared");
  if (!(!sharedBufferResult.hasError())) {
    ADD_FAILURE() << "createTransientBuffer should succeed for culling graph";
    return;
  }

  auto writeResult =
      builder.addBufferWrite(pass0Result.value(), sharedBufferResult.value());
  if (!(!writeResult.hasError())) {
    ADD_FAILURE() << "addBufferWrite should succeed for culling graph";
    return;
  }
  auto readResult =
      builder.addBufferRead(pass1Result.value(), sharedBufferResult.value());
  if (!(!readResult.hasError())) {
    ADD_FAILURE() << "addBufferRead should succeed for culling graph";
    return;
  }

  auto outputImportResult = builder.importTexture(texB, "cull_frame_output");
  if (!(!outputImportResult.hasError())) {
    ADD_FAILURE() << "importTexture should succeed for culling graph";
    return;
  }
  auto outputMarkResult =
      builder.markTextureAsFrameOutput(outputImportResult.value());
  if (!(!outputMarkResult.hasError())) {
    ADD_FAILURE() << "markTextureAsFrameOutput should succeed";
    return;
  }

  auto compileResult = compileBuilder(builder);
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed for culling graph";
    if (compileResult.hasError()) {
      std::cerr << compileResult.error() << "\n";
    }
    return;
  }
  const RenderGraphCompileResult &compiled = compileResult.value();

  if (!(compiled.declaredPassCount == 3u)) {
    ADD_FAILURE() << "culling graph should declare 3 passes";
    return;
  }
  if (!(compiled.culledPassCount == 1u)) {
    ADD_FAILURE() << "exactly one dead pass should be culled";
    return;
  }
  if (!(compiled.rootPassCount == 1u)) {
    ADD_FAILURE() << "frame-output culling graph should have one root writer";
    return;
  }
  if (!(compiled.orderedPassIndices.size() == 2u)) {
    ADD_FAILURE() << "only root-reachable passes should remain after culling";
    return;
  }
  if (!(compiled.orderedPassIndices[0u] == pass0Result.value().value &&
        compiled.orderedPassIndices[1u] == pass1Result.value().value)) {
    ADD_FAILURE() << "culling should preserve producer->output order";
    return;
  }
  if (!(std::find(compiled.orderedPassIndices.begin(),
                  compiled.orderedPassIndices.end(),
                  pass2Result.value().value) ==
        compiled.orderedPassIndices.end())) {
    ADD_FAILURE() << "culled pass index should not be scheduled";
    return;
  }
}

TEST(RenderGraphCompileBehaviorTest, CycleDiagnosticsIncludePassNames) {
  RenderGraphBuilder builder;
  builder.beginFrame(203u);

  const TextureHandle texA{.index = 11u, .generation = 1u};
  const TextureHandle texB{.index = 12u, .generation = 1u};

  auto pass0Result =
      addTestGraphicsPass(builder, makeTestPass("cycle_a", texA), "cycle_a");
  auto pass1Result =
      addTestGraphicsPass(builder, makeTestPass("cycle_b", texB), "cycle_b");
  if (!(!pass0Result.hasError() && !pass1Result.hasError())) {
    ADD_FAILURE() << "addLegacyRenderPass should succeed for cycle graph";
    return;
  }

  auto depResult =
      builder.addDependency(pass0Result.value(), pass1Result.value());
  if (!(!depResult.hasError())) {
    ADD_FAILURE() << "addDependency cycle edge A->B";
    return;
  }
  depResult = builder.addDependency(pass1Result.value(), pass0Result.value());
  if (!(!depResult.hasError())) {
    ADD_FAILURE() << "addDependency cycle edge B->A";
    return;
  }

  auto compileResult = compileBuilder(builder);
  if (!(compileResult.hasError())) {
    ADD_FAILURE() << "compile should fail on dependency cycle";
    return;
  }

  const std::string_view error = compileResult.error();
  if (((error).find("dependency cycle detected") == std::string_view::npos)) {
    ADD_FAILURE() << "cycle error should mention cycle detection";
    return;
  }
  if (((error).find("cycle_a") == std::string_view::npos)) {
    ADD_FAILURE() << "cycle error should include first pass debug name";
    return;
  }
  if (((error).find("cycle_b") == std::string_view::npos)) {
    ADD_FAILURE() << "cycle error should include second pass debug name";
    return;
  }
}

TEST(RenderGraphCompileBehaviorTest, TransientAliasAllocationCorrectness) {
  RenderGraphBuilder builder;
  builder.beginFrame(204u);

  auto pass0Result =
      addTestGraphicsPass(builder, makeTestPass("alias_p0"), "alias_p0");
  auto pass1Result =
      addTestGraphicsPass(builder, makeTestPass("alias_p1"), "alias_p1");
  auto pass2Result =
      addTestGraphicsPass(builder, makeTestPass("alias_p2"), "alias_p2");
  auto pass3Result =
      addTestGraphicsPass(builder, makeTestPass("alias_p3"), "alias_p3");
  if (!(!pass0Result.hasError() && !pass1Result.hasError() &&
        !pass2Result.hasError() && !pass3Result.hasError())) {
    ADD_FAILURE() << "addLegacyRenderPass should succeed for aliasing graph";
    return;
  }

  auto depResult =
      builder.addDependency(pass0Result.value(), pass1Result.value());
  if (!(!depResult.hasError())) {
    ADD_FAILURE() << "addDependency p0->p1 should succeed";
    return;
  }
  depResult = builder.addDependency(pass1Result.value(), pass2Result.value());
  if (!(!depResult.hasError())) {
    ADD_FAILURE() << "addDependency p1->p2 should succeed";
    return;
  }
  depResult = builder.addDependency(pass2Result.value(), pass3Result.value());
  if (!(!depResult.hasError())) {
    ADD_FAILURE() << "addDependency p2->p3 should succeed";
    return;
  }

  auto bufferAResult = builder.createTransientBuffer(
      makeTransientBufferDesc(64u), "alias_buf_a");
  auto bufferBResult = builder.createTransientBuffer(
      makeTransientBufferDesc(64u), "alias_buf_b");
  auto bufferCResult = builder.createTransientBuffer(
      makeTransientBufferDesc(128u), "alias_buf_c");
  if (!(!bufferAResult.hasError() && !bufferBResult.hasError() &&
        !bufferCResult.hasError())) {
    ADD_FAILURE() << "createTransientBuffer should succeed for aliasing graph";
    return;
  }

  auto accessResult =
      builder.addBufferWrite(pass0Result.value(), bufferAResult.value());
  if (!(!accessResult.hasError())) {
    ADD_FAILURE() << "addBufferWrite for buffer A";
    return;
  }
  accessResult =
      builder.addBufferWrite(pass2Result.value(), bufferBResult.value());
  if (!(!accessResult.hasError())) {
    ADD_FAILURE() << "addBufferWrite for buffer B";
    return;
  }
  accessResult =
      builder.addBufferWrite(pass1Result.value(), bufferCResult.value());
  if (!(!accessResult.hasError())) {
    ADD_FAILURE() << "addBufferWrite for buffer C";
    return;
  }
  accessResult =
      builder.addBufferRead(pass3Result.value(), bufferCResult.value());
  if (!(!accessResult.hasError())) {
    ADD_FAILURE() << "addBufferRead for buffer C";
    return;
  }

  auto textureAResult = builder.createTransientTexture(
      makeTransientTextureDesc(Format::RGBA8_UNORM, 16u, 16u), "alias_tex_a");
  auto textureBResult = builder.createTransientTexture(
      makeTransientTextureDesc(Format::RGBA8_UNORM, 16u, 16u), "alias_tex_b");
  auto textureCResult = builder.createTransientTexture(
      makeTransientTextureDesc(Format::RGBA16_FLOAT, 16u, 16u), "alias_tex_c");
  if (!(!textureAResult.hasError() && !textureBResult.hasError() &&
        !textureCResult.hasError())) {
    ADD_FAILURE() << "createTransientTexture should succeed for aliasing graph";
    return;
  }

  accessResult =
      builder.addTextureWrite(pass0Result.value(), textureAResult.value());
  if (!(!accessResult.hasError())) {
    ADD_FAILURE() << "addTextureWrite for texture A";
    return;
  }
  accessResult =
      builder.addTextureWrite(pass2Result.value(), textureBResult.value());
  if (!(!accessResult.hasError())) {
    ADD_FAILURE() << "addTextureWrite for texture B";
    return;
  }
  accessResult =
      builder.addTextureWrite(pass1Result.value(), textureCResult.value());
  if (!(!accessResult.hasError())) {
    ADD_FAILURE() << "addTextureWrite for texture C";
    return;
  }
  accessResult =
      builder.addTextureRead(pass3Result.value(), textureCResult.value());
  if (!(!accessResult.hasError())) {
    ADD_FAILURE() << "addTextureRead for texture C";
    return;
  }

  auto compileResult = compileBuilder(builder);
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed for aliasing graph";
    if (compileResult.hasError()) {
      std::cerr << compileResult.error() << "\n";
    }
    return;
  }
  const RenderGraphCompileResult &compiled = compileResult.value();

  if (!(bufferAResult.value().value <
            compiled.transientBufferAllocationByResource.size() &&
        bufferBResult.value().value <
            compiled.transientBufferAllocationByResource.size() &&
        bufferCResult.value().value <
            compiled.transientBufferAllocationByResource.size())) {
    ADD_FAILURE()
        << "buffer allocation map should contain all transient resources";
    return;
  }
  const uint32_t bufferAllocA =
      compiled.transientBufferAllocationByResource[bufferAResult.value().value];
  const uint32_t bufferAllocB =
      compiled.transientBufferAllocationByResource[bufferBResult.value().value];
  const uint32_t bufferAllocC =
      compiled.transientBufferAllocationByResource[bufferCResult.value().value];
  if (!(compiled.transientBufferPhysicalCount == 2u)) {
    ADD_FAILURE() << "buffer aliasing should collapse 3 logical buffers to 2 "
                     "physical allocations";
    std::cerr << "[INFO] transientBufferPhysicalCount="
              << compiled.transientBufferPhysicalCount << "\n";
    return;
  }
  if (!(compiled.transientTexturePhysicalCount == 2u)) {
    ADD_FAILURE() << "texture aliasing should collapse 3 logical textures to 2 "
                     "physical allocations";
    std::cerr << "[INFO] transientTexturePhysicalCount="
              << compiled.transientTexturePhysicalCount << "\n";
    return;
  }
  if (!(bufferAllocA != UINT32_MAX && bufferAllocB != UINT32_MAX &&
        bufferAllocC != UINT32_MAX)) {
    ADD_FAILURE() << "buffer allocation map entries should be resolved";
    return;
  }
  if (!(bufferAllocA == bufferAllocB)) {
    ADD_FAILURE()
        << "compatible non-overlapping buffers should alias the same slot";
    std::cerr << "[INFO] buffer_alloc_a=" << bufferAllocA
              << " buffer_alloc_b=" << bufferAllocB
              << " buffer_alloc_c=" << bufferAllocC
              << " physical_count=" << compiled.transientBufferPhysicalCount
              << "\n";
    std::cerr << "[INFO] ordered_pass_indices:";
    for (const uint32_t passIndex : compiled.orderedPassIndices) {
      std::cerr << " " << passIndex;
    }
    std::cerr << "\n";
    std::cerr << "[INFO] buffer_lifetimes:";
    for (const auto &lifetime : compiled.transientBufferLifetimes) {
      std::cerr << " [res=" << lifetime.resourceIndex
                << " first=" << lifetime.firstExecutionIndex
                << " last=" << lifetime.lastExecutionIndex << "]";
    }
    std::cerr << "\n";
    return;
  }
  if (!(bufferAllocC != bufferAllocA)) {
    ADD_FAILURE()
        << "incompatible buffer descriptor should use a different slot";
    return;
  }

  if (!(textureAResult.value().value <
            compiled.transientTextureAllocationByResource.size() &&
        textureBResult.value().value <
            compiled.transientTextureAllocationByResource.size() &&
        textureCResult.value().value <
            compiled.transientTextureAllocationByResource.size())) {
    ADD_FAILURE()
        << "texture allocation map should contain all transient resources";
    return;
  }
  const uint32_t textureAllocA =
      compiled
          .transientTextureAllocationByResource[textureAResult.value().value];
  const uint32_t textureAllocB =
      compiled
          .transientTextureAllocationByResource[textureBResult.value().value];
  const uint32_t textureAllocC =
      compiled
          .transientTextureAllocationByResource[textureCResult.value().value];
  if (!(textureAllocA != UINT32_MAX && textureAllocB != UINT32_MAX &&
        textureAllocC != UINT32_MAX)) {
    ADD_FAILURE() << "texture allocation map entries should be resolved";
    return;
  }
  if (!(textureAllocA == textureAllocB)) {
    ADD_FAILURE()
        << "compatible non-overlapping textures should alias the same slot";
    return;
  }
  if (!(textureAllocC != textureAllocA)) {
    ADD_FAILURE()
        << "incompatible texture descriptor should use a different slot";
    return;
  }
}

TEST(RenderGraphCompileBehaviorTest, ExplicitAccessOverridesLegacyInference) {
  RenderGraphBuilder builder;
  builder.beginFrame(205u);

  const BufferHandle sharedDependency{.index = 31u, .generation = 1u};
  const TextureHandle color0{.index = 41u, .generation = 1u};
  const TextureHandle color1{.index = 42u, .generation = 1u};
  std::array<BufferHandle, 1> deps0 = {BufferHandle{}};
  std::array<BufferHandle, 1> deps1 = {BufferHandle{}};

  RenderPass pass0 = makeTestPass("override_p0", color0);
  pass0.dependencyBuffers =
      std::span<const BufferHandle>(deps0.data(), deps0.size());
  RenderPass pass1 = makeTestPass("override_p1", color1);
  pass1.dependencyBuffers =
      std::span<const BufferHandle>(deps1.data(), deps1.size());

  auto pass0Result = addTestGraphicsPass(builder, pass0, pass0.debugLabel);
  auto pass1Result = addTestGraphicsPass(builder, pass1, pass1.debugLabel);
  if (!(!pass0Result.hasError() && !pass1Result.hasError())) {
    ADD_FAILURE()
        << "addLegacyRenderPass should succeed for access override graph";
    return;
  }

  auto importResult =
      builder.importBuffer(sharedDependency, "override_shared_dep");
  if (!(!importResult.hasError())) {
    ADD_FAILURE() << "importBuffer should succeed for access override graph";
    return;
  }
  auto bindResult = builder.bindPassDependencyBuffer(
      pass0Result.value(), 0u, importResult.value(),
      RenderGraphAccessMode::Read);
  if (!(!bindResult.hasError())) {
    ADD_FAILURE()
        << "bindPassDependencyBuffer pass0 read override should succeed";
    return;
  }
  bindResult = builder.bindPassDependencyBuffer(pass1Result.value(), 0u,
                                                importResult.value(),
                                                RenderGraphAccessMode::Read);
  if (!(!bindResult.hasError())) {
    ADD_FAILURE()
        << "bindPassDependencyBuffer pass1 read override should succeed";
    return;
  }

  auto compileResult = compileBuilder(builder);
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed for access override graph";
    if (compileResult.hasError()) {
      std::cerr << compileResult.error() << "\n";
    }
    return;
  }
  const RenderGraphCompileResult &compiled = compileResult.value();

  if (!(compiled.orderedPassIndices.size() == 2u)) {
    ADD_FAILURE() << "access override graph should schedule two passes";
    return;
  }
  if (!(compiled.edges.empty())) {
    ADD_FAILURE()
        << "explicit read overrides should prevent inferred RW hazard edge";
    std::cerr << "[INFO] edge_count=" << compiled.edges.size() << "\n";
    for (const auto &edge : compiled.edges) {
      std::cerr << "[INFO] edge " << edge.before << " -> " << edge.after
                << "\n";
    }
    return;
  }
}

TEST(RenderGraphCompileBehaviorTest,
     NoColorGraphicsPassCanWriteD16AttachmentSampledDepthOnly) {
  RenderGraphBuilder builder;
  builder.beginFrame(209u);

  TextureDesc depthDesc = makeTransientTextureDesc(Format::D16_UNORM, 16u, 16u);
  depthDesc.usage = TextureUsage::AttachmentSampled;
  auto depthResult =
      builder.createTransientTexture(depthDesc, "d16_no_color_depth");
  ASSERT_FALSE(depthResult.hasError());

  RenderGraphGraphicsPassDesc desc{};
  desc.hasColorAttachment = false;
  desc.depth = {.loadOp = LoadOp::Clear,
                .storeOp = StoreOp::Store,
                .clearDepth = 1.0f,
                .clearStencil = 0u};
  desc.depthTexture = depthResult.value();
  desc.debugLabel = "d16_no_color_depth_pass";

  auto passResult = builder.addGraphicsPass(desc);
  ASSERT_FALSE(passResult.hasError()) << passResult.error();
  ASSERT_FALSE(builder.markPassSideEffect(passResult.value()).hasError());

  auto compileResult = compileBuilder(builder);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  const RenderGraphCompileResult &compiled = compileResult.value();
  ASSERT_EQ(compiled.orderedPasses.size(), 1u);
  EXPECT_FALSE(compiled.orderedPasses.front().hasColorAttachment);
  EXPECT_EQ(compiled.orderedPasses.front().depth.clearStencil, 0u);
  ASSERT_EQ(compiled.transientTexturePhysicalAllocations.size(), 1u);
  const TextureDesc &compiledDepthDesc =
      compiled.transientTexturePhysicalAllocations.front().desc;
  EXPECT_EQ(compiledDepthDesc.format, Format::D16_UNORM);
  EXPECT_EQ(compiledDepthDesc.usage, TextureUsage::AttachmentSampled);
}

TEST(RenderGraphCompileBehaviorTest,
     PassLevelTextureDependenciesCanExceedSubmitDependencyLimit) {
  RenderGraphBuilder builder;
  builder.beginFrame(209u);

  auto colorResult = builder.createTransientTexture(
      makeTransientTextureDesc(Format::RGBA8_UNORM, 16u, 16u),
      "many_texture_reads_color");
  ASSERT_FALSE(colorResult.hasError()) << colorResult.error();

  std::array<TextureHandle, kMaxDependencyResources + 4u> dependencyTextures{};
  for (size_t i = 0; i < dependencyTextures.size(); ++i) {
    dependencyTextures[i] = TextureHandle{
        .index = static_cast<uint32_t>(700u + i),
        .generation = 1u,
    };
  }

  RenderGraphGraphicsPassDesc desc{};
  desc.color = {.loadOp = LoadOp::Clear,
                .storeOp = StoreOp::Store,
                .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}};
  desc.colorTexture = colorResult.value();
  desc.dependencyTextures = std::span<const TextureHandle>(
      dependencyTextures.data(), dependencyTextures.size());
  desc.debugLabel = "many_texture_reads";

  auto passResult = builder.addGraphicsPass(desc);
  ASSERT_FALSE(passResult.hasError()) << passResult.error();

  auto compileResult = compileBuilder(builder);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  EXPECT_EQ(compileResult.value().recordedGraphicsPasses.size(), 1u);
}

} // namespace
