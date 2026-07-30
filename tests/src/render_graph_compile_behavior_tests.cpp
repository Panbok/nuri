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

Result<CompiledRenderGraph, std::string>
compileBuilder(RenderGraphBuilder &builder) {
  RenderGraphRuntime runtime;
  return builder.compile(runtime);
}

Result<CompiledRenderGraph, std::string>
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

  const CompiledRenderGraph &compiledA = compileAResult.value();
  const CompiledRenderGraph &compiledB = compileBResult.value();

  if (!(compiledA.plan.orderedPassIndices.size() == 3u)) {
    ADD_FAILURE() << "determinism graph should schedule 3 passes";
    return;
  }
  if (!(compiledA.plan.orderedPassIndices[0u] == 0u &&
        compiledA.plan.orderedPassIndices[1u] == 1u &&
        compiledA.plan.orderedPassIndices[2u] == 2u)) {
    ADD_FAILURE() << "tie-break ordering should follow pass declaration index";
    return;
  }
  if (!(compiledA.plan.orderedPassIndices ==
        compiledB.plan.orderedPassIndices)) {
    ADD_FAILURE()
        << "ordered pass indices should be stable across compile calls";
    return;
  }
  if (!(compiledA.plan.edges.size() == compiledB.plan.edges.size())) {
    ADD_FAILURE() << "edge count should be stable across compile calls";
    return;
  }
  for (size_t i = 0; i < compiledA.plan.edges.size(); ++i) {
    if (!(compiledA.plan.edges[i].before == compiledB.plan.edges[i].before &&
          compiledA.plan.edges[i].after == compiledB.plan.edges[i].after)) {
      ADD_FAILURE()
          << "edge ordering/content should be stable across compile calls";
      return;
    }
  }
}

TEST(RenderGraphCompileBehaviorTest,
     TextureSubresourceHazardsOrderOnlyOverlappingRanges) {
  RenderGraphBuilder builder;
  builder.beginFrame(202u);
  TextureDesc textureDesc =
      makeTransientTextureDesc(Format::RGBA8_UNORM, 32u, 32u);
  textureDesc.numMipLevels = 2u;
  auto textureResult =
      builder.createTransientTexture(textureDesc, "subresource_texture");
  ASSERT_FALSE(textureResult.hasError()) << textureResult.error();

  std::array<RenderGraphPassId, 3u> passes{};
  for (uint32_t i = 0u; i < passes.size(); ++i) {
    auto passResult = addTestGraphicsPass(
        builder, makeTestPass("subresource_pass"), "subresource_pass");
    ASSERT_FALSE(passResult.hasError()) << passResult.error();
    passes[i] = passResult.value();
    ASSERT_FALSE(builder.markPassSideEffect(passes[i]).hasError());
  }
  constexpr RenderGraphSubresourceRange mip0{
      .firstMip = 0u, .mipCount = 1u, .firstLayer = 0u, .layerCount = 1u};
  constexpr RenderGraphSubresourceRange mip1{
      .firstMip = 1u, .mipCount = 1u, .firstLayer = 0u, .layerCount = 1u};
  ASSERT_FALSE(builder
                   .addTextureAccess(passes[0], textureResult.value(),
                                     RenderGraphAccessMode::Write, mip0)
                   .hasError());
  ASSERT_FALSE(builder
                   .addTextureAccess(passes[1], textureResult.value(),
                                     RenderGraphAccessMode::Write, mip1)
                   .hasError());
  ASSERT_FALSE(builder
                   .addTextureAccess(passes[2], textureResult.value(),
                                     RenderGraphAccessMode::Read, mip0)
                   .hasError());

  auto compileResult = compileBuilder(builder);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  const RenderGraphPlan &plan = compileResult.value().plan;
  EXPECT_NE(std::ranges::find_if(plan.edges,
                                 [](const auto &edge) {
                                   return edge.before == 0u && edge.after == 2u;
                                 }),
            plan.edges.end());
  EXPECT_EQ(std::ranges::find_if(plan.edges,
                                 [](const auto &edge) {
                                   return edge.before == 0u && edge.after == 1u;
                                 }),
            plan.edges.end());
  EXPECT_EQ(std::ranges::find_if(plan.edges,
                                 [](const auto &edge) {
                                   return edge.before == 1u && edge.after == 2u;
                                 }),
            plan.edges.end());
  ASSERT_EQ(plan.resourceUses.size(), 3u);
  EXPECT_EQ(plan.resourceUses[0].subresources, mip0);
  EXPECT_EQ(plan.resourceUses[1].subresources, mip1);
  EXPECT_EQ(plan.resourceUses[2].subresources, mip0);
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

  const CompiledRenderGraph &serial = serialCompile.value();
  const CompiledRenderGraph &parallel = parallelCompile.value();

  EXPECT_FALSE(serial.plan.usedParallelCompile);
  EXPECT_EQ(serial.plan.orderedPassIndices, parallel.plan.orderedPassIndices);
  EXPECT_EQ(serial.plan.edges.size(), parallel.plan.edges.size());
  EXPECT_EQ(serial.plan.transientBufferLifetimes.size(),
            parallel.plan.transientBufferLifetimes.size());
  EXPECT_EQ(serial.plan.passBarrierPlans.size(),
            parallel.plan.passBarrierPlans.size());
  EXPECT_EQ(serial.plan.passBarrierRecords.size(),
            parallel.plan.passBarrierRecords.size());

  for (size_t i = 0; i < serial.plan.edges.size(); ++i) {
    const auto &lhs = serial.plan.edges[i];
    const auto &rhs = parallel.plan.edges[i];
    EXPECT_EQ(lhs.before, rhs.before);
    EXPECT_EQ(lhs.after, rhs.after);
  }

  for (size_t i = 0; i < serial.plan.transientBufferLifetimes.size(); ++i) {
    const auto &lhs = serial.plan.transientBufferLifetimes[i];
    const auto &rhs = parallel.plan.transientBufferLifetimes[i];
    EXPECT_EQ(lhs.resourceIndex, rhs.resourceIndex);
    EXPECT_EQ(lhs.firstExecutionIndex, rhs.firstExecutionIndex);
    EXPECT_EQ(lhs.lastExecutionIndex, rhs.lastExecutionIndex);
  }

  for (size_t i = 0; i < serial.plan.passBarrierPlans.size(); ++i) {
    const auto &lhs = serial.plan.passBarrierPlans[i];
    const auto &rhs = parallel.plan.passBarrierPlans[i];
    EXPECT_EQ(lhs.orderedPassIndex, rhs.orderedPassIndex);
    EXPECT_EQ(lhs.barrierCount, rhs.barrierCount);
  }

  for (size_t i = 0; i < serial.plan.passBarrierRecords.size(); ++i) {
    const auto &lhs = serial.plan.passBarrierRecords[i];
    const auto &rhs = parallel.plan.passBarrierRecords[i];
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
  const CompiledRenderGraph &compiled = compileResult.value();
  ASSERT_EQ(compiled.plan.commandResourcePatches.size(), 2u);
  const auto *indirect =
      std::get_if<RenderGraphPlan::MeshDispatchIndirectBufferPatch>(
          &compiled.plan.commandResourcePatches[0]);
  ASSERT_NE(indirect, nullptr);
  EXPECT_EQ(indirect->commandIndex, 0u);
  EXPECT_TRUE(std::holds_alternative<
              RenderGraphPlan::MeshDispatchIndirectCountBufferPatch>(
      compiled.plan.commandResourcePatches[1]));
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
  ComputeDispatchItem dispatchA{};
  dispatchA.pipeline = ComputePipelineHandle{.index = 77u, .generation = 1u};
  dispatchA.dispatch = {.x = 1u, .y = 1u, .z = 1u};
  dispatchA.pushConstants =
      std::span<const std::byte>(pushBytesA.data(), pushBytesA.size());
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
     PreResolvedDrawCountChangesInvalidateStructuralPlan) {
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
  CompiledRenderGraph compiled = std::move(compileResult.value());
  ASSERT_EQ(compiled.commands.orderedPasses.size(), 1u);
  ASSERT_EQ(compiled.commands.orderedPasses[0u].draws.size(), 1u);

  const auto fingerprintB =
      recordFrame(269u, std::span<const DrawItem>(draws.data(), draws.size()));
  ASSERT_FALSE(fingerprintA == fingerprintB);
  auto refreshedCompileResult = compileBuilder(builder);
  ASSERT_FALSE(refreshedCompileResult.hasError())
      << refreshedCompileResult.error();
  compiled = std::move(refreshedCompileResult.value());
  ASSERT_EQ(compiled.commands.orderedPasses.size(), 1u);
  ASSERT_EQ(compiled.commands.orderedPasses[0u].draws.size(), draws.size());
  EXPECT_EQ(compiled.commands.orderedPasses[0u].draws[3u].vertexBuffer.index,
            163u);
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
  const CompiledRenderGraph &compiled = compileResult.value();
  ASSERT_EQ(compiled.commands.orderedPasses.size(), 1u);
  const RenderPass &pass = compiled.commands.orderedPasses[0u];
  EXPECT_EQ(pass.executionMode, RenderPassExecutionMode::CopyOnly);
  EXPECT_EQ(pass.debugLabel, "copy_pass");
  ASSERT_EQ(pass.textureCopies.size(), 1u);
  EXPECT_TRUE(sameTexture(pass.textureCopies[0u].sourceTexture, sourceTexture));
  EXPECT_TRUE(sameTexture(pass.textureCopies[0u].destinationTexture,
                          destinationTexture));
  EXPECT_EQ(pass.textureCopies[0u].sourceX, 1u);
  EXPECT_EQ(pass.textureCopies[0u].destinationY, 4u);
  ASSERT_EQ(compiled.plan.textureCopyRangesByPass.size(), 1u);
  EXPECT_EQ(compiled.plan.textureCopyRangesByPass[0u].count, 1u);
  EXPECT_TRUE(compiled.plan.commandResourcePatches.empty());
}

TEST(RenderGraphCompileBehaviorTest,
     BufferCopyPassCompilesAndRefreshesImportedHandles) {
  RenderGraphBuilder builder;
  auto recordFrame = [&](uint64_t frameIndex, BufferHandle source,
                         BufferHandle destination) {
    builder.beginFrame(frameIndex);
    auto sourceResult = builder.importBuffer(source, "buffer_copy_source");
    auto destinationResult =
        builder.importBuffer(destination, "buffer_copy_destination");
    EXPECT_FALSE(sourceResult.hasError());
    EXPECT_FALSE(destinationResult.hasError());
    const std::array<RenderGraphBufferCopyItem, 1u> copies{{
        {.sourceBuffer = sourceResult.value(),
         .destinationBuffer = destinationResult.value(),
         .sourceOffset = 4u,
         .destinationOffset = 8u,
         .size = 16u},
    }};
    auto passResult = builder.addBufferCopyPass(RenderGraphBufferCopyPassDesc{
        .copies = copies,
        .debugLabel = "buffer_copy_pass",
    });
    EXPECT_FALSE(passResult.hasError());
    return builder.computeGraphFingerprint();
  };

  const BufferHandle sourceA{.index = 321u, .generation = 1u};
  const BufferHandle destinationA{.index = 322u, .generation = 1u};
  const auto fingerprintA = recordFrame(273u, sourceA, destinationA);
  auto compileResult = compileBuilder(builder);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  CompiledRenderGraph compiled = std::move(compileResult.value());
  ASSERT_EQ(compiled.commands.orderedPasses.size(), 1u);
  const RenderPass &pass = compiled.commands.orderedPasses[0u];
  EXPECT_EQ(pass.executionMode, RenderPassExecutionMode::CopyOnly);
  ASSERT_EQ(pass.bufferCopies.size(), 1u);
  EXPECT_TRUE(sameBuffer(pass.bufferCopies[0u].srcBuffer, sourceA));
  EXPECT_TRUE(sameBuffer(pass.bufferCopies[0u].dstBuffer, destinationA));
  EXPECT_EQ(pass.bufferCopies[0u].srcOffset, 4u);
  EXPECT_EQ(pass.bufferCopies[0u].dstOffset, 8u);
  EXPECT_EQ(pass.bufferCopies[0u].size, 16u);
  EXPECT_TRUE(compiled.plan.commandResourcePatches.empty());

  const BufferHandle sourceB{.index = 323u, .generation = 2u};
  const BufferHandle destinationB{.index = 324u, .generation = 2u};
  const auto fingerprintB = recordFrame(274u, sourceB, destinationB);
  ASSERT_TRUE(fingerprintA == fingerprintB);
  compiled.commands = builder.buildFrameCommands(compiled.plan);
  ASSERT_EQ(compiled.commands.orderedPasses[0u].bufferCopies.size(), 1u);
  EXPECT_TRUE(sameBuffer(
      compiled.commands.orderedPasses[0u].bufferCopies[0u].srcBuffer, sourceB));
  EXPECT_TRUE(
      sameBuffer(compiled.commands.orderedPasses[0u].bufferCopies[0u].dstBuffer,
                 destinationB));
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
  CompiledRenderGraph compiled = std::move(compileResult.value());
  ASSERT_EQ(compiled.commands.orderedPasses.size(), 1u);
  ASSERT_EQ(compiled.commands.orderedPasses[0u].textureCopies.size(), 1u);
  EXPECT_TRUE(sameTexture(
      compiled.commands.orderedPasses[0u].textureCopies[0u].sourceTexture,
      sourceA));

  const auto fingerprintB = recordFrame(272u, sourceB, destinationB);
  ASSERT_TRUE(fingerprintA == fingerprintB);
  compiled.commands = builder.buildFrameCommands(compiled.plan);

  ASSERT_EQ(compiled.commands.orderedPasses.size(), 1u);
  ASSERT_EQ(compiled.commands.orderedPasses[0u].textureCopies.size(), 1u);
  EXPECT_TRUE(sameTexture(
      compiled.commands.orderedPasses[0u].textureCopies[0u].sourceTexture,
      sourceB));
  EXPECT_TRUE(sameTexture(
      compiled.commands.orderedPasses[0u].textureCopies[0u].destinationTexture,
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

  ComputeDispatchItem dispatch{};
  dispatch.pipeline = ComputePipelineHandle{.index = 88u, .generation = 1u};
  dispatch.dispatch = {.x = 1u, .y = 1u, .z = 1u};
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
                   .addBufferAccess(passResult.value(),
                                    resultBufferResult.value(),
                                    RenderGraphAccessMode::Write)
                   .hasError());

  auto compileResult = compileBuilder(builder);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  ASSERT_EQ(compileResult.value().commands.orderedPasses.size(), 1u);
  EXPECT_EQ(compileResult.value().commands.orderedPasses[0].executionMode,
            RenderPassExecutionMode::ComputeOnly);
  ASSERT_EQ(
      compileResult.value().commands.orderedPasses[0].preDispatches.size(), 1u);
}

TEST(RenderGraphCompileBehaviorTest,
     CanonicalImportedUsesPreserveExactAccessModes) {
  RenderGraphBuilder builder;
  builder.beginFrame(245u);

  const std::array<BufferHandle, 2u> dependencies = {
      BufferHandle{.index = 501u, .generation = 1u},
      BufferHandle{.index = 502u, .generation = 1u},
  };
  const std::array<RenderGraphImportedBufferUse, 2u> uses{{
      {.buffer = dependencies[0], .access = RenderGraphAccessMode::Read},
      {.buffer = dependencies[1], .access = RenderGraphAccessMode::Write},
  }};
  ComputeDispatchItem dispatch{};
  dispatch.pipeline = ComputePipelineHandle{.index = 91u, .generation = 1u};
  dispatch.dispatch = {.x = 1u, .y = 1u, .z = 1u};
  dispatch.debugLabel = "exact_access_dispatch";
  const std::array<ComputeDispatchItem, 1u> preDispatches = {dispatch};

  RenderGraphGraphicsPassDesc passDesc{};
  passDesc.executionMode = RenderPassExecutionMode::ComputeOnly;
  passDesc.hasColorAttachment = false;
  passDesc.preDispatches = preDispatches;
  passDesc.importedBufferUses = uses;
  passDesc.debugLabel = "exact_access_pass";
  passDesc.markImplicitOutputSideEffect = true;

  auto passResult = builder.addGraphicsPass(passDesc);
  ASSERT_FALSE(passResult.hasError()) << passResult.error();

  auto compileResult = compileBuilder(builder);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  const CompiledRenderGraph &compiled = compileResult.value();
  ASSERT_EQ(compiled.plan.passBarrierRecords.size(), 2u);
  EXPECT_EQ(compiled.plan.passBarrierRecords[0].afterAccess,
            RenderGraphAccessMode::Read);
  EXPECT_EQ(compiled.plan.passBarrierRecords[1].afterAccess,
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
  const CompiledRenderGraph &compiled = compileResult.value();

  if (!(compiled.plan.edges.size() == 1u)) {
    ADD_FAILURE()
        << "overlapping explicit+hazard dependency should collapse to "
           "one edge";
    return;
  }
  if (!(compiled.plan.edges[0u].before == pass0Result.value().value &&
        compiled.plan.edges[0u].after == pass1Result.value().value)) {
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
  const CompiledRenderGraph &compiledB = compileB.value();
  if (!(compiledB.plan.edges.size() == 1u)) {
    ADD_FAILURE() << "frame B explicit dependency should be present after "
                     "beginFrame reset";
    return;
  }
  if (!(compiledB.plan.rootPassCount >= 1u)) {
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
  const CompiledRenderGraph &compiled = compileResult.value();

  if (!(compiled.plan.declaredPassCount == 3u)) {
    ADD_FAILURE() << "culling graph should declare 3 passes";
    return;
  }
  if (!(compiled.plan.culledPassCount == 1u)) {
    ADD_FAILURE() << "exactly one dead pass should be culled";
    return;
  }
  if (!(compiled.plan.rootPassCount == 1u)) {
    ADD_FAILURE() << "frame-output culling graph should have one root writer";
    return;
  }
  if (!(compiled.plan.orderedPassIndices.size() == 2u)) {
    ADD_FAILURE() << "only root-reachable passes should remain after culling";
    return;
  }
  if (!(compiled.plan.orderedPassIndices[0u] == pass0Result.value().value &&
        compiled.plan.orderedPassIndices[1u] == pass1Result.value().value)) {
    ADD_FAILURE() << "culling should preserve producer->output order";
    return;
  }
  if (!(std::find(compiled.plan.orderedPassIndices.begin(),
                  compiled.plan.orderedPassIndices.end(),
                  pass2Result.value().value) ==
        compiled.plan.orderedPassIndices.end())) {
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
  const CompiledRenderGraph &compiled = compileResult.value();

  if (!(bufferAResult.value().value <
            compiled.plan.transientBufferAllocationByResource.size() &&
        bufferBResult.value().value <
            compiled.plan.transientBufferAllocationByResource.size() &&
        bufferCResult.value().value <
            compiled.plan.transientBufferAllocationByResource.size())) {
    ADD_FAILURE()
        << "buffer allocation map should contain all transient resources";
    return;
  }
  const uint32_t bufferAllocA =
      compiled.plan
          .transientBufferAllocationByResource[bufferAResult.value().value];
  const uint32_t bufferAllocB =
      compiled.plan
          .transientBufferAllocationByResource[bufferBResult.value().value];
  const uint32_t bufferAllocC =
      compiled.plan
          .transientBufferAllocationByResource[bufferCResult.value().value];
  if (!(compiled.plan.transientBufferPhysicalCount == 2u)) {
    ADD_FAILURE() << "buffer aliasing should collapse 3 logical buffers to 2 "
                     "physical allocations";
    std::cerr << "[INFO] transientBufferPhysicalCount="
              << compiled.plan.transientBufferPhysicalCount << "\n";
    return;
  }
  if (!(compiled.plan.transientTexturePhysicalCount == 2u)) {
    ADD_FAILURE() << "texture aliasing should collapse 3 logical textures to 2 "
                     "physical allocations";
    std::cerr << "[INFO] transientTexturePhysicalCount="
              << compiled.plan.transientTexturePhysicalCount << "\n";
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
              << " buffer_alloc_c=" << bufferAllocC << " physical_count="
              << compiled.plan.transientBufferPhysicalCount << "\n";
    std::cerr << "[INFO] ordered_pass_indices:";
    for (const uint32_t passIndex : compiled.plan.orderedPassIndices) {
      std::cerr << " " << passIndex;
    }
    std::cerr << "\n";
    std::cerr << "[INFO] buffer_lifetimes:";
    for (const auto &lifetime : compiled.plan.transientBufferLifetimes) {
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
            compiled.plan.transientTextureAllocationByResource.size() &&
        textureBResult.value().value <
            compiled.plan.transientTextureAllocationByResource.size() &&
        textureCResult.value().value <
            compiled.plan.transientTextureAllocationByResource.size())) {
    ADD_FAILURE()
        << "texture allocation map should contain all transient resources";
    return;
  }
  const uint32_t textureAllocA =
      compiled.plan
          .transientTextureAllocationByResource[textureAResult.value().value];
  const uint32_t textureAllocB =
      compiled.plan
          .transientTextureAllocationByResource[textureBResult.value().value];
  const uint32_t textureAllocC =
      compiled.plan
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

TEST(RenderGraphCompileBehaviorTest,
     ExplicitCanonicalAccessMergesAcrossPasses) {
  RenderGraphBuilder builder;
  builder.beginFrame(205u);

  const BufferHandle sharedDependency{.index = 31u, .generation = 1u};
  const TextureHandle color0{.index = 41u, .generation = 1u};
  const TextureHandle color1{.index = 42u, .generation = 1u};
  RenderPass pass0 = makeTestPass("override_p0", color0);
  RenderPass pass1 = makeTestPass("override_p1", color1);

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
  auto bindResult = builder.addBufferAccess(
      pass0Result.value(), importResult.value(), RenderGraphAccessMode::Read);
  if (!(!bindResult.hasError())) {
    ADD_FAILURE() << "addBufferAccess pass0 read should succeed";
    return;
  }
  bindResult = builder.addBufferAccess(
      pass1Result.value(), importResult.value(), RenderGraphAccessMode::Read);
  if (!(!bindResult.hasError())) {
    ADD_FAILURE() << "addBufferAccess pass1 read should succeed";
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
  const CompiledRenderGraph &compiled = compileResult.value();

  if (!(compiled.plan.orderedPassIndices.size() == 2u)) {
    ADD_FAILURE() << "access override graph should schedule two passes";
    return;
  }
  if (!(compiled.plan.edges.empty())) {
    ADD_FAILURE()
        << "explicit read overrides should prevent inferred RW hazard edge";
    std::cerr << "[INFO] edge_count=" << compiled.plan.edges.size() << "\n";
    for (const auto &edge : compiled.plan.edges) {
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
  const CompiledRenderGraph &compiled = compileResult.value();
  ASSERT_EQ(compiled.commands.orderedPasses.size(), 1u);
  EXPECT_FALSE(compiled.commands.orderedPasses.front().hasColorAttachment);
  EXPECT_EQ(compiled.commands.orderedPasses.front().depth.clearStencil, 0u);
  ASSERT_EQ(compiled.plan.transientTexturePhysicalAllocations.size(), 1u);
  const TextureDesc &compiledDepthDesc =
      compiled.plan.transientTexturePhysicalAllocations.front().desc;
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
  desc.debugLabel = "many_texture_reads";

  auto passResult = builder.addGraphicsPass(desc);
  ASSERT_FALSE(passResult.hasError()) << passResult.error();
  ASSERT_FALSE(builder
                   .addImportedTextureReads(passResult.value(),
                                            dependencyTextures, desc.debugLabel)
                   .hasError());

  auto compileResult = compileBuilder(builder);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  EXPECT_EQ(compileResult.value().plan.recordedGraphicsPasses.size(), 1u);
}

} // namespace
