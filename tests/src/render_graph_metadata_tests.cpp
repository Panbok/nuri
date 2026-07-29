#include "tests_pch.h"

#include "render_graph_test_support.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

using namespace nuri;
using namespace nuri::test_support;

bool sameTextureHandle(TextureHandle lhs, TextureHandle rhs) {
  return sameHandle(lhs, rhs);
}

Result<CompiledRenderGraph, std::string>
compileBuilder(RenderGraphBuilder &builder) {
  RenderGraphRuntime runtime;
  return builder.compile(runtime);
}

TEST(RenderGraphMetadataTest, ImportedResourceDedupAndFrameReset) {
  RenderGraphBuilder builder;
  builder.beginFrame(305u);

  const TextureHandle importedTexture{.index = 55u, .generation = 3u};
  const BufferHandle importedBuffer{.index = 77u, .generation = 5u};

  auto importTextureResultA =
      builder.importTexture(importedTexture, "dedup_imported_texture_a");
  auto importTextureResultB =
      builder.importTexture(importedTexture, "dedup_imported_texture_b");
  if (importTextureResultA.hasError() || importTextureResultB.hasError()) {
    ADD_FAILURE() << "importTexture dedup inserts should succeed";
    return;
  }
  if (!(importTextureResultA.value().value ==
        importTextureResultB.value().value)) {
    ADD_FAILURE()
        << "importTexture duplicate handles should map to one resource";
    return;
  }

  auto importBufferResultA =
      builder.importBuffer(importedBuffer, "dedup_imported_buffer_a");
  auto importBufferResultB =
      builder.importBuffer(importedBuffer, "dedup_imported_buffer_b");
  if (importBufferResultA.hasError() || importBufferResultB.hasError()) {
    ADD_FAILURE() << "importBuffer dedup inserts should succeed";
    return;
  }
  if (!(importBufferResultA.value().value ==
        importBufferResultB.value().value)) {
    ADD_FAILURE()
        << "importBuffer duplicate handles should map to one resource";
    return;
  }

  builder.beginFrame(306u);
  auto frameResetTextureResult =
      builder.importTexture(importedTexture, "frame_reset_texture");
  auto frameResetBufferResult =
      builder.importBuffer(importedBuffer, "frame_reset_buffer");
  if (frameResetTextureResult.hasError() || frameResetBufferResult.hasError()) {
    ADD_FAILURE() << "frame-reset imports should succeed";
    return;
  }
  if (!(frameResetTextureResult.value().value == 0u)) {
    ADD_FAILURE()
        << "first imported texture after beginFrame should restart at "
           "resource index 0";
    return;
  }
  if (!(frameResetBufferResult.value().value == 0u)) {
    ADD_FAILURE() << "first imported buffer after beginFrame should restart at "
                     "resource index 0";
    return;
  }
}

TEST(RenderGraphMetadataTest, BarrierPlansTrackStablePerPassTransitions) {
  RenderGraphBuilder builder;
  builder.beginFrame(310u);

  auto transientBufferResult = builder.createTransientBuffer(
      makeTransientBufferDesc(64u), "barrier_buf");
  if (transientBufferResult.hasError()) {
    ADD_FAILURE() << "createTransientBuffer should succeed";
    return;
  }

  RenderPass passA{};
  passA.debugLabel = "barrier_pass_a";
  RenderPass passB{};
  passB.debugLabel = "barrier_pass_b";

  auto passAResult = addTestGraphicsPass(builder, passA, passA.debugLabel);
  auto passBResult = addTestGraphicsPass(builder, passB, passB.debugLabel);
  if (passAResult.hasError() || passBResult.hasError()) {
    ADD_FAILURE() << "addLegacyRenderPass should succeed";
    return;
  }

  auto accessResult = builder.addBufferWrite(passAResult.value(),
                                             transientBufferResult.value());
  if (accessResult.hasError()) {
    ADD_FAILURE() << "addBufferWrite should succeed";
    return;
  }
  accessResult =
      builder.addBufferRead(passBResult.value(), transientBufferResult.value());
  if (accessResult.hasError()) {
    ADD_FAILURE() << "addBufferRead should succeed";
    return;
  }
  auto depResult =
      builder.addDependency(passAResult.value(), passBResult.value());
  if (depResult.hasError()) {
    ADD_FAILURE() << "addDependency should succeed";
    return;
  }
  auto rootResult = builder.markPassSideEffect(passBResult.value());
  if (rootResult.hasError()) {
    ADD_FAILURE() << "markPassSideEffect should succeed";
    return;
  }

  auto compileResult = compileBuilder(builder);
  if (compileResult.hasError()) {
    ADD_FAILURE() << "compile should succeed";
    std::cerr << compileResult.error() << "\n";
    return;
  }
  const CompiledRenderGraph &compiled = compileResult.value();

  if (!(compiled.plan.passBarrierPlans.size() == 2u)) {
    ADD_FAILURE() << "expected two pass barrier plans";
    return;
  }
  if (!(compiled.plan.passBarrierRecords.size() == 2u)) {
    ADD_FAILURE() << "expected two pass barrier records";
    return;
  }

  const PassBarrierPlan &planA = compiled.plan.passBarrierPlans[0u];
  const PassBarrierPlan &planB = compiled.plan.passBarrierPlans[1u];
  if (!(planA.orderedPassIndex == 0u)) {
    ADD_FAILURE() << "expected plan A ordered pass index = 0";
    return;
  }
  if (!(planB.orderedPassIndex == 1u)) {
    ADD_FAILURE() << "expected plan B ordered pass index = 1";
    return;
  }
  if (!(planA.barrierCount == 1u)) {
    ADD_FAILURE() << "expected plan A barrier count = 1";
    return;
  }
  if (!(planB.barrierCount == 1u)) {
    ADD_FAILURE() << "expected plan B barrier count = 1";
    return;
  }

  const RenderGraphBarrierRecord &recordA =
      compiled.plan.passBarrierRecords[planA.barrierOffset];
  if (!(recordA.resourceKind == RenderGraphBarrierResourceKind::Buffer)) {
    ADD_FAILURE() << "expected plan A to target a buffer barrier";
    return;
  }
  if (!(recordA.resourceIndex == transientBufferResult.value().value)) {
    ADD_FAILURE() << "expected plan A buffer resource index to match";
    return;
  }
  if (!(recordA.beforeState == RenderGraphResourceState::Unknown)) {
    ADD_FAILURE() << "expected plan A before-state = Unknown";
    return;
  }
  if (!(recordA.afterState == RenderGraphResourceState::Write)) {
    ADD_FAILURE() << "expected plan A after-state = Write";
    return;
  }

  const RenderGraphBarrierRecord &recordB =
      compiled.plan.passBarrierRecords[planB.barrierOffset];
  if (!(recordB.resourceKind == RenderGraphBarrierResourceKind::Buffer)) {
    ADD_FAILURE() << "expected plan B to target a buffer barrier";
    return;
  }
  if (!(recordB.resourceIndex == transientBufferResult.value().value)) {
    ADD_FAILURE() << "expected plan B buffer resource index to match";
    return;
  }
  if (!(recordB.beforeState == RenderGraphResourceState::Write)) {
    ADD_FAILURE() << "expected plan B before-state = Write";
    return;
  }
  if (!(recordB.afterState == RenderGraphResourceState::Read)) {
    ADD_FAILURE() << "expected plan B after-state = Read";
    return;
  }
  if (!(recordB.beforeAccess == RenderGraphAccessMode::Write)) {
    ADD_FAILURE() << "expected plan B before-access = Write";
    return;
  }
  if (!(recordB.afterAccess == RenderGraphAccessMode::Read)) {
    ADD_FAILURE() << "expected plan B after-access = Read";
    return;
  }
}

TEST(RenderGraphMetadataTest, FrameOutputTexturesEmitFinalPresentBarrierPlan) {
  RenderGraphBuilder builder;
  builder.beginFrame(111u);

  const TextureHandle outputTexture{.index = 901u, .generation = 1u};
  auto importResult = builder.importTexture(outputTexture, "metadata_output");
  ASSERT_FALSE(importResult.hasError());

  RenderPass pass{};
  pass.colorTexture = outputTexture;
  pass.debugLabel = "metadata_frame_output";
  auto passResult = addTestGraphicsPass(builder, pass, pass.debugLabel);
  ASSERT_FALSE(passResult.hasError());

  auto bindResult =
      builder.bindPassColorTexture(passResult.value(), importResult.value());
  ASSERT_FALSE(bindResult.hasError());
  auto outputResult = builder.markTextureAsFrameOutput(importResult.value());
  ASSERT_FALSE(outputResult.hasError());

  auto compileResult = compileBuilder(builder);
  ASSERT_FALSE(compileResult.hasError());
  const CompiledRenderGraph &compiled = compileResult.value();

  ASSERT_EQ(compiled.plan.passBarrierPlans.size(), 1u);
  ASSERT_EQ(compiled.plan.finalBarrierPlan.barrierCount, 1u);
  ASSERT_LT(compiled.plan.finalBarrierPlan.barrierOffset,
            compiled.plan.passBarrierRecords.size());

  const RenderGraphBarrierRecord &record =
      compiled.plan
          .passBarrierRecords[compiled.plan.finalBarrierPlan.barrierOffset];
  EXPECT_EQ(record.resourceKind, RenderGraphBarrierResourceKind::Texture);
  EXPECT_EQ(record.resourceIndex, importResult.value().value);
  EXPECT_EQ(record.beforeState, RenderGraphResourceState::Attachment);
  EXPECT_EQ(record.afterState, RenderGraphResourceState::Present);
  EXPECT_EQ(record.beforeAccess, RenderGraphAccessMode::Write);
  EXPECT_EQ(record.afterAccess, RenderGraphAccessMode::None);
}

TEST(RenderGraphMetadataTest, MultiPassRangeMetadataIntegrity) {
  RenderGraphBuilder builder;
  builder.beginFrame(103u);

  auto transientAResult = builder.createTransientBuffer(
      makeTransientBufferDesc(64u), "multi_transient_a");
  if (transientAResult.hasError()) {
    ADD_FAILURE() << "createTransientBuffer A should succeed";
    return;
  }
  auto transientBResult = builder.createTransientBuffer(
      makeTransientBufferDesc(64u), "multi_transient_b");
  if (transientBResult.hasError()) {
    ADD_FAILURE() << "createTransientBuffer B should succeed";
    return;
  }
  const RenderGraphBufferId transientA = transientAResult.value();
  const RenderGraphBufferId transientB = transientBResult.value();

  std::array<BufferHandle, 2> pass0Deps = {BufferHandle{}, BufferHandle{}};
  std::array<BufferHandle, 1> pass0Dispatch0Deps = {BufferHandle{}};
  std::array<BufferHandle, 1> pass0Dispatch1Deps = {BufferHandle{}};
  ComputeDispatchItem pass0Dispatch0{};
  pass0Dispatch0.dependencyBuffers = std::span<const BufferHandle>(
      pass0Dispatch0Deps.data(), pass0Dispatch0Deps.size());
  ComputeDispatchItem pass0Dispatch1{};
  pass0Dispatch1.dependencyBuffers = std::span<const BufferHandle>(
      pass0Dispatch1Deps.data(), pass0Dispatch1Deps.size());
  std::array<ComputeDispatchItem, 2> pass0PreDispatches = {pass0Dispatch0,
                                                           pass0Dispatch1};
  std::array<DrawItem, 2> pass0Draws = {DrawItem{}, DrawItem{}};

  RenderPass pass0{};
  pass0.dependencyBuffers =
      std::span<const BufferHandle>(pass0Deps.data(), pass0Deps.size());
  pass0.preDispatches = std::span<const ComputeDispatchItem>(
      pass0PreDispatches.data(), pass0PreDispatches.size());
  pass0.draws = std::span<const DrawItem>(pass0Draws.data(), pass0Draws.size());
  pass0.debugLabel = "rg_test_multi_pass0";

  RenderPass pass1{};
  pass1.debugLabel = "rg_test_multi_pass1";

  auto pass0Result = addTestGraphicsPass(builder, pass0, pass0.debugLabel);
  if (pass0Result.hasError()) {
    ADD_FAILURE() << "addLegacyRenderPass pass0 should succeed";
    return;
  }
  auto pass1Result = addTestGraphicsPass(builder, pass1, pass1.debugLabel);
  if (pass1Result.hasError()) {
    ADD_FAILURE() << "addLegacyRenderPass pass1 should succeed";
    return;
  }

  const RenderGraphPassId pass0Id = pass0Result.value();
  auto bindResult = builder.bindPassDependencyBuffer(
      pass0Id, 0u, transientA,
      RenderGraphAccessMode::Read | RenderGraphAccessMode::Write);
  if (bindResult.hasError()) {
    ADD_FAILURE() << "bindPassDependencyBuffer pass0 slot0 should succeed";
    return;
  }
  bindResult = builder.bindPassDependencyBuffer(
      pass0Id, 1u, transientB,
      RenderGraphAccessMode::Read | RenderGraphAccessMode::Write);
  if (bindResult.hasError()) {
    ADD_FAILURE() << "bindPassDependencyBuffer pass0 slot1 should succeed";
    return;
  }

  auto bindPreResult = builder.bindPreDispatchDependencyBuffer(
      pass0Id, 0u, 0u, transientA,
      RenderGraphAccessMode::Read | RenderGraphAccessMode::Write);
  if (bindPreResult.hasError()) {
    ADD_FAILURE() << "bindPreDispatchDependencyBuffer pass0 dispatch0 should "
                     "succeed";
    return;
  }
  bindPreResult = builder.bindPreDispatchDependencyBuffer(
      pass0Id, 1u, 0u, transientB,
      RenderGraphAccessMode::Read | RenderGraphAccessMode::Write);
  if (bindPreResult.hasError()) {
    ADD_FAILURE() << "bindPreDispatchDependencyBuffer pass0 dispatch1 should "
                     "succeed";
    return;
  }

  auto bindDrawResult = builder.bindDrawBuffer(
      pass0Id, 0u, RenderGraphDrawBufferBindingTarget::Vertex, transientA,
      RenderGraphAccessMode::Read);
  if (bindDrawResult.hasError()) {
    ADD_FAILURE() << "bindDrawBuffer pass0 draw0 should succeed";
    return;
  }
  bindDrawResult = builder.bindDrawBuffer(
      pass0Id, 1u, RenderGraphDrawBufferBindingTarget::Index, transientB,
      RenderGraphAccessMode::Read);
  if (bindDrawResult.hasError()) {
    ADD_FAILURE() << "bindDrawBuffer pass0 draw1 should succeed";
    return;
  }

  auto compileResult = compileBuilder(builder);
  if (compileResult.hasError()) {
    ADD_FAILURE() << "compile should succeed";
    std::cerr << compileResult.error() << "\n";
    return;
  }
  const CompiledRenderGraph &compiled = compileResult.value();

  if (!(compiled.commands.orderedPasses.size() == 2u)) {
    ADD_FAILURE() << "expected two ordered passes";
    return;
  }
  if (!(compiled.plan.dependencyBufferRangesByPass.size() == 2u)) {
    ADD_FAILURE() << "dependency range table should have two entries";
    return;
  }
  if (!(compiled.plan.preDispatchRangesByPass.size() == 2u)) {
    ADD_FAILURE() << "pre-dispatch range table should have two entries";
    return;
  }
  if (!(compiled.plan.drawRangesByPass.size() == 2u)) {
    ADD_FAILURE() << "draw range table should have two entries";
    return;
  }

  const auto pass0DepRange = compiled.plan.dependencyBufferRangesByPass[0u];
  const auto pass1DepRange = compiled.plan.dependencyBufferRangesByPass[1u];
  if (!(pass0DepRange.count == 2u)) {
    ADD_FAILURE() << "pass0 dependency range should have two slots";
    return;
  }
  if (!(pass1DepRange.count == 0u)) {
    ADD_FAILURE() << "pass1 dependency range should be empty";
    return;
  }

  const auto pass0PreRange = compiled.plan.preDispatchRangesByPass[0u];
  const auto pass1PreRange = compiled.plan.preDispatchRangesByPass[1u];
  if (!(pass0PreRange.count == 2u)) {
    ADD_FAILURE() << "pass0 pre-dispatch range should have two items";
    return;
  }
  if (!(pass1PreRange.count == 0u)) {
    ADD_FAILURE() << "pass1 pre-dispatch range should be empty";
    return;
  }

  const auto pass0DrawRange = compiled.plan.drawRangesByPass[0u];
  const auto pass1DrawRange = compiled.plan.drawRangesByPass[1u];
  if (!(pass0DrawRange.count == 2u)) {
    ADD_FAILURE() << "pass0 draw range should have two draws";
    return;
  }
  if (!(pass1DrawRange.count == 0u)) {
    ADD_FAILURE() << "pass1 draw range should be empty";
    return;
  }

  if (!(compiled.plan.commandResourcePatches.size() == 6u)) {
    ADD_FAILURE() << "expected six command resource patches";
    return;
  }

  for (const auto &binding : compiled.plan.commandResourcePatches) {
    if (!(binding.orderedPassIndex == 0u)) {
      ADD_FAILURE() << "resource patch should reference pass0";
      return;
    }
    if (binding.target ==
            RenderGraphPlan::CommandResourcePatchTarget::PassDependencyBuffer &&
        !(binding.dependencyIndex < pass0DepRange.count)) {
      ADD_FAILURE() << "dependency patch slot should be in pass0 range";
      return;
    }
    if (binding.target == RenderGraphPlan::CommandResourcePatchTarget::
                              PreDispatchDependencyBuffer &&
        !(binding.commandIndex < pass0PreRange.count)) {
      ADD_FAILURE() << "pre-dispatch patch should be in pass0 range";
      return;
    }
    const auto target = binding.target;
    const bool drawPatch =
        target ==
            RenderGraphPlan::CommandResourcePatchTarget::DrawVertexBuffer ||
        target == RenderGraphPlan::CommandResourcePatchTarget::DrawIndexBuffer;
    if (drawPatch && !(binding.commandIndex < pass0DrawRange.count)) {
      ADD_FAILURE() << "draw patch should be in pass0 range";
      return;
    }
  }
}

TEST(RenderGraphMetadataTest, StructuralCompileMetadataIntegrity) {
  RenderGraphBuilder builder;
  builder.beginFrame(106u);
  builder.setInferredSideEffectSuppression(true);

  const TextureHandle colorA{.index = 201u, .generation = 1u};
  const TextureHandle colorB{.index = 202u, .generation = 1u};

  RenderPass implicitPass{};
  implicitPass.debugLabel = "struct_implicit";
  auto implicitResult =
      addTestGraphicsPass(builder, implicitPass, implicitPass.debugLabel);
  if (implicitResult.hasError()) {
    ADD_FAILURE() << "addLegacyRenderPass implicit should succeed";
    return;
  }

  RenderPass producerPass{};
  producerPass.debugLabel = "struct_producer";
  producerPass.colorTexture = colorA;
  auto producerResult =
      addTestGraphicsPass(builder, producerPass, producerPass.debugLabel);
  if (producerResult.hasError()) {
    ADD_FAILURE() << "addLegacyRenderPass producer should succeed";
    return;
  }

  RenderPass outputPass{};
  outputPass.debugLabel = "struct_output";
  outputPass.colorTexture = colorB;
  auto outputResult =
      addTestGraphicsPass(builder, outputPass, outputPass.debugLabel);
  if (outputResult.hasError()) {
    ADD_FAILURE() << "addLegacyRenderPass output should succeed";
    return;
  }

  auto depResult =
      builder.addDependency(producerResult.value(), outputResult.value());
  if (depResult.hasError()) {
    ADD_FAILURE() << "addDependency producer->output should succeed";
    return;
  }

  auto outputImportResult = builder.importTexture(colorB, "struct_output_tex");
  if (outputImportResult.hasError()) {
    ADD_FAILURE() << "importTexture output should succeed";
    return;
  }
  auto markResult =
      builder.markTextureAsFrameOutput(outputImportResult.value());
  if (markResult.hasError()) {
    ADD_FAILURE() << "markTextureAsFrameOutput should succeed";
    return;
  }

  auto compileResult = compileBuilder(builder);
  if (compileResult.hasError()) {
    ADD_FAILURE() << "compile should succeed";
    std::cerr << compileResult.error() << "\n";
    return;
  }
  const CompiledRenderGraph &compiled = compileResult.value();

  if (!(compiled.plan.declaredPassCount == 3u)) {
    ADD_FAILURE() << "structural test should declare three passes";
    return;
  }
  if (!(compiled.plan.culledPassCount == 1u)) {
    ADD_FAILURE() << "structural test should cull one implicit pass";
    return;
  }
  if (!(compiled.commands.orderedPasses.size() ==
        compiled.plan.orderedPassIndices.size())) {
    ADD_FAILURE() << "ordered pass tables should have matching sizes";
    return;
  }
  if (!(compiled.commands.passDebugNames.size() ==
        compiled.plan.declaredPassCount)) {
    ADD_FAILURE() << "pass debug-name table should match declared pass count";
    return;
  }
  if (!(compiled.commands.orderedPasses.size() +
            compiled.plan.culledPassCount ==
        compiled.plan.declaredPassCount)) {
    ADD_FAILURE() << "ordered + culled pass counts should match declared pass "
                     "count";
    return;
  }

  std::array<bool, 3> seenPass{false, false, false};
  for (const uint32_t passIndex : compiled.plan.orderedPassIndices) {
    if (!(passIndex < compiled.plan.declaredPassCount)) {
      ADD_FAILURE() << "ordered pass index should be in declared-pass range";
      return;
    }
    if (!(!seenPass[passIndex])) {
      ADD_FAILURE() << "ordered pass index table should not contain duplicates";
      return;
    }
    seenPass[passIndex] = true;
  }
}

TEST(RenderGraphMetadataTest, TransientAllocationMetadataIntegrity) {
  RenderGraphBuilder builder;
  builder.beginFrame(107u);

  RenderPass pass0{};
  pass0.debugLabel = "alloc_meta_pass0";
  auto pass0Result = addTestGraphicsPass(builder, pass0, pass0.debugLabel);
  if (pass0Result.hasError()) {
    ADD_FAILURE() << "addLegacyRenderPass pass0 should succeed";
    return;
  }

  RenderPass pass1{};
  pass1.debugLabel = "alloc_meta_pass1";
  auto pass1Result = addTestGraphicsPass(builder, pass1, pass1.debugLabel);
  if (pass1Result.hasError()) {
    ADD_FAILURE() << "addLegacyRenderPass pass1 should succeed";
    return;
  }

  auto depResult =
      builder.addDependency(pass0Result.value(), pass1Result.value());
  if (depResult.hasError()) {
    ADD_FAILURE() << "addDependency pass0->pass1 should succeed";
    return;
  }

  auto transientTextureAResult = builder.createTransientTexture(
      makeTransientTextureDesc(Format::RGBA8_UNORM, 16u, 16u), "alloc_tex_a");
  auto transientTextureBResult = builder.createTransientTexture(
      makeTransientTextureDesc(Format::RGBA8_UNORM, 16u, 16u), "alloc_tex_b");
  auto transientBufferAResult = builder.createTransientBuffer(
      makeTransientBufferDesc(64u), "alloc_buf_a");
  auto transientBufferBResult = builder.createTransientBuffer(
      makeTransientBufferDesc(64u), "alloc_buf_b");
  if (transientTextureAResult.hasError() ||
      transientTextureBResult.hasError() || transientBufferAResult.hasError() ||
      transientBufferBResult.hasError()) {
    ADD_FAILURE() << "createTransientTexture/Buffer should succeed";
    return;
  }

  auto accessResult = builder.addTextureWrite(pass0Result.value(),
                                              transientTextureAResult.value());
  if (accessResult.hasError()) {
    ADD_FAILURE() << "addTextureWrite pass0 texA should succeed";
    return;
  }
  accessResult = builder.addTextureWrite(pass1Result.value(),
                                         transientTextureBResult.value());
  if (accessResult.hasError()) {
    ADD_FAILURE() << "addTextureWrite pass1 texB should succeed";
    return;
  }
  auto bufferAccessResult = builder.addBufferWrite(
      pass0Result.value(), transientBufferAResult.value());
  if (bufferAccessResult.hasError()) {
    ADD_FAILURE() << "addBufferWrite pass0 bufA should succeed";
    return;
  }
  bufferAccessResult = builder.addBufferWrite(pass1Result.value(),
                                              transientBufferBResult.value());
  if (bufferAccessResult.hasError()) {
    ADD_FAILURE() << "addBufferWrite pass1 bufB should succeed";
    return;
  }

  auto compileResult = compileBuilder(builder);
  if (compileResult.hasError()) {
    ADD_FAILURE() << "compile should succeed";
    std::cerr << compileResult.error() << "\n";
    return;
  }
  const CompiledRenderGraph &compiled = compileResult.value();

  if (!(compiled.plan.transientTextureAllocations.size() ==
        compiled.plan.transientTextureLifetimes.size())) {
    ADD_FAILURE()
        << "transient texture allocation/lifetime counts should match";
    return;
  }
  if (!(compiled.plan.transientBufferAllocations.size() ==
        compiled.plan.transientBufferLifetimes.size())) {
    ADD_FAILURE() << "transient buffer allocation/lifetime counts should match";
    return;
  }
  if (!(compiled.plan.transientTexturePhysicalAllocations.size() ==
        compiled.plan.transientTexturePhysicalCount)) {
    ADD_FAILURE() << "transient texture physical allocation count should match";
    return;
  }
  if (!(compiled.plan.transientBufferPhysicalAllocations.size() ==
        compiled.plan.transientBufferPhysicalCount)) {
    ADD_FAILURE() << "transient buffer physical allocation count should match";
    return;
  }

  uint32_t previousTextureResourceIndex = UINT32_MAX;
  for (size_t i = 0; i < compiled.plan.transientTextureAllocations.size();
       ++i) {
    const auto &allocation = compiled.plan.transientTextureAllocations[i];
    if (!(allocation.resourceIndex <
          compiled.plan.transientTextureAllocationByResource.size())) {
      ADD_FAILURE()
          << "transient texture allocation resource index should be in "
             "range";
      return;
    }
    if (!(allocation.allocationIndex <
          compiled.plan.transientTexturePhysicalCount)) {
      ADD_FAILURE() << "transient texture allocation slot index should be in "
                       "range";
      return;
    }
    if (i > 0u) {
      if (!(allocation.resourceIndex > previousTextureResourceIndex)) {
        ADD_FAILURE() << "transient texture allocations should be strictly "
                         "ordered by resource index";
        return;
      }
    }
    previousTextureResourceIndex = allocation.resourceIndex;
    if (!(compiled.plan
              .transientTextureAllocationByResource[allocation.resourceIndex] ==
          allocation.allocationIndex)) {
      ADD_FAILURE() << "transient texture allocation should match resource "
                       "allocation map";
      return;
    }
  }

  uint32_t previousBufferResourceIndex = UINT32_MAX;
  for (size_t i = 0; i < compiled.plan.transientBufferAllocations.size(); ++i) {
    const auto &allocation = compiled.plan.transientBufferAllocations[i];
    if (!(allocation.resourceIndex <
          compiled.plan.transientBufferAllocationByResource.size())) {
      ADD_FAILURE()
          << "transient buffer allocation resource index should be in "
             "range";
      return;
    }
    if (!(allocation.allocationIndex <
          compiled.plan.transientBufferPhysicalCount)) {
      ADD_FAILURE() << "transient buffer allocation slot index should be in "
                       "range";
      return;
    }
    if (i > 0u) {
      if (!(allocation.resourceIndex > previousBufferResourceIndex)) {
        ADD_FAILURE() << "transient buffer allocations should be strictly "
                         "ordered by resource index";
        return;
      }
    }
    previousBufferResourceIndex = allocation.resourceIndex;
    if (!(compiled.plan
              .transientBufferAllocationByResource[allocation.resourceIndex] ==
          allocation.allocationIndex)) {
      ADD_FAILURE()
          << "transient buffer allocation should match resource allocation map";
      return;
    }
  }

  std::vector<uint8_t> seenTextureSlots(
      compiled.plan.transientTexturePhysicalCount, 0u);
  for (const auto &allocation :
       compiled.plan.transientTexturePhysicalAllocations) {
    if (!(allocation.allocationIndex < seenTextureSlots.size())) {
      ADD_FAILURE()
          << "transient texture physical allocation slot should be in "
             "range";
      return;
    }
    if (!(seenTextureSlots[allocation.allocationIndex] == 0u)) {
      ADD_FAILURE() << "transient texture physical allocation slots should be "
                       "unique";
      return;
    }
    seenTextureSlots[allocation.allocationIndex] = 1u;

    if (!(allocation.representativeResourceIndex <
          compiled.plan.transientTextureAllocationByResource.size())) {
      ADD_FAILURE() << "transient texture representative resource should be in "
                       "range";
      return;
    }
    if (!(compiled.plan.transientTextureAllocationByResource
              [allocation.representativeResourceIndex] ==
          allocation.allocationIndex)) {
      ADD_FAILURE()
          << "transient texture representative map entry should match "
             "allocation slot";
      return;
    }
  }
  for (const uint8_t seen : seenTextureSlots) {
    if (!(seen == 1u)) {
      ADD_FAILURE()
          << "transient texture physical allocation table should cover "
             "all slots";
      return;
    }
  }

  std::vector<uint8_t> seenBufferSlots(
      compiled.plan.transientBufferPhysicalCount, 0u);
  for (const auto &allocation :
       compiled.plan.transientBufferPhysicalAllocations) {
    if (!(allocation.allocationIndex < seenBufferSlots.size())) {
      ADD_FAILURE() << "transient buffer physical allocation slot should be in "
                       "range";
      return;
    }
    if (!(seenBufferSlots[allocation.allocationIndex] == 0u)) {
      ADD_FAILURE() << "transient buffer physical allocation slots should be "
                       "unique";
      return;
    }
    seenBufferSlots[allocation.allocationIndex] = 1u;

    if (!(allocation.representativeResourceIndex <
          compiled.plan.transientBufferAllocationByResource.size())) {
      ADD_FAILURE() << "transient buffer representative resource should be in "
                       "range";
      return;
    }
    if (!(compiled.plan.transientBufferAllocationByResource
              [allocation.representativeResourceIndex] ==
          allocation.allocationIndex)) {
      ADD_FAILURE() << "transient buffer representative map entry should match "
                       "allocation slot";
      return;
    }
  }
  for (const uint8_t seen : seenBufferSlots) {
    if (!(seen == 1u)) {
      ADD_FAILURE()
          << "transient buffer physical allocation table should cover "
             "all slots";
      return;
    }
  }
}

} // namespace
