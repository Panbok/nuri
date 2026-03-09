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

TEST(RenderGraphMetadataTest, TransientTextureBindingMetadata) {
  RenderGraphBuilder builder;
  builder.beginFrame(104u);

  auto transientColorResult = builder.createTransientTexture(
      makeTransientTextureDesc(Format::RGBA8_UNORM, 16u, 16u),
      "test_color_tex");
  auto transientDepthResult = builder.createTransientTexture(
      makeTransientTextureDesc(Format::D32_FLOAT, 16u, 16u), "test_depth_tex");
  if (transientColorResult.hasError() || transientDepthResult.hasError()) {
    ADD_FAILURE() << "createTransientTexture should succeed";
    return;
  }
  const RenderGraphTextureId transientColor = transientColorResult.value();
  const RenderGraphTextureId transientDepth = transientDepthResult.value();

  RenderPass pass{};
  pass.debugLabel = "rg_test_transient_tex_binding";
  auto addPassResult = addTestGraphicsPass(builder, pass, pass.debugLabel);
  if (addPassResult.hasError()) {
    ADD_FAILURE() << "addLegacyRenderPass should succeed";
    return;
  }
  const RenderGraphPassId passId = addPassResult.value();

  auto bindResult = builder.bindPassColorTexture(passId, transientColor);
  if (bindResult.hasError()) {
    ADD_FAILURE() << "bindPassColorTexture transient should succeed";
    return;
  }
  bindResult = builder.bindPassDepthTexture(passId, transientDepth);
  if (bindResult.hasError()) {
    ADD_FAILURE() << "bindPassDepthTexture transient should succeed";
    return;
  }

  auto compileResult = builder.compile();
  if (compileResult.hasError()) {
    ADD_FAILURE() << "compile should succeed";
    if (compileResult.hasError()) {
      std::cerr << compileResult.error() << "\n";
    }
    return;
  }
  const RenderGraphCompileResult &compiled = compileResult.value();

  if (!(compiled.orderedPasses.size() == 1u)) {
    ADD_FAILURE() << "expected one ordered pass";
    return;
  }
  if (!(compiled.unresolvedTextureBindings.size() == 2u)) {
    ADD_FAILURE() << "expected unresolved color and depth texture bindings";
    return;
  }

  bool sawColorBinding = false;
  bool sawDepthBinding = false;
  for (const auto &binding : compiled.unresolvedTextureBindings) {
    if (!(binding.orderedPassIndex == 0u)) {
      ADD_FAILURE() << "unexpected unresolved texture pass index";
      return;
    }
    if (binding.target ==
        RenderGraphCompileResult::PassTextureBindingTarget::Color) {
      if (!(binding.textureResourceIndex == transientColor.value)) {
        ADD_FAILURE() << "unexpected unresolved color texture resource index";
        return;
      }
      sawColorBinding = true;
    } else if (binding.target ==
               RenderGraphCompileResult::PassTextureBindingTarget::Depth) {
      if (!(binding.textureResourceIndex == transientDepth.value)) {
        ADD_FAILURE() << "unexpected unresolved depth texture resource index";
        return;
      }
      sawDepthBinding = true;
    } else {
      ADD_FAILURE() << "unexpected unresolved texture binding target";
      return;
    }
  }

  if (!(sawColorBinding && sawDepthBinding)) {
    ADD_FAILURE() << "expected both color and depth unresolved texture "
                     "bindings";
    return;
  }
}

TEST(RenderGraphMetadataTest, ImportedTextureBindingMetadataResolved) {
  RenderGraphBuilder builder;
  builder.beginFrame(105u);

  const TextureHandle importedColor{.index = 33u, .generation = 1u};
  const TextureHandle importedDepth{.index = 34u, .generation = 1u};

  auto importedColorResult =
      builder.importTexture(importedColor, "test_imported_color");
  auto importedDepthResult =
      builder.importTexture(importedDepth, "test_imported_depth");
  if (importedColorResult.hasError() || importedDepthResult.hasError()) {
    ADD_FAILURE() << "importTexture should succeed";
    return;
  }

  RenderPass pass{};
  pass.debugLabel = "rg_test_imported_tex_binding";
  auto addPassResult = addTestGraphicsPass(builder, pass, pass.debugLabel);
  if (addPassResult.hasError()) {
    ADD_FAILURE() << "addLegacyRenderPass should succeed";
    return;
  }
  const RenderGraphPassId passId = addPassResult.value();

  auto bindResult =
      builder.bindPassColorTexture(passId, importedColorResult.value());
  if (bindResult.hasError()) {
    ADD_FAILURE() << "bindPassColorTexture imported should succeed";
    return;
  }
  bindResult =
      builder.bindPassDepthTexture(passId, importedDepthResult.value());
  if (bindResult.hasError()) {
    ADD_FAILURE() << "bindPassDepthTexture imported should succeed";
    return;
  }

  auto compileResult = builder.compile();
  if (compileResult.hasError()) {
    ADD_FAILURE() << "compile should succeed";
    if (compileResult.hasError()) {
      std::cerr << compileResult.error() << "\n";
    }
    return;
  }
  const RenderGraphCompileResult &compiled = compileResult.value();

  if (!(compiled.unresolvedTextureBindings.empty())) {
    ADD_FAILURE() << "imported texture bindings should resolve at compile";
    return;
  }
  if (!(compiled.orderedPasses.size() == 1u)) {
    ADD_FAILURE() << "expected one ordered pass";
    return;
  }
  if (!(sameTextureHandle(compiled.orderedPasses[0u].colorTexture,
                          importedColor))) {
    ADD_FAILURE() << "resolved color texture handle mismatch";
    return;
  }
  if (!(sameTextureHandle(compiled.orderedPasses[0u].depthTexture,
                          importedDepth))) {
    ADD_FAILURE() << "resolved depth texture handle mismatch";
    return;
  }
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

TEST(RenderGraphMetadataTest, ExplicitLegacyRegistrationWithoutInference) {
  RenderGraphBuilder builder;
  builder.beginFrame(309u);

  const TextureHandle colorTexture{.index = 210u, .generation = 3u};
  const TextureHandle depthTexture{.index = 211u, .generation = 3u};
  const BufferHandle dependencyBuffer{.index = 310u, .generation = 9u};
  const BufferHandle vertexBuffer{.index = 311u, .generation = 9u};
  const BufferHandle indexBuffer{.index = 312u, .generation = 9u};

  std::array<BufferHandle, 1> dependencyBuffers = {dependencyBuffer};
  DrawItem draw{};
  draw.vertexBuffer = vertexBuffer;
  draw.indexBuffer = indexBuffer;
  std::array<DrawItem, 1> draws = {draw};

  RenderPass pass{};
  pass.debugLabel = "metadata_explicit_legacy_registration";
  pass.colorTexture = colorTexture;
  pass.depthTexture = depthTexture;
  pass.dependencyBuffers = std::span<const BufferHandle>(
      dependencyBuffers.data(), dependencyBuffers.size());
  pass.draws = std::span<const DrawItem>(draws.data(), draws.size());

  auto addResult = addTestGraphicsPass(builder, pass, pass.debugLabel, false);
  if (addResult.hasError()) {
    ADD_FAILURE() << "addLegacyRenderPass explicit path should succeed";
    if (addResult.hasError()) {
      std::cerr << addResult.error() << "\n";
    }
    return;
  }
  const RenderGraphPassId passId = addResult.value();

  auto colorImportResult =
      builder.importTexture(colorTexture, "metadata_explicit_color");
  auto depthImportResult =
      builder.importTexture(depthTexture, "metadata_explicit_depth");
  auto depImportResult =
      builder.importBuffer(dependencyBuffer, "metadata_explicit_dep");
  auto vbImportResult =
      builder.importBuffer(vertexBuffer, "metadata_explicit_vb");
  auto ibImportResult =
      builder.importBuffer(indexBuffer, "metadata_explicit_ib");
  if (colorImportResult.hasError() || depthImportResult.hasError() ||
      depImportResult.hasError() || vbImportResult.hasError() ||
      ibImportResult.hasError()) {
    ADD_FAILURE() << "explicit registration imports should succeed";
    return;
  }

  auto bindResult =
      builder.bindPassColorTexture(passId, colorImportResult.value());
  if (bindResult.hasError()) {
    ADD_FAILURE() << "bindPassColorTexture explicit path should succeed";
    return;
  }
  bindResult = builder.bindPassDepthTexture(passId, depthImportResult.value());
  if (bindResult.hasError()) {
    ADD_FAILURE() << "bindPassDepthTexture explicit path should succeed";
    return;
  }
  bindResult = builder.bindPassDependencyBuffer(
      passId, 0u, depImportResult.value(),
      RenderGraphAccessMode::Read | RenderGraphAccessMode::Write);
  if (bindResult.hasError()) {
    ADD_FAILURE() << "bindPassDependencyBuffer explicit path should succeed";
    return;
  }
  bindResult = builder.bindDrawBuffer(
      passId, 0u, RenderGraphCompileResult::DrawBufferBindingTarget::Vertex,
      vbImportResult.value(), RenderGraphAccessMode::Read);
  if (bindResult.hasError()) {
    ADD_FAILURE() << "bindDrawBuffer vertex explicit path should succeed";
    return;
  }
  bindResult = builder.bindDrawBuffer(
      passId, 0u, RenderGraphCompileResult::DrawBufferBindingTarget::Index,
      ibImportResult.value(), RenderGraphAccessMode::Read);
  if (bindResult.hasError()) {
    ADD_FAILURE() << "bindDrawBuffer index explicit path should succeed";
    return;
  }

  auto compileResult = builder.compile();
  if (compileResult.hasError()) {
    ADD_FAILURE() << "compile should succeed for explicit registration path";
    if (compileResult.hasError()) {
      std::cerr << compileResult.error() << "\n";
    }
    return;
  }
  const RenderGraphCompileResult &compiled = compileResult.value();

  if (!(compiled.orderedPasses.size() == 1u)) {
    ADD_FAILURE()
        << "explicit no-inference registration path should schedule one "
           "pass";
    return;
  }
  if (!(compiled.unresolvedTextureBindings.empty())) {
    ADD_FAILURE()
        << "explicit no-inference registration path should resolve imported "
           "attachment bindings";
    return;
  }
  if (!(sameTextureHandle(compiled.orderedPasses[0u].colorTexture,
                          colorTexture))) {
    ADD_FAILURE()
        << "explicit no-inference registration color texture mismatch";
    return;
  }
  if (!(sameTextureHandle(compiled.orderedPasses[0u].depthTexture,
                          depthTexture))) {
    ADD_FAILURE()
        << "explicit no-inference registration depth texture mismatch";
    return;
  }
  if (!(compiled.dependencyBufferRangesByPass.size() == 1u)) {
    ADD_FAILURE()
        << "explicit no-inference registration dependency range table size "
           "mismatch";
    return;
  }
  const auto &range = compiled.dependencyBufferRangesByPass[0u];
  if (!(range.count == 1u)) {
    ADD_FAILURE()
        << "explicit no-inference registration dependency count mismatch";
    return;
  }
  if (!(range.offset < compiled.resolvedDependencyBuffers.size())) {
    ADD_FAILURE()
        << "explicit no-inference registration dependency offset out of "
           "range";
    return;
  }
  if (!(sameHandle(compiled.resolvedDependencyBuffers[range.offset],
                   dependencyBuffer))) {
    ADD_FAILURE()
        << "explicit no-inference registration dependency handle mismatch";
    return;
  }

  if (!(compiled.orderedPasses[0u].draws.size() == 1u)) {
    ADD_FAILURE() << "explicit no-inference registration draw count mismatch";
    return;
  }
  const DrawItem &resolvedDraw = compiled.orderedPasses[0u].draws[0u];
  if (!(sameHandle(resolvedDraw.vertexBuffer, vertexBuffer))) {
    ADD_FAILURE()
        << "explicit no-inference registration vertex buffer mismatch";
    return;
  }
  if (!(sameHandle(resolvedDraw.indexBuffer, indexBuffer))) {
    ADD_FAILURE() << "explicit no-inference registration index buffer mismatch";
    return;
  }
}

TEST(RenderGraphMetadataTest, AccessAndSideEffectDedupStateResetsAcrossFrames) {
  RenderGraphBuilder builder;
  builder.beginFrame(307u);

  auto transientBufferResultA = builder.createTransientBuffer(
      makeTransientBufferDesc(64u), "frame_a_buf");
  if (transientBufferResultA.hasError()) {
    ADD_FAILURE() << "frame A createTransientBuffer should succeed";
    return;
  }

  RenderPass passA{};
  passA.debugLabel = "frame_a_pass";
  auto passAResult = addTestGraphicsPass(builder, passA, passA.debugLabel);
  if (passAResult.hasError()) {
    ADD_FAILURE() << "frame A addLegacyRenderPass should succeed";
    return;
  }

  auto accessResult = builder.addBufferRead(passAResult.value(),
                                            transientBufferResultA.value());
  if (accessResult.hasError()) {
    ADD_FAILURE() << "frame A first addBufferRead should succeed";
    return;
  }
  accessResult = builder.addBufferRead(passAResult.value(),
                                       transientBufferResultA.value());
  if (accessResult.hasError()) {
    ADD_FAILURE() << "frame A duplicate addBufferRead should succeed";
    return;
  }

  auto sideEffectResult = builder.markPassSideEffect(passAResult.value());
  if (sideEffectResult.hasError()) {
    ADD_FAILURE() << "frame A first markPassSideEffect should succeed";
    return;
  }
  sideEffectResult = builder.markPassSideEffect(passAResult.value());
  if (sideEffectResult.hasError()) {
    ADD_FAILURE() << "frame A duplicate markPassSideEffect should succeed";
    return;
  }

  auto compileAResult = builder.compile();
  if (compileAResult.hasError()) {
    ADD_FAILURE() << "frame A compile should succeed";
    if (compileAResult.hasError()) {
      std::cerr << compileAResult.error() << "\n";
    }
    return;
  }

  builder.beginFrame(308u);
  auto transientBufferResultB = builder.createTransientBuffer(
      makeTransientBufferDesc(64u), "frame_b_buf");
  if (transientBufferResultB.hasError()) {
    ADD_FAILURE() << "frame B createTransientBuffer should succeed";
    return;
  }

  RenderPass passB{};
  passB.debugLabel = "frame_b_pass";
  auto passBResult = addTestGraphicsPass(builder, passB, passB.debugLabel);
  if (passBResult.hasError()) {
    ADD_FAILURE() << "frame B addLegacyRenderPass should succeed";
    return;
  }

  accessResult = builder.addBufferRead(passBResult.value(),
                                       transientBufferResultB.value());
  if (accessResult.hasError()) {
    ADD_FAILURE() << "frame B addBufferRead should succeed";
    return;
  }
  sideEffectResult = builder.markPassSideEffect(passBResult.value());
  if (sideEffectResult.hasError()) {
    ADD_FAILURE() << "frame B markPassSideEffect should succeed";
    return;
  }

  auto compileBResult = builder.compile();
  if (compileBResult.hasError()) {
    ADD_FAILURE() << "frame B compile should succeed";
    if (compileBResult.hasError()) {
      std::cerr << compileBResult.error() << "\n";
    }
    return;
  }

  const RenderGraphCompileResult &compiledB = compileBResult.value();
  if (!(compiledB.orderedPassIndices.size() == 1u)) {
    ADD_FAILURE() << "frame B should contain exactly one pass";
    return;
  }
  if (!(compiledB.rootPassCount == 1u)) {
    ADD_FAILURE() << "frame B should preserve side-effect root marking";
    return;
  }
}

TEST(RenderGraphMetadataTest, UnresolvedTransientBindingMetadata) {
  RenderGraphBuilder builder;
  builder.beginFrame(101u);

  auto transientResult =
      builder.createTransientBuffer(makeTransientBufferDesc(64u), "test_buf");
  if (transientResult.hasError()) {
    ADD_FAILURE() << "createTransientBuffer should succeed";
    return;
  }
  const RenderGraphBufferId transientBuffer = transientResult.value();

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
  pass.debugLabel = "rg_test_unresolved";

  auto addPassResult = addTestGraphicsPass(builder, pass, pass.debugLabel);
  if (addPassResult.hasError()) {
    ADD_FAILURE() << "addLegacyRenderPass should succeed";
    return;
  }
  const RenderGraphPassId passId = addPassResult.value();

  auto bindPassDepResult = builder.bindPassDependencyBuffer(
      passId, 0u, transientBuffer,
      RenderGraphAccessMode::Read | RenderGraphAccessMode::Write);
  if (bindPassDepResult.hasError()) {
    ADD_FAILURE() << "bindPassDependencyBuffer should succeed";
    return;
  }

  auto bindPreDispatchDepResult = builder.bindPreDispatchDependencyBuffer(
      passId, 0u, 0u, transientBuffer,
      RenderGraphAccessMode::Read | RenderGraphAccessMode::Write);
  if (bindPreDispatchDepResult.hasError()) {
    ADD_FAILURE() << "bindPreDispatchDependencyBuffer should succeed";
    return;
  }

  auto bindDrawBufferResult = builder.bindDrawBuffer(
      passId, 0u, RenderGraphCompileResult::DrawBufferBindingTarget::Vertex,
      transientBuffer, RenderGraphAccessMode::Read);
  if (bindDrawBufferResult.hasError()) {
    ADD_FAILURE() << "bindDrawBuffer should succeed";
    return;
  }

  auto compileResult = builder.compile();
  if (compileResult.hasError()) {
    ADD_FAILURE() << "compile should succeed";
    if (compileResult.hasError()) {
      std::cerr << compileResult.error() << "\n";
    }
    return;
  }
  const RenderGraphCompileResult &compiled = compileResult.value();

  if (!(compiled.orderedPasses.size() == 1u)) {
    ADD_FAILURE() << "expected one ordered pass";
    return;
  }
  if (!(compiled.dependencyBufferRangesByPass.size() == 1u)) {
    ADD_FAILURE() << "dependency range count mismatch";
    return;
  }
  if (!(compiled.preDispatchRangesByPass.size() == 1u)) {
    ADD_FAILURE() << "pre-dispatch range count mismatch";
    return;
  }
  if (!(compiled.drawRangesByPass.size() == 1u)) {
    ADD_FAILURE() << "draw range count mismatch";
    return;
  }
  if (!(compiled.preDispatchDependencyRanges.size() == 1u)) {
    ADD_FAILURE() << "pre-dispatch dependency range count mismatch";
    return;
  }
  if (!(compiled.unresolvedDependencyBufferBindings.size() == 1u)) {
    ADD_FAILURE() << "expected one unresolved pass dependency buffer binding";
    return;
  }
  if (!(compiled.unresolvedPreDispatchDependencyBufferBindings.size() == 1u)) {
    ADD_FAILURE()
        << "expected one unresolved pre-dispatch dependency buffer binding";
    return;
  }
  if (!(compiled.unresolvedDrawBufferBindings.size() == 1u)) {
    ADD_FAILURE() << "expected one unresolved draw buffer binding";
    return;
  }

  const auto depRange = compiled.dependencyBufferRangesByPass[0u];
  if (!(depRange.count == 1u)) {
    ADD_FAILURE() << "expected pass dependency count = 1";
    return;
  }

  const auto preDispatchRange = compiled.preDispatchRangesByPass[0u];
  if (!(preDispatchRange.count == 1u)) {
    ADD_FAILURE() << "expected pre-dispatch count = 1";
    return;
  }

  const auto drawRange = compiled.drawRangesByPass[0u];
  if (!(drawRange.count == 1u)) {
    ADD_FAILURE() << "expected draw count = 1";
    return;
  }

  const auto unresolvedDep = compiled.unresolvedDependencyBufferBindings[0u];
  if (!(unresolvedDep.orderedPassIndex == 0u)) {
    ADD_FAILURE() << "unexpected unresolved dependency pass index";
    return;
  }
  if (!(unresolvedDep.dependencyBufferIndex == 0u)) {
    ADD_FAILURE() << "unexpected unresolved dependency slot index";
    return;
  }
  if (!(unresolvedDep.bufferResourceIndex == transientBuffer.value)) {
    ADD_FAILURE() << "unexpected unresolved dependency resource index";
    return;
  }

  const auto unresolvedPreDep =
      compiled.unresolvedPreDispatchDependencyBufferBindings[0u];
  if (!(unresolvedPreDep.orderedPassIndex == 0u)) {
    ADD_FAILURE() << "unexpected unresolved pre-dispatch pass index";
    return;
  }
  if (!(unresolvedPreDep.preDispatchIndex == 0u)) {
    ADD_FAILURE() << "unexpected unresolved pre-dispatch index";
    return;
  }
  if (!(unresolvedPreDep.dependencyBufferIndex == 0u)) {
    ADD_FAILURE() << "unexpected unresolved pre-dispatch dependency slot";
    return;
  }
  if (!(unresolvedPreDep.bufferResourceIndex == transientBuffer.value)) {
    ADD_FAILURE() << "unexpected unresolved pre-dispatch resource index";
    return;
  }

  const auto unresolvedDraw = compiled.unresolvedDrawBufferBindings[0u];
  if (!(unresolvedDraw.orderedPassIndex == 0u)) {
    ADD_FAILURE() << "unexpected unresolved draw pass index";
    return;
  }
  if (!(unresolvedDraw.drawIndex == 0u)) {
    ADD_FAILURE() << "unexpected unresolved draw index";
    return;
  }
  if (!(unresolvedDraw.target ==
        RenderGraphCompileResult::DrawBufferBindingTarget::Vertex)) {
    ADD_FAILURE() << "unexpected unresolved draw target";
    return;
  }
  if (!(unresolvedDraw.bufferResourceIndex == transientBuffer.value)) {
    ADD_FAILURE() << "unexpected unresolved draw resource index";
    return;
  }
}

TEST(RenderGraphMetadataTest, ResolvedImportedBindingMetadata) {
  RenderGraphBuilder builder;
  builder.beginFrame(102u);

  const BufferHandle importedHandle{.index = 17u, .generation = 1u};
  std::array<BufferHandle, 1> passDeps = {importedHandle};
  std::array<BufferHandle, 1> dispatchDeps = {importedHandle};
  ComputeDispatchItem dispatch{};
  dispatch.dependencyBuffers =
      std::span<const BufferHandle>(dispatchDeps.data(), dispatchDeps.size());
  std::array<ComputeDispatchItem, 1> preDispatches = {dispatch};
  DrawItem draw{};
  draw.vertexBuffer = importedHandle;
  std::array<DrawItem, 1> draws = {draw};

  RenderPass pass{};
  pass.dependencyBuffers =
      std::span<const BufferHandle>(passDeps.data(), passDeps.size());
  pass.preDispatches = std::span<const ComputeDispatchItem>(
      preDispatches.data(), preDispatches.size());
  pass.draws = std::span<const DrawItem>(draws.data(), draws.size());
  pass.debugLabel = "rg_test_resolved";

  auto addPassResult = addTestGraphicsPass(builder, pass, pass.debugLabel);
  if (addPassResult.hasError()) {
    ADD_FAILURE() << "addLegacyRenderPass should succeed";
    return;
  }

  auto compileResult = builder.compile();
  if (compileResult.hasError()) {
    ADD_FAILURE() << "compile should succeed";
    if (compileResult.hasError()) {
      std::cerr << compileResult.error() << "\n";
    }
    return;
  }
  const RenderGraphCompileResult &compiled = compileResult.value();

  if (!(compiled.unresolvedDependencyBufferBindings.empty())) {
    ADD_FAILURE() << "expected no unresolved pass dependency bindings";
    return;
  }
  if (!(compiled.unresolvedPreDispatchDependencyBufferBindings.empty())) {
    ADD_FAILURE() << "expected no unresolved pre-dispatch bindings";
    return;
  }
  if (!(compiled.unresolvedDrawBufferBindings.empty())) {
    ADD_FAILURE() << "expected no unresolved draw bindings";
    return;
  }
  if (!(compiled.resolvedDependencyBuffers.size() == 1u)) {
    ADD_FAILURE() << "expected one resolved pass dependency buffer";
    return;
  }
  if (!(compiled.resolvedPreDispatchDependencyBuffers.size() == 1u)) {
    ADD_FAILURE() << "expected one resolved pre-dispatch dependency buffer";
    return;
  }

  if (!(sameHandle(compiled.resolvedDependencyBuffers[0u], importedHandle))) {
    ADD_FAILURE() << "resolved pass dependency handle mismatch";
    return;
  }
  if (!(sameHandle(compiled.resolvedPreDispatchDependencyBuffers[0u],
                   importedHandle))) {
    ADD_FAILURE() << "resolved pre-dispatch dependency handle mismatch";
    return;
  }
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
      pass0Id, 0u, RenderGraphCompileResult::DrawBufferBindingTarget::Vertex,
      transientA, RenderGraphAccessMode::Read);
  if (bindDrawResult.hasError()) {
    ADD_FAILURE() << "bindDrawBuffer pass0 draw0 should succeed";
    return;
  }
  bindDrawResult = builder.bindDrawBuffer(
      pass0Id, 1u, RenderGraphCompileResult::DrawBufferBindingTarget::Index,
      transientB, RenderGraphAccessMode::Read);
  if (bindDrawResult.hasError()) {
    ADD_FAILURE() << "bindDrawBuffer pass0 draw1 should succeed";
    return;
  }

  auto compileResult = builder.compile();
  if (compileResult.hasError()) {
    ADD_FAILURE() << "compile should succeed";
    if (compileResult.hasError()) {
      std::cerr << compileResult.error() << "\n";
    }
    return;
  }
  const RenderGraphCompileResult &compiled = compileResult.value();

  if (!(compiled.orderedPasses.size() == 2u)) {
    ADD_FAILURE() << "expected two ordered passes";
    return;
  }
  if (!(compiled.dependencyBufferRangesByPass.size() == 2u)) {
    ADD_FAILURE() << "dependency range table should have two entries";
    return;
  }
  if (!(compiled.preDispatchRangesByPass.size() == 2u)) {
    ADD_FAILURE() << "pre-dispatch range table should have two entries";
    return;
  }
  if (!(compiled.drawRangesByPass.size() == 2u)) {
    ADD_FAILURE() << "draw range table should have two entries";
    return;
  }

  const auto pass0DepRange = compiled.dependencyBufferRangesByPass[0u];
  const auto pass1DepRange = compiled.dependencyBufferRangesByPass[1u];
  if (!(pass0DepRange.count == 2u)) {
    ADD_FAILURE() << "pass0 dependency range should have two slots";
    return;
  }
  if (!(pass1DepRange.count == 0u)) {
    ADD_FAILURE() << "pass1 dependency range should be empty";
    return;
  }

  const auto pass0PreRange = compiled.preDispatchRangesByPass[0u];
  const auto pass1PreRange = compiled.preDispatchRangesByPass[1u];
  if (!(pass0PreRange.count == 2u)) {
    ADD_FAILURE() << "pass0 pre-dispatch range should have two items";
    return;
  }
  if (!(pass1PreRange.count == 0u)) {
    ADD_FAILURE() << "pass1 pre-dispatch range should be empty";
    return;
  }

  const auto pass0DrawRange = compiled.drawRangesByPass[0u];
  const auto pass1DrawRange = compiled.drawRangesByPass[1u];
  if (!(pass0DrawRange.count == 2u)) {
    ADD_FAILURE() << "pass0 draw range should have two draws";
    return;
  }
  if (!(pass1DrawRange.count == 0u)) {
    ADD_FAILURE() << "pass1 draw range should be empty";
    return;
  }

  if (!(compiled.unresolvedDependencyBufferBindings.size() == 2u)) {
    ADD_FAILURE() << "expected two unresolved pass dependency bindings";
    return;
  }
  if (!(compiled.unresolvedPreDispatchDependencyBufferBindings.size() == 2u)) {
    ADD_FAILURE() << "expected two unresolved pre-dispatch dependency bindings";
    return;
  }
  if (!(compiled.unresolvedDrawBufferBindings.size() == 2u)) {
    ADD_FAILURE() << "expected two unresolved draw bindings";
    return;
  }

  for (const auto &binding : compiled.unresolvedDependencyBufferBindings) {
    if (!(binding.orderedPassIndex == 0u)) {
      ADD_FAILURE() << "unresolved dependency should reference pass0";
      return;
    }
    if (!(binding.dependencyBufferIndex < pass0DepRange.count)) {
      ADD_FAILURE() << "unresolved dependency slot should be in pass0 range";
      return;
    }
  }
  for (const auto &binding :
       compiled.unresolvedPreDispatchDependencyBufferBindings) {
    if (!(binding.orderedPassIndex == 0u)) {
      ADD_FAILURE() << "unresolved pre-dispatch should reference pass0";
      return;
    }
    if (!(binding.preDispatchIndex < pass0PreRange.count)) {
      ADD_FAILURE() << "unresolved pre-dispatch index should be in pass0 range";
      return;
    }
  }
  for (const auto &binding : compiled.unresolvedDrawBufferBindings) {
    if (!(binding.orderedPassIndex == 0u)) {
      ADD_FAILURE() << "unresolved draw should reference pass0";
      return;
    }
    if (!(binding.drawIndex < pass0DrawRange.count)) {
      ADD_FAILURE() << "unresolved draw index should be in pass0 range";
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

  auto compileResult = builder.compile();
  if (compileResult.hasError()) {
    ADD_FAILURE() << "compile should succeed";
    if (compileResult.hasError()) {
      std::cerr << compileResult.error() << "\n";
    }
    return;
  }
  const RenderGraphCompileResult &compiled = compileResult.value();

  if (!(compiled.declaredPassCount == 3u)) {
    ADD_FAILURE() << "structural test should declare three passes";
    return;
  }
  if (!(compiled.culledPassCount == 1u)) {
    ADD_FAILURE() << "structural test should cull one implicit pass";
    return;
  }
  if (!(compiled.orderedPasses.size() == compiled.orderedPassIndices.size())) {
    ADD_FAILURE() << "ordered pass tables should have matching sizes";
    return;
  }
  if (!(compiled.passDebugNames.size() == compiled.declaredPassCount)) {
    ADD_FAILURE() << "pass debug-name table should match declared pass count";
    return;
  }
  if (!(compiled.orderedPasses.size() + compiled.culledPassCount ==
        compiled.declaredPassCount)) {
    ADD_FAILURE() << "ordered + culled pass counts should match declared pass "
                     "count";
    return;
  }

  std::array<bool, 3> seenPass{false, false, false};
  for (const uint32_t passIndex : compiled.orderedPassIndices) {
    if (!(passIndex < compiled.declaredPassCount)) {
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

  auto compileResult = builder.compile();
  if (compileResult.hasError()) {
    ADD_FAILURE() << "compile should succeed";
    if (compileResult.hasError()) {
      std::cerr << compileResult.error() << "\n";
    }
    return;
  }
  const RenderGraphCompileResult &compiled = compileResult.value();

  if (!(compiled.transientTextureAllocations.size() ==
        compiled.transientTextureLifetimes.size())) {
    ADD_FAILURE()
        << "transient texture allocation/lifetime counts should match";
    return;
  }
  if (!(compiled.transientBufferAllocations.size() ==
        compiled.transientBufferLifetimes.size())) {
    ADD_FAILURE() << "transient buffer allocation/lifetime counts should match";
    return;
  }
  if (!(compiled.transientTexturePhysicalAllocations.size() ==
        compiled.transientTexturePhysicalCount)) {
    ADD_FAILURE() << "transient texture physical allocation count should match";
    return;
  }
  if (!(compiled.transientBufferPhysicalAllocations.size() ==
        compiled.transientBufferPhysicalCount)) {
    ADD_FAILURE() << "transient buffer physical allocation count should match";
    return;
  }

  uint32_t previousTextureResourceIndex = UINT32_MAX;
  for (size_t i = 0; i < compiled.transientTextureAllocations.size(); ++i) {
    const auto &allocation = compiled.transientTextureAllocations[i];
    if (!(allocation.resourceIndex <
          compiled.transientTextureAllocationByResource.size())) {
      ADD_FAILURE()
          << "transient texture allocation resource index should be in "
             "range";
      return;
    }
    if (!(allocation.allocationIndex <
          compiled.transientTexturePhysicalCount)) {
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
    if (!(compiled
              .transientTextureAllocationByResource[allocation.resourceIndex] ==
          allocation.allocationIndex)) {
      ADD_FAILURE() << "transient texture allocation should match resource "
                       "allocation map";
      return;
    }
  }

  uint32_t previousBufferResourceIndex = UINT32_MAX;
  for (size_t i = 0; i < compiled.transientBufferAllocations.size(); ++i) {
    const auto &allocation = compiled.transientBufferAllocations[i];
    if (!(allocation.resourceIndex <
          compiled.transientBufferAllocationByResource.size())) {
      ADD_FAILURE()
          << "transient buffer allocation resource index should be in "
             "range";
      return;
    }
    if (!(allocation.allocationIndex < compiled.transientBufferPhysicalCount)) {
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
    if (!(compiled
              .transientBufferAllocationByResource[allocation.resourceIndex] ==
          allocation.allocationIndex)) {
      ADD_FAILURE()
          << "transient buffer allocation should match resource allocation map";
      return;
    }
  }

  std::vector<uint8_t> seenTextureSlots(compiled.transientTexturePhysicalCount,
                                        0u);
  for (const auto &allocation : compiled.transientTexturePhysicalAllocations) {
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
          compiled.transientTextureAllocationByResource.size())) {
      ADD_FAILURE() << "transient texture representative resource should be in "
                       "range";
      return;
    }
    if (!(compiled.transientTextureAllocationByResource
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

  std::vector<uint8_t> seenBufferSlots(compiled.transientBufferPhysicalCount,
                                       0u);
  for (const auto &allocation : compiled.transientBufferPhysicalAllocations) {
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
          compiled.transientBufferAllocationByResource.size())) {
      ADD_FAILURE() << "transient buffer representative resource should be in "
                       "range";
      return;
    }
    if (!(compiled.transientBufferAllocationByResource
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
