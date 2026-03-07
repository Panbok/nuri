#include "nuri/pch.h"

#include "render_graph_test_support.h"

#include <gtest/gtest.h>

#include "nuri/core/layer.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>

namespace {

using namespace nuri;
using namespace nuri::test_support;

TEST(RenderGraphCompileBehaviorTest, CompileDeterminismAndTieBreak) {
  RenderGraphBuilder builder;
  builder.beginFrame(201u);

  auto pass0Result =
      addTestGraphicsPass(builder, makeTestPass("det_p0"), "det_p0");
  auto pass1Result =
      addTestGraphicsPass(builder, makeTestPass("det_p1"), "det_p1");
  auto pass2Result =
      addTestGraphicsPass(builder, makeTestPass("det_p2"), "det_p2");
  if (!(!pass0Result.hasError() && !pass1Result.hasError() &&
        !pass2Result.hasError())) {
    ADD_FAILURE() << "addLegacyRenderPass should succeed for determinism graph";
    return;
  }

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

  auto compileAResult = builder.compile();
  auto compileBResult = builder.compile();
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
     DuplicateExplicitDependenciesAreDeduplicated) {
  RenderGraphBuilder builder;
  builder.beginFrame(216u);

  auto pass0Result = addTestGraphicsPass(
      builder, makeTestPass("dedup_explicit_p0"), "dedup_explicit_p0");
  auto pass1Result = addTestGraphicsPass(
      builder, makeTestPass("dedup_explicit_p1"), "dedup_explicit_p1");
  if (!(!pass0Result.hasError() && !pass1Result.hasError())) {
    ADD_FAILURE()
        << "addLegacyRenderPass should succeed for explicit dependency "
           "dedup graph";
    return;
  }

  auto depResult =
      builder.addDependency(pass0Result.value(), pass1Result.value());
  if (!(!depResult.hasError())) {
    ADD_FAILURE() << "first addDependency should succeed";
    return;
  }
  depResult = builder.addDependency(pass0Result.value(), pass1Result.value());
  if (!(!depResult.hasError())) {
    ADD_FAILURE() << "duplicate addDependency should succeed";
    return;
  }
  depResult = builder.addDependency(pass0Result.value(), pass1Result.value());
  if (!(!depResult.hasError())) {
    ADD_FAILURE() << "second duplicate addDependency should succeed";
    return;
  }

  auto compileResult = builder.compile();
  if (!(!compileResult.hasError())) {
    ADD_FAILURE()
        << "compile should succeed for explicit dependency dedup graph";
    if (compileResult.hasError()) {
      std::cerr << compileResult.error() << "\n";
    }
    return;
  }
  const RenderGraphCompileResult &compiled = compileResult.value();

  if (!(compiled.orderedPassIndices.size() == 2u)) {
    ADD_FAILURE() << "dedup graph should schedule two passes";
    return;
  }
  if (!(compiled.edges.size() == 1u)) {
    ADD_FAILURE()
        << "duplicate explicit dependencies should collapse to one edge";
    return;
  }
  if (!(compiled.edges[0u].before == pass0Result.value().value &&
        compiled.edges[0u].after == pass1Result.value().value)) {
    ADD_FAILURE()
        << "deduped edge should preserve explicit dependency direction";
    return;
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

  auto compileResult = builder.compile();
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

  auto compileA = builder.compile();
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

  auto compileB = builder.compile();
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

TEST(RenderGraphCompileBehaviorTest,
     GraphicsPassWithImportedColorBindingCompiles) {
  RenderGraphBuilder builder;
  builder.beginFrame(220u);

  const TextureHandle colorTexture{.index = 902u, .generation = 1u};
  RenderPass pass{};
  pass.debugLabel = "missing_explicit_color_binding";
  pass.colorTexture = colorTexture;

  auto addResult = addTestGraphicsPass(builder, pass, pass.debugLabel);
  if (!(!addResult.hasError())) {
    ADD_FAILURE() << "addTestGraphicsPass color-binding case should succeed";
    return;
  }

  auto compileResult = builder.compile();
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed for color-binding case";
    return;
  }
}

TEST(RenderGraphCompileBehaviorTest,
     GraphicsPassWithDependencyBindingCompiles) {
  RenderGraphBuilder builder;
  builder.beginFrame(221u);

  const BufferHandle dependencyBuffer{.index = 903u, .generation = 1u};
  std::array<BufferHandle, 1> dependencies = {dependencyBuffer};

  RenderPass pass{};
  pass.debugLabel = "missing_explicit_dependency_binding";
  pass.dependencyBuffers =
      std::span<const BufferHandle>(dependencies.data(), dependencies.size());

  auto addResult = addTestGraphicsPass(builder, pass, pass.debugLabel);
  if (!(!addResult.hasError())) {
    ADD_FAILURE() << "addTestGraphicsPass dependency case should "
                     "succeed";
    return;
  }

  auto compileResult = builder.compile();
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed for dependency-binding case";
    return;
  }
}

TEST(RenderGraphCompileBehaviorTest, AddGraphicsPassNativeDeclarationPath) {
  RenderGraphBuilder builder;
  builder.beginFrame(222u);

  auto colorResult = builder.createTransientTexture(
      makeTransientTextureDesc(Format::RGBA8_UNORM, 32u, 32u),
      "native_graphics_color");
  auto depthResult = builder.createTransientTexture(
      makeTransientTextureDesc(Format::D32_FLOAT, 32u, 32u),
      "native_graphics_depth");
  if (!(!colorResult.hasError() && !depthResult.hasError())) {
    ADD_FAILURE()
        << "createTransientTexture should succeed for native graphics "
           "pass";
    return;
  }

  const BufferHandle dependencyBuffer{.index = 904u, .generation = 1u};
  std::array<BufferHandle, 1u> dependencies = {dependencyBuffer};

  DrawItem draw{};
  draw.vertexBuffer = BufferHandle{.index = 905u, .generation = 1u};
  std::array<DrawItem, 1u> draws = {draw};

  RenderGraphGraphicsPassDesc passDesc{};
  passDesc.color = {.loadOp = LoadOp::Clear,
                    .storeOp = StoreOp::Store,
                    .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}};
  passDesc.colorTexture = colorResult.value();
  passDesc.depth = {.loadOp = LoadOp::Clear,
                    .storeOp = StoreOp::Store,
                    .clearDepth = 1.0f,
                    .clearStencil = 0};
  passDesc.depthTexture = depthResult.value();
  passDesc.dependencyBuffers =
      std::span<const BufferHandle>(dependencies.data(), dependencies.size());
  passDesc.draws = std::span<const DrawItem>(draws.data(), draws.size());
  passDesc.debugLabel = "native_graphics_pass";

  auto addResult = builder.addGraphicsPass(passDesc);
  if (!(!addResult.hasError())) {
    ADD_FAILURE() << "addGraphicsPass should succeed for native declaration "
                     "path";
    return;
  }

  auto compileResult = builder.compile();
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed for native declaration path";
    if (compileResult.hasError()) {
      std::cerr << compileResult.error() << "\n";
    }
    return;
  }
  const RenderGraphCompileResult &compiled = compileResult.value();

  if (!(compiled.orderedPasses.size() == 1u)) {
    ADD_FAILURE() << "native declaration path should schedule one pass";
    return;
  }
  if (!(compiled.unresolvedTextureBindings.size() == 2u)) {
    ADD_FAILURE()
        << "native declaration path should emit unresolved color+depth "
           "transient bindings";
    return;
  }
  if (!(compiled.dependencyBufferRangesByPass.size() == 1u &&
        compiled.dependencyBufferRangesByPass[0u].count == 1u)) {
    ADD_FAILURE()
        << "native declaration path should preserve dependency buffer "
           "slots";
    return;
  }
  if (!(compiled.drawRangesByPass.size() == 1u &&
        compiled.drawRangesByPass[0u].count == 1u)) {
    ADD_FAILURE() << "native declaration path should preserve draw ranges";
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

  auto compileResult = builder.compile();
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

  auto compileResult = builder.compile();
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

TEST(RenderGraphCompileBehaviorTest,
     InferredSideEffectSuppressedByExplicitFrameOutputRoots) {
  RenderGraphBuilder builder;
  builder.beginFrame(206u);
  builder.setInferredSideEffectSuppression(true);

  const TextureHandle colorA{.index = 51u, .generation = 1u};
  const TextureHandle colorB{.index = 52u, .generation = 1u};

  auto implicitResult =
      addTestGraphicsPass(builder, makeTestPass("cull_implicit_backbuffer"),
                          "cull_implicit_backbuffer");
  auto producerResult = addTestGraphicsPass(
      builder, makeTestPass("cull_producer", colorA), "cull_producer");
  auto outputResult = addTestGraphicsPass(
      builder, makeTestPass("cull_output", colorB), "cull_output");
  if (!(!implicitResult.hasError() && !producerResult.hasError() &&
        !outputResult.hasError())) {
    ADD_FAILURE() << "addLegacyRenderPass should succeed for inferred-root "
                     "suppression graph";
    return;
  }

  auto depResult =
      builder.addDependency(producerResult.value(), outputResult.value());
  if (!(!depResult.hasError())) {
    ADD_FAILURE() << "addDependency producer->output should succeed";
    return;
  }

  auto importResult = builder.importTexture(colorB, "explicit_frame_output");
  if (!(!importResult.hasError())) {
    ADD_FAILURE() << "importTexture should succeed for explicit frame output";
    return;
  }
  auto outputMarkResult =
      builder.markTextureAsFrameOutput(importResult.value());
  if (!(!outputMarkResult.hasError())) {
    ADD_FAILURE() << "markTextureAsFrameOutput should succeed";
    return;
  }

  auto compileResult = builder.compile();
  if (!(!compileResult.hasError())) {
    ADD_FAILURE()
        << "compile should succeed for inferred-root suppression graph";
    if (compileResult.hasError()) {
      std::cerr << compileResult.error() << "\n";
    }
    return;
  }
  const RenderGraphCompileResult &compiled = compileResult.value();

  if (!(compiled.declaredPassCount == 3u)) {
    ADD_FAILURE()
        << "inferred-root suppression graph should declare three passes";
    return;
  }
  if (!(compiled.culledPassCount == 1u)) {
    ADD_FAILURE() << "implicit inferred side-effect pass should be culled when "
                     "explicit frame output roots exist";
    return;
  }
  if (!(compiled.rootPassCount == 1u)) {
    ADD_FAILURE() << "expected a single explicit frame-output root";
    return;
  }
  if (!(std::find(compiled.orderedPassIndices.begin(),
                  compiled.orderedPassIndices.end(),
                  implicitResult.value().value) ==
        compiled.orderedPassIndices.end())) {
    ADD_FAILURE() << "inferred implicit side-effect pass should not be "
                     "scheduled";
    return;
  }
  if (!(compiled.orderedPassIndices.size() == 2u)) {
    ADD_FAILURE() << "producer and output passes should remain scheduled";
    return;
  }
}

TEST(RenderGraphCompileBehaviorTest, ExplicitSideEffectUpgradesInferredMark) {
  RenderGraphBuilder builder;
  builder.beginFrame(207u);
  builder.setInferredSideEffectSuppression(true);

  const TextureHandle colorOut{.index = 61u, .generation = 1u};

  auto implicitResult =
      addTestGraphicsPass(builder, makeTestPass("explicit_upgrade_target"),
                          "explicit_upgrade_target");
  auto outputResult = addTestGraphicsPass(
      builder, makeTestPass("explicit_upgrade_output", colorOut),
      "explicit_upgrade_output");
  if (!(!implicitResult.hasError() && !outputResult.hasError())) {
    ADD_FAILURE() << "addLegacyRenderPass should succeed for explicit-upgrade "
                     "graph";
    return;
  }

  auto explicitMarkResult = builder.markPassSideEffect(implicitResult.value());
  if (!(!explicitMarkResult.hasError())) {
    ADD_FAILURE() << "explicit markPassSideEffect should succeed";
    return;
  }

  auto importResult =
      builder.importTexture(colorOut, "explicit_upgrade_output_tex");
  if (!(!importResult.hasError())) {
    ADD_FAILURE() << "importTexture should succeed for explicit-upgrade graph";
    return;
  }
  auto outputMarkResult =
      builder.markTextureAsFrameOutput(importResult.value());
  if (!(!outputMarkResult.hasError())) {
    ADD_FAILURE() << "markTextureAsFrameOutput should succeed";
    return;
  }

  auto compileResult = builder.compile();
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed for explicit-upgrade graph";
    if (compileResult.hasError()) {
      std::cerr << compileResult.error() << "\n";
    }
    return;
  }
  const RenderGraphCompileResult &compiled = compileResult.value();

  if (!(compiled.culledPassCount == 0u)) {
    ADD_FAILURE()
        << "explicit side-effect mark should keep upgraded pass alive";
    return;
  }
  if (!(compiled.rootPassCount == 2u)) {
    ADD_FAILURE()
        << "expected both explicit side-effect and frame-output roots";
    return;
  }
  if (!(std::find(compiled.orderedPassIndices.begin(),
                  compiled.orderedPassIndices.end(),
                  implicitResult.value().value) !=
        compiled.orderedPassIndices.end())) {
    ADD_FAILURE() << "explicitly marked side-effect pass should remain "
                     "scheduled";
    return;
  }
}

TEST(RenderGraphCompileBehaviorTest,
     DefaultPolicyKeepsInferredSideEffectRoots) {
  RenderGraphBuilder builder;
  builder.beginFrame(208u);

  const TextureHandle colorOut{.index = 71u, .generation = 1u};

  auto implicitResult =
      addTestGraphicsPass(builder, makeTestPass("default_policy_implicit"),
                          "default_policy_implicit");
  auto outputResult = addTestGraphicsPass(
      builder, makeTestPass("default_policy_output", colorOut),
      "default_policy_output");
  if (!(!implicitResult.hasError() && !outputResult.hasError())) {
    ADD_FAILURE() << "addLegacyRenderPass should succeed for default-policy "
                     "graph";
    return;
  }

  auto importResult = builder.importTexture(colorOut, "default_policy_out_tex");
  if (!(!importResult.hasError())) {
    ADD_FAILURE() << "importTexture should succeed for default-policy graph";
    return;
  }
  auto outputMarkResult =
      builder.markTextureAsFrameOutput(importResult.value());
  if (!(!outputMarkResult.hasError())) {
    ADD_FAILURE() << "markTextureAsFrameOutput should succeed";
    return;
  }

  auto compileResult = builder.compile();
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed for default-policy graph";
    if (compileResult.hasError()) {
      std::cerr << compileResult.error() << "\n";
    }
    return;
  }
  const RenderGraphCompileResult &compiled = compileResult.value();

  if (!(compiled.culledPassCount == 1u)) {
    ADD_FAILURE()
        << "default policy should cull implicit pass without explicit "
           "side-effect mark";
    return;
  }
  if (!(std::find(compiled.orderedPassIndices.begin(),
                  compiled.orderedPassIndices.end(),
                  implicitResult.value().value) ==
        compiled.orderedPassIndices.end())) {
    ADD_FAILURE()
        << "implicit pass without explicit side-effect mark should be "
           "culled";
    return;
  }
}

class TestImplicitOutputLegacyLayer final : public Layer {
public:
  explicit TestImplicitOutputLegacyLayer(TextureHandle explicitColorOut)
      : explicitColorOut_(explicitColorOut) {}

  Result<bool, std::string>
  buildRenderGraph(RenderFrameContext &, RenderGraphBuilder &graph) override {
    RenderGraphGraphicsPassDesc implicitDesc{};
    implicitDesc.debugLabel = "legacy_bridge_implicit";
    auto implicitResult = graph.addGraphicsPass(implicitDesc);
    if (implicitResult.hasError()) {
      return Result<bool, std::string>::makeError(implicitResult.error());
    }

    auto colorImportResult =
        graph.importTexture(explicitColorOut_, "legacy_bridge_output_color");
    if (colorImportResult.hasError()) {
      return Result<bool, std::string>::makeError(colorImportResult.error());
    }

    RenderGraphGraphicsPassDesc outputDesc{};
    outputDesc.colorTexture = colorImportResult.value();
    outputDesc.debugLabel = "legacy_bridge_output";
    auto outputResult = graph.addGraphicsPass(outputDesc);
    if (outputResult.hasError()) {
      return Result<bool, std::string>::makeError(outputResult.error());
    }
    return Result<bool, std::string>::makeResult(true);
  }

private:
  TextureHandle explicitColorOut_{};
};

class TestDepthOverrideLegacyLayer final : public Layer {
public:
  explicit TestDepthOverrideLegacyLayer(TextureHandle depthTexture)
      : depthTexture_(depthTexture) {}

  Result<bool, std::string>
  buildRenderGraph(RenderFrameContext &frame,
                   RenderGraphBuilder &graph) override {
    const TextureHandle sceneDepthTexture = resolveFrameDepthTexture(frame);
    RenderGraphTextureId sceneDepthGraphTexture{};
    if (const RenderGraphTextureId *publishedSceneDepth =
            frame.channels.tryGet<RenderGraphTextureId>(
                kFrameChannelSceneDepthGraphTexture);
        publishedSceneDepth != nullptr) {
      sceneDepthGraphTexture = *publishedSceneDepth;
    }

    RenderGraphGraphicsPassDesc desc{};
    desc.debugLabel = "legacy_bridge_depth_override";
    if (nuri::isValid(sceneDepthTexture) &&
        nuri::isValid(sceneDepthGraphTexture) &&
        sceneDepthTexture.index == depthTexture_.index &&
        sceneDepthTexture.generation == depthTexture_.generation) {
      desc.depthTexture = sceneDepthGraphTexture;
    } else {
      auto depthImportResult = graph.importTexture(
          depthTexture_, "legacy_bridge_depth_override_depth");
      if (depthImportResult.hasError()) {
        return Result<bool, std::string>::makeError(depthImportResult.error());
      }
      desc.depthTexture = depthImportResult.value();
    }

    auto addResult = graph.addGraphicsPass(desc);
    if (addResult.hasError()) {
      return Result<bool, std::string>::makeError(addResult.error());
    }
    return Result<bool, std::string>::makeResult(true);
  }

private:
  TextureHandle depthTexture_{};
};

TEST(RenderGraphCompileBehaviorTest,
     DefaultLayerBridgeMarksImplicitOutputSideEffect) {
  RenderGraphBuilder builder;
  builder.beginFrame(210u);
  builder.setInferredSideEffectSuppression(true);

  const TextureHandle colorOut{.index = 91u, .generation = 1u};
  TestImplicitOutputLegacyLayer layer(colorOut);
  RenderFrameContext frame{};

  auto buildResult = layer.buildRenderGraph(frame, builder);
  if (!(!buildResult.hasError())) {
    ADD_FAILURE() << "Layer::buildRenderGraph should succeed for default "
                     "legacy bridge test";
    if (buildResult.hasError()) {
      std::cerr << buildResult.error() << "\n";
    }
    return;
  }

  auto outputImportResult =
      builder.importTexture(colorOut, "legacy_bridge_out");
  if (!(!outputImportResult.hasError())) {
    ADD_FAILURE() << "importTexture should succeed for default legacy bridge "
                     "test";
    return;
  }
  auto outputMarkResult =
      builder.markTextureAsFrameOutput(outputImportResult.value());
  if (!(!outputMarkResult.hasError())) {
    ADD_FAILURE() << "markTextureAsFrameOutput should succeed for default "
                     "legacy bridge test";
    return;
  }

  auto compileResult = builder.compile();
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed for default legacy bridge test";
    if (compileResult.hasError()) {
      std::cerr << compileResult.error() << "\n";
    }
    return;
  }
  const RenderGraphCompileResult &compiled = compileResult.value();

  if (!(compiled.declaredPassCount == 2u)) {
    ADD_FAILURE() << "default bridge test should declare two passes";
    return;
  }
  if (!(compiled.culledPassCount == 0u)) {
    ADD_FAILURE() << "default bridge implicit output pass should be kept by "
                     "explicit side-effect mark";
    return;
  }
  if (!(compiled.rootPassCount == 2u)) {
    ADD_FAILURE() << "expected implicit-output side-effect root and explicit "
                     "frame-output root";
    return;
  }
  if (!(compiled.orderedPassIndices.size() == 2u)) {
    ADD_FAILURE() << "both passes should remain scheduled";
    return;
  }
  if (!(compiled.orderedPassIndices[0u] == 0u &&
        compiled.orderedPassIndices[1u] == 1u)) {
    ADD_FAILURE() << "default bridge test should preserve pass order";
    return;
  }
}

TEST(RenderGraphCompileBehaviorTest,
     DefaultLayerBridgeUsesSceneDepthGraphOverride) {
  RenderGraphBuilder builder;
  builder.beginFrame(211u);

  auto sceneDepthResult = builder.createTransientTexture(
      makeTransientTextureDesc(Format::D32_FLOAT, 32u, 32u),
      "legacy_bridge_scene_depth");
  if (!(!sceneDepthResult.hasError())) {
    ADD_FAILURE() << "createTransientTexture scene depth should succeed";
    return;
  }

  const TextureHandle sceneDepthTexture{.index = 101u, .generation = 1u};
  TestDepthOverrideLegacyLayer layer(sceneDepthTexture);
  RenderFrameContext frame{};
  frame.channels.publish<TextureHandle>(kFrameChannelSceneDepthTexture,
                                        sceneDepthTexture);
  frame.channels.publish<RenderGraphTextureId>(
      kFrameChannelSceneDepthGraphTexture, sceneDepthResult.value());

  auto buildResult = layer.buildRenderGraph(frame, builder);
  if (!(!buildResult.hasError())) {
    ADD_FAILURE() << "Layer::buildRenderGraph should succeed for depth "
                     "override bridge test";
    if (buildResult.hasError()) {
      std::cerr << buildResult.error() << "\n";
    }
    return;
  }

  auto compileResult = builder.compile();
  if (!(!compileResult.hasError())) {
    ADD_FAILURE() << "compile should succeed for depth override bridge test";
    if (compileResult.hasError()) {
      std::cerr << compileResult.error() << "\n";
    }
    return;
  }
  const RenderGraphCompileResult &compiled = compileResult.value();

  if (!(compiled.orderedPasses.size() == 1u)) {
    ADD_FAILURE() << "depth override bridge should schedule one pass";
    return;
  }
  if (!(compiled.unresolvedTextureBindings.size() == 1u)) {
    ADD_FAILURE() << "depth override bridge should emit one unresolved depth "
                     "binding";
    return;
  }
  const auto &binding = compiled.unresolvedTextureBindings[0u];
  if (!(binding.target ==
        RenderGraphCompileResult::PassTextureBindingTarget::Depth)) {
    ADD_FAILURE()
        << "depth override bridge should bind unresolved depth target";
    return;
  }
  if (!(binding.textureResourceIndex == sceneDepthResult.value().value)) {
    ADD_FAILURE() << "depth override bridge should bind published scene-depth "
                     "graph texture";
    return;
  }
  if (!(!nuri::isValid(compiled.orderedPasses[0u].depthTexture))) {
    ADD_FAILURE() << "resolved pass depth slot should stay unresolved for "
                     "transient depth binding";
    return;
  }
}

TEST(RenderGraphCompileBehaviorTest,
     AddLegacyRenderPassRejectsDependencyBufferCountOverContractLimit) {
  RenderGraphBuilder builder;
  builder.beginFrame(212u);

  std::array<BufferHandle, kMaxDependencyBuffers + 1u> deps{};
  RenderPass pass{};
  pass.debugLabel = "contract_dep_limit";
  pass.dependencyBuffers =
      std::span<const BufferHandle>(deps.data(), deps.size());

  auto addResult = addTestGraphicsPass(builder, pass, pass.debugLabel);
  if (!(addResult.hasError())) {
    ADD_FAILURE()
        << "addLegacyRenderPass should reject dependency buffer count "
           "over kMaxDependencyBuffers";
    return;
  }
  if (((addResult.error()).find("exceeds kMaxDependencyBuffers") ==
       std::string_view::npos)) {
    ADD_FAILURE() << "error should mention kMaxDependencyBuffers contract "
                     "limit";
    return;
  }
}

TEST(
    RenderGraphCompileBehaviorTest,
    AddLegacyRenderPassRejectsPreDispatchDependencyBufferCountOverContractLimit) {
  RenderGraphBuilder builder;
  builder.beginFrame(213u);

  std::array<BufferHandle, kMaxDependencyBuffers + 1u> dispatchDeps{};
  ComputeDispatchItem dispatch{};
  dispatch.dependencyBuffers =
      std::span<const BufferHandle>(dispatchDeps.data(), dispatchDeps.size());
  std::array<ComputeDispatchItem, 1u> preDispatches = {dispatch};

  RenderPass pass{};
  pass.debugLabel = "contract_predispatch_dep_limit";
  pass.preDispatches = std::span<const ComputeDispatchItem>(
      preDispatches.data(), preDispatches.size());

  auto addResult = addTestGraphicsPass(builder, pass, pass.debugLabel);
  if (!(addResult.hasError())) {
    ADD_FAILURE()
        << "addLegacyRenderPass should reject pre-dispatch dependency buffer "
           "count over kMaxDependencyBuffers";
    return;
  }
  if (((addResult.error()).find("exceeds kMaxDependencyBuffers") ==
       std::string_view::npos)) {
    ADD_FAILURE() << "error should mention kMaxDependencyBuffers contract "
                     "limit";
    return;
  }
}

TEST(RenderGraphCompileBehaviorTest,
     BindPassDependencyBufferRejectsIndexOverContractLimit) {
  RenderGraphBuilder builder;
  builder.beginFrame(214u);

  std::array<BufferHandle, kMaxDependencyBuffers> deps{};
  RenderPass pass{};
  pass.debugLabel = "contract_bind_dep_limit";
  pass.dependencyBuffers =
      std::span<const BufferHandle>(deps.data(), deps.size());
  auto addResult = addTestGraphicsPass(builder, pass, pass.debugLabel);
  if (!(!addResult.hasError())) {
    ADD_FAILURE() << "addLegacyRenderPass should succeed for bind dependency "
                     "contract-limit test";
    return;
  }

  auto transientBufferResult = builder.createTransientBuffer(
      makeTransientBufferDesc(64u), "contract_bind_dep_limit_buffer");
  if (!(!transientBufferResult.hasError())) {
    ADD_FAILURE() << "createTransientBuffer should succeed for bind dependency "
                     "contract-limit test";
    return;
  }

  auto bindResult = builder.bindPassDependencyBuffer(
      addResult.value(), static_cast<uint32_t>(kMaxDependencyBuffers),
      transientBufferResult.value());
  if (!(bindResult.hasError())) {
    ADD_FAILURE() << "bindPassDependencyBuffer should reject dependency index "
                     "over kMaxDependencyBuffers";
    return;
  }
  if (((bindResult.error()).find("exceeds kMaxDependencyBuffers") ==
       std::string_view::npos)) {
    ADD_FAILURE() << "bindPassDependencyBuffer error should mention "
                     "kMaxDependencyBuffers contract limit";
    return;
  }
}

TEST(RenderGraphCompileBehaviorTest,
     BindPreDispatchDependencyBufferRejectsIndexOverContractLimit) {
  RenderGraphBuilder builder;
  builder.beginFrame(215u);

  std::array<BufferHandle, kMaxDependencyBuffers> dispatchDeps{};
  ComputeDispatchItem dispatch{};
  dispatch.dependencyBuffers =
      std::span<const BufferHandle>(dispatchDeps.data(), dispatchDeps.size());
  std::array<ComputeDispatchItem, 1u> preDispatches = {dispatch};

  RenderPass pass{};
  pass.debugLabel = "contract_bind_predispatch_dep_limit";
  pass.preDispatches = std::span<const ComputeDispatchItem>(
      preDispatches.data(), preDispatches.size());
  auto addResult = addTestGraphicsPass(builder, pass, pass.debugLabel);
  if (!(!addResult.hasError())) {
    ADD_FAILURE() << "addLegacyRenderPass should succeed for bind "
                     "pre-dispatch dependency contract-limit test";
    return;
  }

  auto transientBufferResult = builder.createTransientBuffer(
      makeTransientBufferDesc(64u),
      "contract_bind_predispatch_dep_limit_buffer");
  if (!(!transientBufferResult.hasError())) {
    ADD_FAILURE() << "createTransientBuffer should succeed for bind "
                     "pre-dispatch dependency contract-limit test";
    return;
  }

  auto bindResult = builder.bindPreDispatchDependencyBuffer(
      addResult.value(), 0u, static_cast<uint32_t>(kMaxDependencyBuffers),
      transientBufferResult.value());
  if (!(bindResult.hasError())) {
    ADD_FAILURE() << "bindPreDispatchDependencyBuffer should reject dependency "
                     "index over kMaxDependencyBuffers";
    return;
  }
  if (((bindResult.error()).find("exceeds kMaxDependencyBuffers") ==
       std::string_view::npos)) {
    ADD_FAILURE() << "bindPreDispatchDependencyBuffer error should mention "
                     "kMaxDependencyBuffers contract limit";
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

  auto compileResult = builder.compile();
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

  auto compileResult = builder.compile();
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

} // namespace
