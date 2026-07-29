#include "tests_pch.h"

#include "render_graph_test_support.h"

#include <gtest/gtest.h>

#include "nuri/gfx/render_graph/render_graph_telemetry.h"

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory_resource>
#include <span>
#include <string>

namespace {

using namespace nuri;
using namespace nuri::test_support;

std::filesystem::path makeTempPath(std::string_view stem) {
  const auto tick =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  return std::filesystem::temp_directory_path() /
         ("nuri_" + std::string(stem) + "_" + std::to_string(tick));
}

void populateTelemetryCompileResult(CompiledRenderGraph &compiled,
                                    std::pmr::memory_resource *memory) {
  compiled.commands.frameIndex = 42u;
  compiled.plan.declaredPassCount = 3u;
  compiled.plan.culledPassCount = 1u;
  compiled.plan.rootPassCount = 2u;
  compiled.plan.usedParallelCompile = true;
  compiled.commands.usedParallelPayloadResolution = true;
  compiled.plan.usedParallelHazardAnalysis = true;
  compiled.plan.usedParallelLifetimeAnalysis = false;
  compiled.plan.resourceStats.importedTextures = 4u;
  compiled.plan.resourceStats.transientTextures = 5u;
  compiled.plan.resourceStats.importedBuffers = 6u;
  compiled.plan.resourceStats.transientBuffers = 7u;
  compiled.plan.transientTexturePhysicalCount = 8u;
  compiled.plan.transientBufferPhysicalCount = 9u;

  std::pmr::string firstPassName(memory);
  firstPassName = "first_pass";
  compiled.commands.passDebugNames.push_back(std::move(firstPassName));

  std::pmr::string secondPassName(memory);
  secondPassName = "second_pass";
  compiled.commands.passDebugNames.push_back(std::move(secondPassName));
  compiled.plan.orderedPassIndices.push_back(1u);
  compiled.plan.orderedPassIndices.push_back(0u);
  compiled.plan.edges.push_back({.before = 0u, .after = 1u});
  compiled.plan.passBarrierPlans.push_back(
      {.orderedPassIndex = 0u, .barrierOffset = 0u, .barrierCount = 1u});
  compiled.plan.passBarrierPlans.push_back(
      {.orderedPassIndex = 1u, .barrierOffset = 1u, .barrierCount = 1u});
  compiled.plan.finalBarrierPlan = {.barrierOffset = 2u, .barrierCount = 1u};
  compiled.plan.passBarrierRecords.push_back(
      {.resourceKind = RenderGraphBarrierResourceKind::Texture,
       .resourceIndex = 0u,
       .beforeAccess = RenderGraphAccessMode::None,
       .afterAccess = RenderGraphAccessMode::Write,
       .beforeState = RenderGraphResourceState::Unknown,
       .afterState = RenderGraphResourceState::Attachment});
  compiled.plan.passBarrierRecords.push_back(
      {.resourceKind = RenderGraphBarrierResourceKind::Buffer,
       .resourceIndex = 4u,
       .beforeAccess = RenderGraphAccessMode::Write,
       .afterAccess = RenderGraphAccessMode::Read,
       .beforeState = RenderGraphResourceState::Write,
       .afterState = RenderGraphResourceState::Read});
  compiled.plan.passBarrierRecords.push_back(
      {.resourceKind = RenderGraphBarrierResourceKind::Texture,
       .resourceIndex = 0u,
       .beforeAccess = RenderGraphAccessMode::Write,
       .afterAccess = RenderGraphAccessMode::None,
       .beforeState = RenderGraphResourceState::Attachment,
       .afterState = RenderGraphResourceState::Present});

  compiled.plan.transientTextureLifetimes.push_back({.resourceIndex = 3u,
                                                     .firstExecutionIndex = 0u,
                                                     .lastExecutionIndex = 2u});
  compiled.plan.transientBufferLifetimes.push_back({.resourceIndex = 4u,
                                                    .firstExecutionIndex = 1u,
                                                    .lastExecutionIndex = 3u});
  compiled.plan.transientTextureAllocations.push_back(
      {.resourceIndex = 3u, .allocationIndex = 1u});
  compiled.plan.transientBufferAllocations.push_back(
      {.resourceIndex = 4u, .allocationIndex = 2u});
  compiled.plan.transientTextureAllocationByResource = {UINT32_MAX, 1u,
                                                        UINT32_MAX, 1u};
  compiled.plan.transientBufferAllocationByResource = {2u, UINT32_MAX,
                                                       UINT32_MAX};

  RenderGraphPlan::TransientTexturePhysicalAllocation texturePhysical{};
  texturePhysical.allocationIndex = 1u;
  texturePhysical.representativeResourceIndex = 3u;
  texturePhysical.desc =
      makeTransientTextureDesc(Format::RGBA8_UNORM, 64u, 32u);
  compiled.plan.transientTexturePhysicalAllocations.push_back(texturePhysical);

  RenderGraphPlan::TransientBufferPhysicalAllocation bufferPhysical{};
  bufferPhysical.allocationIndex = 2u;
  bufferPhysical.representativeResourceIndex = 4u;
  bufferPhysical.desc = makeTransientBufferDesc(128u);
  compiled.plan.transientBufferPhysicalAllocations.push_back(bufferPhysical);

  compiled.plan.commandResourcePatches.push_back(
      {.orderedPassIndex = 0u,
       .resourceIndex = 3u,
       .target = RenderGraphPlan::CommandResourcePatchTarget::PassColor});
  compiled.commands.resolvedDependencyBuffers.push_back(
      BufferHandle{.index = 11u, .generation = 2u});
  compiled.plan.dependencyBufferRangesByPass.push_back(
      {.offset = 0u, .count = 1u});
  compiled.plan.commandResourcePatches.push_back(
      {.orderedPassIndex = 0u,
       .dependencyIndex = 0u,
       .resourceIndex = 4u,
       .resourceKind = RenderGraphResourceKind::Buffer,
       .target =
           RenderGraphPlan::CommandResourcePatchTarget::PassDependencyBuffer});
  compiled.plan.preDispatchRangesByPass.push_back({.offset = 0u, .count = 1u});
  compiled.plan.preDispatchDependencyRanges.push_back(
      {.offset = 0u, .count = 1u});
  compiled.plan.commandResourcePatches.push_back(
      {.orderedPassIndex = 0u,
       .commandIndex = 0u,
       .dependencyIndex = 0u,
       .resourceIndex = 4u,
       .resourceKind = RenderGraphResourceKind::Buffer,
       .target = RenderGraphPlan::CommandResourcePatchTarget::
           PreDispatchDependencyBuffer});
  compiled.plan.drawRangesByPass.push_back({.offset = 0u, .count = 2u});
  compiled.plan.commandResourcePatches.push_back(
      {.orderedPassIndex = 0u,
       .commandIndex = 0u,
       .resourceIndex = 4u,
       .resourceKind = RenderGraphResourceKind::Buffer,
       .target =
           RenderGraphPlan::CommandResourcePatchTarget::DrawVertexBuffer});
  compiled.commands.resolvedPreDispatchDependencyBuffers.push_back(
      BufferHandle{.index = 12u, .generation = 3u});
  compiled.commands.ownedPreDispatches.push_back(ComputeDispatchItem{});
  compiled.commands.ownedDrawItems.push_back(DrawItem{});
  DrawItem indirectDraw{};
  indirectDraw.indirectDrawCount = 7u;
  compiled.commands.ownedDrawItems.push_back(indirectDraw);
  RenderPass orderedPass{};
  orderedPass.draws =
      std::span<const DrawItem>(compiled.commands.ownedDrawItems.data(),
                                compiled.commands.ownedDrawItems.size());
  compiled.commands.orderedPasses.push_back(orderedPass);
}

void populateTelemetryExecutionMetadata(
    RenderGraphExecutionMetadata &execution,
    std::pmr::memory_resource *memoryResource = nullptr) {
  std::pmr::memory_resource *executionMemory =
      memoryResource != nullptr ? memoryResource
                                : std::pmr::get_default_resource();
  static_cast<void>(executionMemory);
  execution.usedParallelCompile = true;
  execution.usedParallelRecording = true;
  execution.recordedCommandBuffers.push_back(
      {.firstOrderedPassIndex = 0u, .passCount = 1u});
  execution.recordedCommandBuffers.push_back(
      {.firstOrderedPassIndex = 1u, .passCount = 1u});
  execution.submitBatches.push_back({.commandBufferOffset = 0u,
                                     .commandBufferCount = 2u,
                                     .presentsFrameOutput = true});
  execution.passRanges.push_back(
      {.workerIndex = 0u, .firstOrderedPassIndex = 0u, .passCount = 1u});
  execution.passRanges.push_back(
      {.workerIndex = 1u, .firstOrderedPassIndex = 1u, .passCount = 1u});
}

TEST(RenderGraphTelemetryTest, CaptureRequestIsExplicitAndConsumed) {
  EnvVarGuard envGuard("NURI_RENDER_GRAPH_DUMP", "0");
  RenderGraphTelemetryService telemetry;
  EXPECT_FALSE(telemetry.captureRequested());
  EXPECT_EQ(telemetry.requestedCaptureLevel(), RenderGraphTelemetryLevel::None);

  telemetry.requestCapture(RenderGraphTelemetryLevel::Metadata);
  EXPECT_TRUE(telemetry.captureRequested());
  EXPECT_EQ(telemetry.requestedCaptureLevel(),
            RenderGraphTelemetryLevel::Metadata);
  telemetry.requestCapture(RenderGraphTelemetryLevel::PassTimings);
  EXPECT_EQ(telemetry.requestedCaptureLevel(),
            RenderGraphTelemetryLevel::PassTimings);

  CompiledRenderGraph compiled;
  telemetry.capture({compiled.plan, compiled.commands});
  EXPECT_FALSE(telemetry.captureRequested());
  EXPECT_TRUE(telemetry.hasSnapshot());
}

TEST(RenderGraphTelemetryTest, CaptureDeepCopiesStructuredData) {
  std::array<std::byte, 32 * 1024> serviceBytes{};
  std::pmr::monotonic_buffer_resource serviceMemory(serviceBytes.data(),
                                                    serviceBytes.size());
  RenderGraphTelemetryService telemetry(&serviceMemory);

  {
    std::array<std::byte, 32 * 1024> compileBytes{};
    std::pmr::monotonic_buffer_resource compileMemory(compileBytes.data(),
                                                      compileBytes.size());
    CompiledRenderGraph compiled(&compileMemory);
    RenderGraphExecutionMetadata execution(&compileMemory);
    populateTelemetryCompileResult(compiled, &compileMemory);
    populateTelemetryExecutionMetadata(execution, &compileMemory);
    telemetry.capture({compiled.plan, compiled.commands}, execution);
  }

  const RenderGraphTelemetrySnapshot *snapshot = telemetry.latestSnapshot();
  ASSERT_NE(snapshot, nullptr);
  EXPECT_EQ(snapshot->summary.frameIndex, 42u);
  EXPECT_EQ(snapshot->summary.declaredPassCount, 3u);
  EXPECT_EQ(snapshot->summary.importedTextures, 4u);
  ASSERT_EQ(snapshot->passDebugNames.size(), 2u);
  EXPECT_EQ(snapshot->passDebugNames[0], "first_pass");
  EXPECT_EQ(snapshot->passDebugNames[1], "second_pass");
  ASSERT_EQ(snapshot->plan.orderedPassIndices.size(), 2u);
  EXPECT_EQ(snapshot->plan.orderedPassIndices[0], 1u);
  EXPECT_EQ(snapshot->plan.orderedPassIndices[1], 0u);
  ASSERT_EQ(snapshot->execution.recordedCommandBuffers.size(), 2u);
  EXPECT_EQ(snapshot->execution.recordedCommandBuffers[0].firstOrderedPassIndex,
            0u);
  EXPECT_EQ(snapshot->execution.submitBatches.size(), 1u);
  EXPECT_TRUE(snapshot->execution.submitBatches[0].presentsFrameOutput);
  ASSERT_EQ(snapshot->execution.passRanges.size(), 2u);
  EXPECT_EQ(snapshot->execution.passRanges[0].workerIndex, 0u);
  EXPECT_EQ(snapshot->execution.passRanges[0].firstOrderedPassIndex, 0u);
  EXPECT_EQ(snapshot->execution.passRanges[0].passCount, 1u);
  EXPECT_EQ(snapshot->execution.passRanges[1].workerIndex, 1u);
  EXPECT_EQ(snapshot->execution.passRanges[1].firstOrderedPassIndex, 1u);
  EXPECT_EQ(snapshot->execution.passRanges[1].passCount, 1u);
  EXPECT_TRUE(snapshot->summary.usedParallelCompile);
  EXPECT_TRUE(snapshot->summary.usedParallelPayloadResolution);
  EXPECT_TRUE(snapshot->summary.usedParallelHazardAnalysis);
  EXPECT_FALSE(snapshot->summary.usedParallelLifetimeAnalysis);
  EXPECT_TRUE(snapshot->summary.usedParallelRecording);
  EXPECT_EQ(snapshot->summary.ownedDrawItemCount, 2u);
  EXPECT_NE(snapshot->summary.compileFingerprint, 0ull);
  EXPECT_NE(snapshot->summary.barrierFingerprint, 0ull);
  EXPECT_NE(snapshot->summary.executionFingerprint, 0ull);
  EXPECT_EQ(snapshot->summary.finalBarrierRecordCount, 1u);
  EXPECT_EQ(snapshot->plan.finalBarrierPlan.barrierCount, 1u);
  ASSERT_EQ(snapshot->plan.edges.size(), 1u);
  EXPECT_EQ(snapshot->plan.edges[0].before, 0u);
  EXPECT_EQ(snapshot->plan.edges[0].after, 1u);
  ASSERT_EQ(snapshot->plan.commandResourcePatches.size(), 4u);
  EXPECT_EQ(snapshot->plan.commandResourcePatches.back().resourceIndex, 4u);
}

TEST(RenderGraphTelemetryTest, WriteDumpSerializesSnapshotAndValidatesInputs) {
  std::array<std::byte, 32 * 1024> serviceBytes{};
  std::pmr::monotonic_buffer_resource serviceMemory(serviceBytes.data(),
                                                    serviceBytes.size());
  RenderGraphTelemetryService telemetry(&serviceMemory);

  EXPECT_TRUE(telemetry.writeLatestTextDump("ignored.txt").hasError());

  std::array<std::byte, 32 * 1024> compileBytes{};
  std::pmr::monotonic_buffer_resource compileMemory(compileBytes.data(),
                                                    compileBytes.size());
  CompiledRenderGraph compiled(&compileMemory);
  RenderGraphExecutionMetadata execution(&compileMemory);
  populateTelemetryCompileResult(compiled, &compileMemory);
  populateTelemetryExecutionMetadata(execution, &compileMemory);
  telemetry.capture({compiled.plan, compiled.commands}, execution);

  EXPECT_TRUE(telemetry.writeLatestTextDump("").hasError());

  const std::filesystem::path outputPath =
      makeTempPath("render_graph_dump.txt");
  const auto dumpResult =
      telemetry.writeLatestTextDump(outputPath.generic_string());
  ASSERT_FALSE(dumpResult.hasError());

  std::ifstream file(outputPath);
  ASSERT_TRUE(file.is_open());
  const std::string contents((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());
  EXPECT_NE(contents.find("frame_index: 42"), std::string::npos);
  EXPECT_NE(contents.find("first_pass"), std::string::npos);
  EXPECT_NE(contents.find("compile_fingerprint:"), std::string::npos);
  EXPECT_NE(contents.find("final_barrier_record_count: 1"), std::string::npos);
  EXPECT_NE(contents.find("used_parallel_payload_resolution: 1"),
            std::string::npos);
  EXPECT_NE(contents.find("owned_draw_items: 2"), std::string::npos);
  EXPECT_NE(contents.find("Recorded Command Buffers:"), std::string::npos);
  EXPECT_NE(contents.find("Submit Batches:"), std::string::npos);
  EXPECT_NE(contents.find("Final Barrier Plan:"), std::string::npos);

  std::error_code ec;
  std::filesystem::remove(outputPath, ec);
}

TEST(RenderGraphTelemetryTest, SuggestDumpPathUsesEnvDirectorySeed) {
  const std::filesystem::path dumpDirectory = makeTempPath("telemetry_seed");
  EnvVarGuard envGuard("NURI_RENDER_GRAPH_DUMP",
                       dumpDirectory.generic_string());

  RenderGraphTelemetryService telemetry;
  EXPECT_TRUE(telemetry.captureRequested());
  EXPECT_EQ(telemetry.requestedCaptureLevel(),
            RenderGraphTelemetryLevel::PassTimings);

  std::array<std::byte, 16 * 1024> compileBytes{};
  std::pmr::monotonic_buffer_resource compileMemory(compileBytes.data(),
                                                    compileBytes.size());
  CompiledRenderGraph compiled(&compileMemory);
  RenderGraphExecutionMetadata execution(&compileMemory);
  populateTelemetryCompileResult(compiled, &compileMemory);
  populateTelemetryExecutionMetadata(execution, &compileMemory);
  telemetry.capture({compiled.plan, compiled.commands}, execution);

  const std::filesystem::path suggested = telemetry.suggestDumpPath();
  EXPECT_EQ(suggested.parent_path(), dumpDirectory);
  EXPECT_EQ(suggested.filename(), "render_graph_frame_42.txt");
}

} // namespace
