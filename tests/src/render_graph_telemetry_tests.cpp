#include "nuri/pch.h"

#include "render_graph_test_support.h"

#include <gtest/gtest.h>

#include "nuri/gfx/render_graph/render_graph_telemetry.h"

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory_resource>
#include <string>

namespace {

using namespace nuri;
using namespace nuri::test_support;

std::filesystem::path makeTempPath(std::string_view stem) {
  const auto tick = std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count();
  return std::filesystem::temp_directory_path() /
         ("nuri_" + std::string(stem) + "_" + std::to_string(tick));
}

void populateTelemetryCompileResult(RenderGraphCompileResult &compiled,
                                    std::pmr::memory_resource *memory) {
  compiled.frameIndex = 42u;
  compiled.declaredPassCount = 3u;
  compiled.culledPassCount = 1u;
  compiled.rootPassCount = 2u;
  compiled.resourceStats.importedTextures = 4u;
  compiled.resourceStats.transientTextures = 5u;
  compiled.resourceStats.importedBuffers = 6u;
  compiled.resourceStats.transientBuffers = 7u;
  compiled.transientTexturePhysicalCount = 8u;
  compiled.transientBufferPhysicalCount = 9u;

  compiled.passDebugNames.emplace_back("first_pass", memory);
  compiled.passDebugNames.emplace_back("second_pass", memory);
  compiled.orderedPassIndices.push_back(1u);
  compiled.orderedPassIndices.push_back(0u);
  compiled.edges.push_back({.before = 0u, .after = 1u});

  compiled.transientTextureLifetimes.push_back(
      {.resourceIndex = 3u, .firstExecutionIndex = 0u, .lastExecutionIndex = 2u});
  compiled.transientBufferLifetimes.push_back(
      {.resourceIndex = 4u, .firstExecutionIndex = 1u, .lastExecutionIndex = 3u});
  compiled.transientTextureAllocations.push_back(
      {.resourceIndex = 3u, .allocationIndex = 1u});
  compiled.transientBufferAllocations.push_back(
      {.resourceIndex = 4u, .allocationIndex = 2u});
  compiled.transientTextureAllocationByResource = {UINT32_MAX, 1u, UINT32_MAX, 1u};
  compiled.transientBufferAllocationByResource = {2u, UINT32_MAX, UINT32_MAX};

  RenderGraphCompileResult::TransientTexturePhysicalAllocation texturePhysical{};
  texturePhysical.allocationIndex = 1u;
  texturePhysical.representativeResourceIndex = 3u;
  texturePhysical.desc = makeTransientTextureDesc(Format::RGBA8_UNORM, 64u, 32u);
  compiled.transientTexturePhysicalAllocations.push_back(texturePhysical);

  RenderGraphCompileResult::TransientBufferPhysicalAllocation bufferPhysical{};
  bufferPhysical.allocationIndex = 2u;
  bufferPhysical.representativeResourceIndex = 4u;
  bufferPhysical.desc = makeTransientBufferDesc(128u);
  compiled.transientBufferPhysicalAllocations.push_back(bufferPhysical);

  compiled.unresolvedTextureBindings.push_back(
      {.orderedPassIndex = 0u,
       .textureResourceIndex = 3u,
       .target = RenderGraphCompileResult::PassTextureBindingTarget::Color});
  compiled.resolvedDependencyBuffers.push_back(
      BufferHandle{.index = 11u, .generation = 2u});
  compiled.dependencyBufferRangesByPass.push_back({.offset = 0u, .count = 1u});
  compiled.unresolvedDependencyBufferBindings.push_back(
      {.orderedPassIndex = 0u, .dependencyBufferIndex = 0u, .bufferResourceIndex = 4u});
  compiled.preDispatchRangesByPass.push_back({.offset = 0u, .count = 1u});
  compiled.preDispatchDependencyRanges.push_back({.offset = 0u, .count = 1u});
  compiled.unresolvedPreDispatchDependencyBufferBindings.push_back(
      {.orderedPassIndex = 0u,
       .preDispatchIndex = 0u,
       .dependencyBufferIndex = 0u,
       .bufferResourceIndex = 4u});
  compiled.drawRangesByPass.push_back({.offset = 0u, .count = 1u});
  compiled.unresolvedDrawBufferBindings.push_back(
      {.orderedPassIndex = 0u,
       .drawIndex = 0u,
       .target = RenderGraphCompileResult::DrawBufferBindingTarget::Vertex,
       .bufferResourceIndex = 4u});
  compiled.resolvedPreDispatchDependencyBuffers.push_back(
      BufferHandle{.index = 12u, .generation = 3u});
  compiled.ownedPreDispatches.push_back(ComputeDispatchItem{});
  compiled.ownedDrawItems.push_back(DrawItem{});
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
    RenderGraphCompileResult compiled(&compileMemory);
    populateTelemetryCompileResult(compiled, &compileMemory);
    telemetry.capture(compiled);
  }

  const RenderGraphTelemetrySnapshot *snapshot = telemetry.latestSnapshot();
  ASSERT_NE(snapshot, nullptr);
  EXPECT_EQ(snapshot->summary.frameIndex, 42u);
  EXPECT_EQ(snapshot->summary.declaredPassCount, 3u);
  EXPECT_EQ(snapshot->summary.importedTextures, 4u);
  ASSERT_EQ(snapshot->passNames.size(), 2u);
  EXPECT_EQ(snapshot->passNames[0], "first_pass");
  EXPECT_EQ(snapshot->passNames[1], "second_pass");
  ASSERT_EQ(snapshot->orderedPassIndices.size(), 2u);
  EXPECT_EQ(snapshot->orderedPassIndices[0], 1u);
  EXPECT_EQ(snapshot->orderedPassIndices[1], 0u);
  ASSERT_EQ(snapshot->edges.size(), 1u);
  EXPECT_EQ(snapshot->edges[0].before, 0u);
  EXPECT_EQ(snapshot->edges[0].after, 1u);
  ASSERT_EQ(snapshot->unresolvedDrawBufferBindings.size(), 1u);
  EXPECT_EQ(snapshot->unresolvedDrawBufferBindings[0].bufferResourceIndex, 4u);
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
  RenderGraphCompileResult compiled(&compileMemory);
  populateTelemetryCompileResult(compiled, &compileMemory);
  telemetry.capture(compiled);

  EXPECT_TRUE(telemetry.writeLatestTextDump("").hasError());

  const std::filesystem::path outputPath = makeTempPath("render_graph_dump.txt");
  const auto dumpResult =
      telemetry.writeLatestTextDump(outputPath.generic_string());
  ASSERT_FALSE(dumpResult.hasError());

  std::ifstream file(outputPath);
  ASSERT_TRUE(file.is_open());
  const std::string contents((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());
  EXPECT_NE(contents.find("frame_index: 42"), std::string::npos);
  EXPECT_NE(contents.find("first_pass"), std::string::npos);
  EXPECT_NE(contents.find("pass_exec[0].draw[0].vertex <- buf[4]"),
            std::string::npos);

  std::error_code ec;
  std::filesystem::remove(outputPath, ec);
}

TEST(RenderGraphTelemetryTest, SuggestDumpPathUsesEnvDirectorySeed) {
  const std::filesystem::path dumpDirectory = makeTempPath("telemetry_seed");
  EnvVarGuard envGuard("NURI_RENDER_GRAPH_DUMP", dumpDirectory.generic_string());

  RenderGraphTelemetryService telemetry;

  std::array<std::byte, 16 * 1024> compileBytes{};
  std::pmr::monotonic_buffer_resource compileMemory(compileBytes.data(),
                                                    compileBytes.size());
  RenderGraphCompileResult compiled(&compileMemory);
  populateTelemetryCompileResult(compiled, &compileMemory);
  telemetry.capture(compiled);

  const std::filesystem::path suggested = telemetry.suggestDumpPath();
  EXPECT_EQ(suggested.parent_path(), dumpDirectory);
  EXPECT_EQ(suggested.filename(), "render_graph_frame_42.txt");
}

} // namespace
