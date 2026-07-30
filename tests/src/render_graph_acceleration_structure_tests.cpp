#include "tests_pch.h"

#include "nuri/gfx/render_graph/render_graph.h"

#include <gtest/gtest.h>

#include <array>

namespace {

using namespace nuri;

struct GraphInputs {
  BufferHandle vertices{};
  BufferHandle indices{};
  AccelerationStructureHandle blas{};
  AccelerationStructureHandle tlas{};
  float instanceX = 0.0f;
};

struct GraphIds {
  RenderGraphBufferId vertices{};
  RenderGraphBufferId indices{};
  RenderGraphAccelerationStructureId blas{};
  RenderGraphAccelerationStructureId tlas{};
};

Result<GraphIds, std::string> buildGraph(RenderGraphBuilder &builder,
                                         const GraphInputs &inputs) {
  auto vertices = builder.importBuffer(inputs.vertices, "rt_vertices");
  auto indices = builder.importBuffer(inputs.indices, "rt_indices");
  auto blas = builder.importAccelerationStructure(inputs.blas, "rt_blas");
  auto tlas = builder.importAccelerationStructure(inputs.tlas, "rt_tlas");
  if (vertices.hasError() || indices.hasError() || blas.hasError() ||
      tlas.hasError()) {
    return Result<GraphIds, std::string>::makeError("resource import failed");
  }

  RenderGraphGraphicsPassDesc writerDesc{};
  writerDesc.hasColorAttachment = false;
  writerDesc.markImplicitOutputSideEffect = false;
  writerDesc.debugLabel = "geometry_writer";
  auto writer = builder.addGraphicsPass(writerDesc);
  if (writer.hasError()) {
    return Result<GraphIds, std::string>::makeError(writer.error());
  }
  auto writeVertices = builder.addBufferWrite(writer.value(), vertices.value());
  auto writeIndices = builder.addBufferWrite(writer.value(), indices.value());
  if (writeVertices.hasError() || writeIndices.hasError()) {
    return Result<GraphIds, std::string>::makeError(
        "geometry write declaration failed");
  }

  const std::array geometries{AccelerationStructureTriangleGeometryDesc{
      .vertexBuffer = inputs.vertices,
      .indexBuffer = inputs.indices,
      .vertexCount = 3u,
      .indexCount = 3u,
      .flags = AccelerationStructureGeometryFlags::Opaque,
  }};
  const std::array blasBuilds{AccelerationStructureBuildItem{
      .command =
          BuildBlasItem{
              .destination = inputs.blas,
              .geometries = geometries,
          },
  }};
  const std::array blasBuffers{
      RenderGraphBufferUse{.buffer = vertices.value(),
                           .access = RenderGraphAccessMode::Read},
      RenderGraphBufferUse{.buffer = indices.value(),
                           .access = RenderGraphAccessMode::Read},
  };
  const std::array blasUses{RenderGraphAccelerationStructureUse{
      .accelerationStructure = blas.value(),
      .access = RenderGraphAccelerationStructureAccess::BuildWrite,
  }};
  auto blasPass = builder.addAccelerationStructurePass(
      RenderGraphAccelerationStructurePassDesc{
          .builds = blasBuilds,
          .buffers = blasBuffers,
          .accelerationStructures = blasUses,
          .debugLabel = "build_blas",
      });
  if (blasPass.hasError()) {
    return Result<GraphIds, std::string>::makeError(blasPass.error());
  }

  AccelerationStructureInstanceDesc instance{};
  instance.transform.rowMajor3x4[3u] = inputs.instanceX;
  instance.bottomLevel = inputs.blas;
  const std::array instances{instance};
  const std::array tlasBuilds{AccelerationStructureBuildItem{
      .command =
          BuildTlasItem{
              .destination = inputs.tlas,
              .instances = instances,
          },
  }};
  const std::array tlasUses{
      RenderGraphAccelerationStructureUse{
          .accelerationStructure = blas.value(),
          .access = RenderGraphAccelerationStructureAccess::BuildRead,
      },
      RenderGraphAccelerationStructureUse{
          .accelerationStructure = tlas.value(),
          .access = RenderGraphAccelerationStructureAccess::BuildWrite,
      },
  };
  auto tlasPass = builder.addAccelerationStructurePass(
      RenderGraphAccelerationStructurePassDesc{
          .builds = tlasBuilds,
          .accelerationStructures = tlasUses,
          .debugLabel = "build_tlas",
      });
  if (tlasPass.hasError()) {
    return Result<GraphIds, std::string>::makeError(tlasPass.error());
  }

  RenderGraphGraphicsPassDesc queryDesc{};
  queryDesc.hasColorAttachment = false;
  queryDesc.markImplicitOutputSideEffect = false;
  queryDesc.debugLabel = "query_tlas";
  auto queryPass = builder.addGraphicsPass(queryDesc);
  if (queryPass.hasError()) {
    return Result<GraphIds, std::string>::makeError(queryPass.error());
  }
  auto queryRead = builder.addAccelerationStructureAccess(
      queryPass.value(), tlas.value(),
      RenderGraphAccelerationStructureAccess::RayQueryRead);
  auto root = builder.markPassSideEffect(queryPass.value());
  if (queryRead.hasError() || root.hasError()) {
    return Result<GraphIds, std::string>::makeError("query declaration failed");
  }
  return Result<GraphIds, std::string>::makeResult(GraphIds{
      .vertices = vertices.value(),
      .indices = indices.value(),
      .blas = blas.value(),
      .tlas = tlas.value(),
  });
}

TEST(RenderGraphAccelerationStructureTest,
     RejectsMissingBuildInputDeclaration) {
  RenderGraphBuilder builder;
  builder.beginFrame(1u);
  const BufferHandle vertices{.index = 1u, .generation = 1u};
  const BufferHandle indices{.index = 2u, .generation = 1u};
  const AccelerationStructureHandle blas{.index = 3u, .generation = 1u};
  ASSERT_FALSE(builder.importBuffer(vertices).hasError());
  ASSERT_FALSE(builder.importBuffer(indices).hasError());
  auto graphBlas = builder.importAccelerationStructure(blas);
  ASSERT_FALSE(graphBlas.hasError());
  const std::array geometries{AccelerationStructureTriangleGeometryDesc{
      .vertexBuffer = vertices,
      .indexBuffer = indices,
      .vertexCount = 3u,
      .indexCount = 3u,
  }};
  const std::array builds{AccelerationStructureBuildItem{
      .command = BuildBlasItem{.destination = blas, .geometries = geometries},
  }};
  const std::array uses{RenderGraphAccelerationStructureUse{
      .accelerationStructure = graphBlas.value(),
      .access = RenderGraphAccelerationStructureAccess::BuildWrite,
  }};
  auto result = builder.addAccelerationStructurePass(
      RenderGraphAccelerationStructurePassDesc{
          .builds = builds,
          .accelerationStructures = uses,
      });
  ASSERT_TRUE(result.hasError());
  EXPECT_NE(result.error().find("missing a read-only buffer declaration"),
            std::string::npos);
}

TEST(RenderGraphAccelerationStructureTest,
     OrdersBuildInputsBlasTlasAndRayQuery) {
  RenderGraphBuilder builder;
  builder.beginFrame(2u);
  const GraphInputs inputs{
      .vertices = {.index = 10u, .generation = 1u},
      .indices = {.index = 11u, .generation = 1u},
      .blas = {.index = 12u, .generation = 1u},
      .tlas = {.index = 13u, .generation = 1u},
  };
  ASSERT_FALSE(buildGraph(builder, inputs).hasError());
  RenderGraphRuntime runtime;
  auto result = builder.compile(runtime);
  ASSERT_FALSE(result.hasError()) << result.error();
  const CompiledRenderGraph &compiled = result.value();
  ASSERT_EQ(compiled.plan.orderedPassIndices.size(), 4u);
  EXPECT_EQ(compiled.plan.orderedPassIndices[0u], 0u);
  EXPECT_EQ(compiled.plan.orderedPassIndices[1u], 1u);
  EXPECT_EQ(compiled.plan.orderedPassIndices[2u], 2u);
  EXPECT_EQ(compiled.plan.orderedPassIndices[3u], 3u);
  EXPECT_EQ(compiled.plan.resourceStats.importedAccelerationStructures, 2u);

  bool sawBuildInput = false;
  bool sawBlasWriteToRead = false;
  bool sawTlasWriteToQuery = false;
  for (const RenderGraphBarrierRecord &barrier :
       compiled.plan.passBarrierRecords) {
    sawBuildInput =
        sawBuildInput ||
        (barrier.resourceKind == RenderGraphBarrierResourceKind::Buffer &&
         barrier.afterState ==
             RenderGraphResourceState::AccelerationStructureBuildInput);
    sawBlasWriteToRead =
        sawBlasWriteToRead ||
        (barrier.resourceKind ==
             RenderGraphBarrierResourceKind::AccelerationStructure &&
         barrier.resourceIndex == 0u &&
         barrier.beforeState ==
             RenderGraphResourceState::AccelerationStructureBuildWrite &&
         barrier.afterState ==
             RenderGraphResourceState::AccelerationStructureBuildRead);
    sawTlasWriteToQuery =
        sawTlasWriteToQuery ||
        (barrier.resourceKind ==
             RenderGraphBarrierResourceKind::AccelerationStructure &&
         barrier.resourceIndex == 1u &&
         barrier.beforeState ==
             RenderGraphResourceState::AccelerationStructureBuildWrite &&
         barrier.afterState == RenderGraphResourceState::RayQueryRead);
  }
  EXPECT_TRUE(sawBuildInput);
  EXPECT_TRUE(sawBlasWriteToRead);
  EXPECT_TRUE(sawTlasWriteToQuery);
}

TEST(RenderGraphAccelerationStructureTest,
     RefreshesHandlesAndTransformsWithoutChangingPlanIdentity) {
  RenderGraphBuilder builder;
  builder.beginFrame(3u);
  const GraphInputs first{
      .vertices = {.index = 20u, .generation = 1u},
      .indices = {.index = 21u, .generation = 1u},
      .blas = {.index = 22u, .generation = 1u},
      .tlas = {.index = 23u, .generation = 1u},
      .instanceX = 1.0f,
  };
  ASSERT_FALSE(buildGraph(builder, first).hasError());
  const auto firstFingerprint = builder.computeGraphFingerprint();
  RenderGraphRuntime runtime;
  auto compileResult = builder.compile(runtime);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  CompiledRenderGraph compiled = std::move(compileResult.value());

  builder.beginFrame(4u);
  const GraphInputs second{
      .vertices = {.index = 20u, .generation = 2u},
      .indices = {.index = 21u, .generation = 2u},
      .blas = {.index = 22u, .generation = 2u},
      .tlas = {.index = 23u, .generation = 2u},
      .instanceX = 9.0f,
  };
  ASSERT_FALSE(buildGraph(builder, second).hasError());
  EXPECT_EQ(builder.computeGraphFingerprint(), firstFingerprint);
  compiled.commands = builder.buildFrameCommands(compiled.plan);
  ASSERT_EQ(compiled.commands.accelerationStructureHandlesByResource.size(),
            2u);
  EXPECT_EQ(compiled.commands.accelerationStructureHandlesByResource[0u],
            second.blas);
  EXPECT_EQ(compiled.commands.accelerationStructureHandlesByResource[1u],
            second.tlas);
  ASSERT_EQ(
      compiled.commands.orderedPasses[2u].accelerationStructureBuilds.size(),
      1u);
  const auto *tlasBuild =
      std::get_if<BuildTlasItem>(&compiled.commands.orderedPasses[2u]
                                      .accelerationStructureBuilds[0u]
                                      .command);
  ASSERT_NE(tlasBuild, nullptr);
  EXPECT_EQ(tlasBuild->destination, second.tlas);
  ASSERT_EQ(tlasBuild->instances.size(), 1u);
  EXPECT_EQ(tlasBuild->instances[0u].bottomLevel, second.blas);
  EXPECT_FLOAT_EQ(tlasBuild->instances[0u].transform.rowMajor3x4[3u], 9.0f);
}

} // namespace
