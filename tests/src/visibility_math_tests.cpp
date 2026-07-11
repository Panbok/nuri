#include "tests_pch.h"

#include "nuri/gfx/visibility/visibility.h"
#include "nuri/gfx/renderers/detail/visibility_math.h"
#include "nuri/scene/camera.h"

#include <array>

#include <gtest/gtest.h>

namespace nuri {
namespace {

Camera makeTestCamera() {
  Camera camera(glm::vec3(0.0f), glm::quat(1.0f, 0.0f, 0.0f, 0.0f));
  camera.setPerspective(PerspectiveParams{
      .fovYRadians = glm::radians(90.0f),
      .nearPlane = 0.1f,
      .farPlane = 10.0f,
  });
  return camera;
}

} // namespace

TEST(VisibilityMathTests, ClassifiesNearAndFarSphereEdges) {
  const Camera camera = makeTestCamera();
  const CameraFrameState frame = makeCameraFrameState(camera, 1.0f);
  const visibility_detail::FrustumPlanes frustum =
      visibility_detail::buildCameraFrustumPlanes(frame);

  EXPECT_EQ(visibility_detail::VisibilityClassification::Inside,
            visibility_detail::classifySphere(frustum,
                                              glm::vec3(0.0f, 0.0f, -1.0f),
                                              0.1f));
  EXPECT_EQ(visibility_detail::VisibilityClassification::Outside,
            visibility_detail::classifySphere(frustum,
                                              glm::vec3(0.0f, 0.0f, -10.25f),
                                              0.1f));
  EXPECT_EQ(visibility_detail::VisibilityClassification::Outside,
            visibility_detail::classifySphere(frustum,
                                              glm::vec3(0.0f, 0.0f, 0.25f),
                                              0.1f));
}

TEST(VisibilityMathTests, ClassifiesSidePlaneSphereEdges) {
  const Camera camera = makeTestCamera();
  const CameraFrameState frame = makeCameraFrameState(camera, 1.0f);
  const visibility_detail::FrustumPlanes frustum =
      visibility_detail::buildCameraFrustumPlanes(frame);

  EXPECT_EQ(visibility_detail::VisibilityClassification::Inside,
            visibility_detail::classifySphere(frustum,
                                              glm::vec3(0.0f, 0.0f, -2.0f),
                                              0.25f));
  EXPECT_EQ(visibility_detail::VisibilityClassification::Intersects,
            visibility_detail::classifySphere(frustum,
                                              glm::vec3(2.0f, 0.0f, -2.0f),
                                              0.25f));
  EXPECT_EQ(visibility_detail::VisibilityClassification::Outside,
            visibility_detail::classifySphere(frustum,
                                              glm::vec3(3.0f, 0.0f, -2.0f),
                                              0.25f));
}

TEST(VisibilityMathTests, ClassifiesTransformedAabbConservatively) {
  const Camera camera = makeTestCamera();
  const CameraFrameState frame = makeCameraFrameState(camera, 1.0f);
  const visibility_detail::FrustumPlanes frustum =
      visibility_detail::buildCameraFrustumPlanes(frame);
  const BoundingBox localBounds(glm::vec3(-0.5f), glm::vec3(0.5f));

  EXPECT_EQ(visibility_detail::VisibilityClassification::Inside,
            visibility_detail::classifyTransformedBounds(
                frustum, localBounds,
                glm::translate(glm::mat4(1.0f),
                               glm::vec3(0.0f, 0.0f, -2.0f))));
  EXPECT_EQ(visibility_detail::VisibilityClassification::Intersects,
            visibility_detail::classifyTransformedBounds(
                frustum, localBounds,
                glm::translate(glm::mat4(1.0f),
                               glm::vec3(2.0f, 0.0f, -2.0f))));
  EXPECT_EQ(visibility_detail::VisibilityClassification::Outside,
            visibility_detail::classifyTransformedBounds(
                frustum, localBounds,
                glm::translate(glm::mat4(1.0f),
                               glm::vec3(4.0f, 0.0f, -2.0f))));
}

TEST(VisibilityMathTests, SupportsZeroToOneDepthClipConvention) {
  const glm::mat4 viewProj = glm::orthoRH_ZO(-1.0f, 1.0f, -1.0f, 1.0f, 0.5f,
                                             5.0f);
  const visibility_detail::FrustumPlanes frustum =
      visibility_detail::buildFrustumPlanes(
          viewProj, visibility_detail::DepthClipConvention::ZeroToOne);

  EXPECT_EQ(visibility_detail::VisibilityClassification::Inside,
            visibility_detail::classifySphere(frustum,
                                              glm::vec3(0.0f, 0.0f, -1.0f),
                                              0.1f));
  EXPECT_EQ(visibility_detail::VisibilityClassification::Outside,
            visibility_detail::classifySphere(frustum,
                                              glm::vec3(0.0f, 0.0f, -5.25f),
                                              0.1f));
  EXPECT_EQ(visibility_detail::VisibilityClassification::Outside,
            visibility_detail::classifySphere(frustum,
                                              glm::vec3(0.0f, 0.0f, -0.25f),
                                              0.1f));
}

TEST(VisibilityMathTests, CpuEvaluationKeepsSubmeshCandidatesIndependent) {
  const Camera camera = makeTestCamera();
  const CameraFrameState frame = makeCameraFrameState(camera, 1.0f);
  const visibility_detail::FrustumPlanes frustum =
      visibility_detail::buildCameraFrustumPlanes(frame);

  VisibilityPassRequest request{};
  request.frustum = frustum;
  request.enableCpuFrustumCulling = true;

  std::array<VisibilityCandidate, 2> candidates{};
  candidates[0].renderableIndex = 0u;
  candidates[0].templateIndex = 3u;
  candidates[0].submeshIndex = 0u;
  candidates[0].localBounds =
      BoundingBox(glm::vec3(3.75f, -0.25f, -2.25f),
                  glm::vec3(4.25f, 0.25f, -1.75f));
  candidates[0].worldFromLocal = glm::mat4(1.0f);
  candidates[1].renderableIndex = 0u;
  candidates[1].templateIndex = 4u;
  candidates[1].submeshIndex = 1u;
  candidates[1].localBounds =
      BoundingBox(glm::vec3(-0.25f, -0.25f, -2.25f),
                  glm::vec3(0.25f, 0.25f, -1.75f));
  candidates[1].worldFromLocal = glm::mat4(1.0f);

  VisibilityFrameState state;
  const VisibilityPassResult result = state.evaluateCpu(request, candidates);

  ASSERT_EQ(result.visibleCandidateIndices.size(), 1u);
  const uint32_t visibleCandidateIndex = result.visibleCandidateIndices[0];
  ASSERT_LT(visibleCandidateIndex, candidates.size());
  EXPECT_EQ(candidates[visibleCandidateIndex].templateIndex, 4u);
  EXPECT_EQ(candidates[visibleCandidateIndex].submeshIndex, 1u);
  EXPECT_EQ(result.cpuRejected, 1u);
}

} // namespace nuri
