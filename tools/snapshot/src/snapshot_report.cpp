#include "nuri/tools/snapshot/snapshot_report.h"

#include "nuri/tools/snapshot/snapshot_capture_point.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string_view>

#include <yyjson.h>

namespace nuri::tools::snapshot {
namespace {

using JsonMutDocPtr =
    std::unique_ptr<yyjson_mut_doc, decltype(&yyjson_mut_doc_free)>;
using JsonDocPtr = std::unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)>;

void addString(yyjson_mut_doc *doc, yyjson_mut_val *object, const char *key,
               const std::string &value) {
  yyjson_mut_obj_add_strcpy(doc, object, key, value.c_str());
}

void addPath(yyjson_mut_doc *doc, yyjson_mut_val *object, const char *key,
             const std::filesystem::path &value) {
  addString(doc, object, key, value.generic_string());
}

yyjson_mut_val *makeStringArray(yyjson_mut_doc *doc,
                                const std::vector<std::string> &values) {
  yyjson_mut_val *array = yyjson_mut_arr(doc);
  for (const std::string &value : values) {
    yyjson_mut_arr_add_strcpy(doc, array, value.c_str());
  }
  return array;
}

yyjson_mut_val *makeEnvironmentObject(yyjson_mut_doc *doc,
                                      const SnapshotEnvironment &env) {
  yyjson_mut_val *object = yyjson_mut_obj(doc);
  addPath(doc, object, "repoRoot", env.repoRoot);
  addString(doc, object, "commitHash", env.commitHash);
  addString(doc, object, "branchName", env.branchName);
  yyjson_mut_obj_add_bool(doc, object, "dirty", env.dirty);
  addString(doc, object, "osName", env.osName);
  addString(doc, object, "osVersion", env.osVersion);
  addString(doc, object, "cpuName", env.cpuName);
  yyjson_mut_obj_add_uint(doc, object, "cpuLogicalThreadCount",
                          env.cpuLogicalThreadCount);
  addString(doc, object, "gpuBackend", env.gpuBackend);
  addString(doc, object, "gpuBackendSource", env.gpuBackendSource);
  addString(doc, object, "gpuDeviceName", env.gpuDeviceName);
  yyjson_mut_obj_add_uint(doc, object, "swapchainImageCount",
                          env.swapchainImageCount);
  addString(doc, object, "requestedPresentMode", env.requestedPresentMode);
  addString(doc, object, "resolvedPresentMode", env.resolvedPresentMode);
  addString(doc, object, "presentModeSource", env.presentModeSource);
  addString(doc, object, "requestedWindowMode", env.requestedWindowMode);
  addString(doc, object, "resolvedWindowMode", env.resolvedWindowMode);
  yyjson_mut_obj_add_bool(doc, object, "windowVisible", env.windowVisible);
  yyjson_mut_obj_add_uint(doc, object, "renderGraphWorkerCount",
                          env.renderGraphWorkerCount);
  yyjson_mut_obj_add_bool(doc, object, "renderGraphParallelCompile",
                          env.renderGraphParallelCompile);
  yyjson_mut_obj_add_bool(doc, object, "renderGraphParallelRecording",
                          env.renderGraphParallelRecording);
  addString(doc, object, "buildType", env.buildType);
  addString(doc, object, "cmakeToolProfile", env.cmakeToolProfile);
  addString(doc, object, "vcpkgManifestFeatures", env.vcpkgManifestFeatures);
  yyjson_mut_obj_add_bool(doc, object, "NURI_BUILD_SHARED", env.buildShared);
  yyjson_mut_obj_add_bool(doc, object, "NURI_WITH_LOGGING", env.loggingEnabled);
  yyjson_mut_obj_add_bool(doc, object, "NURI_WITH_ASSERTS", env.assertsEnabled);
  yyjson_mut_obj_add_bool(doc, object, "NURI_WITH_TRACY", env.tracyEnabled);
  yyjson_mut_obj_add_bool(doc, object, "NURI_WITH_TRACY_GPU",
                          env.tracyGpuEnabled);
  yyjson_mut_obj_add_bool(doc, object, "NURI_WITH_TRACY_GPU_DRAW_ZONES",
                          env.tracyGpuDrawZonesEnabled);
  yyjson_mut_obj_add_bool(doc, object, "devChecks", env.devChecks);
  return object;
}

yyjson_mut_val *makeCaseObject(yyjson_mut_doc *doc,
                               const SnapshotCase &snapshotCase) {
  yyjson_mut_val *object = yyjson_mut_obj(doc);
  yyjson_mut_obj_add_uint(doc, object, "schemaVersion",
                          snapshotCase.schemaVersion);
  addString(doc, object, "id", snapshotCase.id);
  addString(doc, object, "suite", snapshotCase.suite);
  addString(doc, object, "description", snapshotCase.description);
  addString(doc, object, "backend", snapshotCase.backend);
  yyjson_mut_val *resolution = yyjson_mut_arr(doc);
  yyjson_mut_arr_add_uint(doc, resolution, snapshotCase.resolution[0]);
  yyjson_mut_arr_add_uint(doc, resolution, snapshotCase.resolution[1]);
  yyjson_mut_obj_add_val(doc, object, "resolution", resolution);
  addString(doc, object, "presentMode", snapshotCase.presentMode);
  addString(doc, object, "windowMode", snapshotCase.windowMode);
  yyjson_mut_obj_add_uint(doc, object, "warmupFrames",
                          snapshotCase.warmupFrames);
  yyjson_mut_obj_add_uint(doc, object, "captureFrame",
                          snapshotCase.captureFrame);
  yyjson_mut_obj_add_bool(doc, object, "authoritative",
                          snapshotCase.authoritative);
  yyjson_mut_val *captures = yyjson_mut_arr(doc);
  for (const SnapshotCaptureTarget &capture : snapshotCase.captures) {
    yyjson_mut_val *entry = yyjson_mut_obj(doc);
    addString(doc, entry, "target", capture.name);
    addString(doc, entry, "profile", capture.profile);
    yyjson_mut_obj_add_bool(doc, entry, "required", capture.required);
    yyjson_mut_arr_add_val(captures, entry);
  }
  yyjson_mut_obj_add_val(doc, object, "captures", captures);
  return object;
}

yyjson_mut_val *makeMetricsObject(yyjson_mut_doc *doc,
                                  const SnapshotCompareMetrics &metrics) {
  yyjson_mut_val *object = yyjson_mut_obj(doc);
  yyjson_mut_obj_add_real(doc, object, "meanAbsError", metrics.meanAbsError);
  yyjson_mut_obj_add_real(doc, object, "rmse", metrics.rmse);
  yyjson_mut_obj_add_real(doc, object, "maxAbsError", metrics.maxAbsError);
  yyjson_mut_obj_add_real(doc, object, "p99AbsError", metrics.p99AbsError);
  yyjson_mut_obj_add_uint(doc, object, "failingValues", metrics.failingValues);
  yyjson_mut_obj_add_uint(doc, object, "comparedValues",
                          metrics.comparedValues);
  return object;
}

yyjson_mut_val *makeRendererMetricsObject(yyjson_mut_doc *doc,
                                          const RenderFrameMetrics &metrics) {
  yyjson_mut_val *object = yyjson_mut_obj(doc);
  yyjson_mut_obj_add_uint(doc, object, "frameIndex", metrics.frameIndex);

  const OpaqueFrameMetrics &opaque = metrics.opaque;
  yyjson_mut_val *opaqueObject = yyjson_mut_obj(doc);
  yyjson_mut_obj_add_uint(doc, opaqueObject, "totalInstances",
                          opaque.totalInstances);
  yyjson_mut_obj_add_uint(doc, opaqueObject, "visibleInstances",
                          opaque.visibleInstances);
  yyjson_mut_obj_add_uint(doc, opaqueObject, "instancedDraws",
                          opaque.instancedDraws);
  yyjson_mut_obj_add_uint(doc, opaqueObject, "indirectDrawCalls",
                          opaque.indirectDrawCalls);
  yyjson_mut_obj_add_uint(doc, opaqueObject, "depthPrepassDraws",
                          opaque.depthPrepassDraws);
  yyjson_mut_obj_add_uint(doc, opaqueObject, "computeDispatches",
                          opaque.computeDispatches);
  yyjson_mut_obj_add_uint(doc, opaqueObject, "meshletDispatches",
                          opaque.meshletDispatches);
  yyjson_mut_obj_add_uint(doc, opaqueObject, "meshletTaskGroups",
                          opaque.meshletTaskGroups);
  yyjson_mut_obj_add_uint(doc, opaqueObject, "meshletCandidateCount",
                          opaque.meshletCandidateCount);
  yyjson_mut_obj_add_uint(doc, opaqueObject, "meshletModeRequired",
                          opaque.meshletModeRequired);
  yyjson_mut_obj_add_uint(doc, opaqueObject, "meshletModeActive",
                          opaque.meshletModeActive);
  yyjson_mut_obj_add_uint(doc, opaqueObject, "meshletRejectedMissingFeature",
                          opaque.meshletRejectedMissingFeature);
  yyjson_mut_obj_add_uint(doc, opaqueObject, "meshletRejectedMissingAssetData",
                          opaque.meshletRejectedMissingAssetData);
  yyjson_mut_obj_add_uint(doc, opaqueObject, "meshletRejectedIncompatibleFrame",
                          opaque.meshletRejectedIncompatibleFrame);
  yyjson_mut_obj_add_val(doc, object, "opaque", opaqueObject);

  const VisibilityFrameMetrics &visibility = metrics.visibility;
  yyjson_mut_val *visibilityObject = yyjson_mut_obj(doc);
  yyjson_mut_obj_add_uint(doc, visibilityObject, "cpuMainCandidates",
                          visibility.cpuMainCandidates);
  yyjson_mut_obj_add_uint(doc, visibilityObject, "cpuMainVisibleCandidates",
                          visibility.cpuMainVisibleCandidates);
  yyjson_mut_obj_add_uint(doc, visibilityObject, "cpuMainRejected",
                          visibility.cpuMainRejected);
  yyjson_mut_obj_add_uint(doc, visibilityObject, "gpuMainCandidates",
                          visibility.gpuMainCandidates);
  yyjson_mut_obj_add_uint(doc, visibilityObject, "gpuMainVisibleCandidates",
                          visibility.gpuMainVisibleCandidates);
  yyjson_mut_obj_add_uint(doc, visibilityObject, "gpuMainRejectedFrustum",
                          visibility.gpuMainRejectedFrustum);
  yyjson_mut_obj_add_uint(doc, visibilityObject, "gpuMainRejectedOcclusion",
                          visibility.gpuMainRejectedOcclusion);
  yyjson_mut_obj_add_uint(doc, visibilityObject, "gpuOutputOverflowCount",
                          visibility.gpuOutputOverflowCount);
  yyjson_mut_obj_add_uint(doc, visibilityObject, "gpuMainReadbackAvailable",
                          visibility.gpuMainReadbackAvailable);
  yyjson_mut_obj_add_uint(doc, visibilityObject, "gpuMainReadbackSourceFrame",
                          visibility.gpuMainReadbackSourceFrame);
  yyjson_mut_obj_add_uint(doc, visibilityObject,
                          "gpuMainReadbackStaleFrameCount",
                          visibility.gpuMainReadbackStaleFrameCount);
  yyjson_mut_obj_add_uint(doc, visibilityObject, "gpuMainReadbackErrorCount",
                          visibility.gpuMainReadbackErrorCount);
  yyjson_mut_obj_add_uint(doc, visibilityObject,
                          "gpuMainReadbackVisibleCandidates",
                          visibility.gpuMainReadbackVisibleCandidates);
  yyjson_mut_obj_add_uint(doc, visibilityObject, "gpuMainVisibleListMismatches",
                          visibility.gpuMainVisibleListMismatches);
  yyjson_mut_obj_add_uint(doc, visibilityObject, "gpuIndirectDrawUsed",
                          visibility.gpuIndirectDrawUsed);
  yyjson_mut_obj_add_uint(doc, visibilityObject, "gpuIndirectDrawFallback",
                          visibility.gpuIndirectDrawFallback);
  yyjson_mut_obj_add_uint(doc, visibilityObject, "gpuIndirectDrawCommands",
                          visibility.gpuIndirectDrawCommands);
  yyjson_mut_obj_add_uint(doc, visibilityObject,
                          "gpuIndirectDrawReadbackCommands",
                          visibility.gpuIndirectDrawReadbackCommands);
  yyjson_mut_obj_add_uint(doc, visibilityObject,
                          "gpuIndirectDrawReadbackTombstoned",
                          visibility.gpuIndirectDrawReadbackTombstoned);
  yyjson_mut_obj_add_uint(doc, visibilityObject,
                          "gpuIndirectDrawReadbackVisible",
                          visibility.gpuIndirectDrawReadbackVisible);
  yyjson_mut_obj_add_uint(doc, visibilityObject, "indirectMeshDispatchCount",
                          visibility.indirectMeshDispatchCount);
  yyjson_mut_obj_add_uint(doc, visibilityObject, "meshletRejectedFrustum",
                          visibility.meshletRejectedFrustum);
  yyjson_mut_obj_add_uint(doc, visibilityObject, "meshletRejectedCone",
                          visibility.meshletRejectedCone);
  yyjson_mut_obj_add_uint(doc, visibilityObject, "meshletRejectedOcclusion",
                          visibility.meshletRejectedOcclusion);
  yyjson_mut_obj_add_uint(doc, visibilityObject, "meshletOcclusionAvailable",
                          visibility.meshletOcclusionAvailable);
  yyjson_mut_obj_add_uint(doc, visibilityObject, "meshletPayloadOverflowCount",
                          visibility.meshletPayloadOverflowCount);
  yyjson_mut_obj_add_uint(doc, visibilityObject, "meshletReadbackAvailable",
                          visibility.meshletReadbackAvailable);
  yyjson_mut_obj_add_uint(doc, visibilityObject, "meshletReadbackSourceFrame",
                          visibility.meshletReadbackSourceFrame);
  yyjson_mut_obj_add_uint(doc, visibilityObject,
                          "meshletReadbackStaleFrameCount",
                          visibility.meshletReadbackStaleFrameCount);
  yyjson_mut_obj_add_uint(doc, visibilityObject, "meshletReadbackErrorCount",
                          visibility.meshletReadbackErrorCount);
  yyjson_mut_obj_add_uint(doc, visibilityObject, "meshletEmitted",
                          visibility.meshletEmitted);
  yyjson_mut_obj_add_uint(doc, visibilityObject, "meshletTaskGroupsExecuted",
                          visibility.meshletTaskGroupsExecuted);
  yyjson_mut_obj_add_uint(doc, visibilityObject, "uncertainVisible",
                          visibility.uncertainVisible);
  yyjson_mut_obj_add_uint(doc, visibilityObject, "shadowCpuCandidates",
                          visibility.shadowCpuCandidates);
  yyjson_mut_obj_add_uint(doc, visibilityObject, "shadowCpuRejected",
                          visibility.shadowCpuRejected);
  yyjson_mut_obj_add_uint(doc, visibilityObject, "shadowMeshletCandidates",
                          visibility.shadowMeshletCandidates);
  yyjson_mut_obj_add_uint(doc, visibilityObject,
                          "shadowMeshletReadbackAvailable",
                          visibility.shadowMeshletReadbackAvailable);
  yyjson_mut_obj_add_uint(doc, visibilityObject,
                          "shadowMeshletReadbackSourceFrame",
                          visibility.shadowMeshletReadbackSourceFrame);
  yyjson_mut_obj_add_uint(doc, visibilityObject,
                          "shadowMeshletReadbackStaleFrameCount",
                          visibility.shadowMeshletReadbackStaleFrameCount);
  yyjson_mut_obj_add_uint(doc, visibilityObject,
                          "shadowMeshletReadbackErrorCount",
                          visibility.shadowMeshletReadbackErrorCount);
  yyjson_mut_obj_add_uint(doc, visibilityObject, "shadowMeshletRejectedBounds",
                          visibility.shadowMeshletRejectedBounds);
  yyjson_mut_obj_add_uint(doc, visibilityObject, "occlusionAvailable",
                          visibility.occlusionAvailable);
  yyjson_mut_obj_add_val(doc, object, "visibility", visibilityObject);

  const ShadowFrameMetrics &shadow = metrics.shadow;
  yyjson_mut_val *shadowObject = yyjson_mut_obj(doc);
  yyjson_mut_obj_add_uint(doc, shadowObject, "cascadeCount",
                          shadow.cascadeCount);
  yyjson_mut_obj_add_uint(doc, shadowObject, "shadowMapSize",
                          shadow.shadowMapSize);
  yyjson_mut_obj_add_uint(doc, shadowObject, "totalDraws", shadow.totalDraws);
  yyjson_mut_obj_add_uint(doc, shadowObject, "totalCulledDraws",
                          shadow.totalCulledDraws);
  yyjson_mut_obj_add_uint(doc, shadowObject, "meshletDispatchCount",
                          shadow.shadowMeshletDispatchCount);
  yyjson_mut_obj_add_uint(doc, shadowObject, "meshletTaskGroupCount",
                          shadow.shadowMeshletTaskGroupCount);
  yyjson_mut_obj_add_uint(doc, shadowObject, "staticCasterEntries",
                          shadow.staticCasterEntries);
  yyjson_mut_obj_add_uint(doc, shadowObject, "dynamicCasterEntries",
                          shadow.dynamicCasterEntries);
  yyjson_mut_obj_add_uint(doc, shadowObject, "staticCacheReused",
                          shadow.staticCacheReused);
  yyjson_mut_obj_add_real(doc, shadowObject, "minCascadeTexelWorldSize",
                          shadow.minCascadeTexelWorldSize);
  yyjson_mut_obj_add_real(doc, shadowObject, "averageCascadeTexelWorldSize",
                          shadow.averageCascadeTexelWorldSize);
  yyjson_mut_obj_add_real(doc, shadowObject, "maxCascadeTexelWorldSize",
                          shadow.maxCascadeTexelWorldSize);
  yyjson_mut_obj_add_val(doc, object, "shadow", shadowObject);

  const AntiAliasingFrameMetrics &aa = metrics.antiAliasing;
  yyjson_mut_val *aaObject = yyjson_mut_obj(doc);
  yyjson_mut_obj_add_bool(doc, aaObject, "historyValid", aa.historyValid);
  yyjson_mut_obj_add_bool(doc, aaObject, "temporalDataValid",
                          aa.temporalDataValid);
  yyjson_mut_obj_add_strcpy(
      doc, aaObject, "historyResetReason",
      std::string(temporalHistoryResetReasonName(aa.historyResetReason))
          .c_str());
  yyjson_mut_obj_add_uint(doc, aaObject, "historyResetCount",
                          aa.historyResetCount);
  yyjson_mut_obj_add_uint(doc, aaObject, "framesSinceHistoryReset",
                          aa.framesSinceHistoryReset);
  yyjson_mut_obj_add_real(doc, aaObject, "cameraPositionDelta",
                          aa.cameraPositionDelta);
  yyjson_mut_obj_add_real(doc, aaObject, "jitterDeltaMagnitude",
                          aa.jitterDeltaMagnitude);
  addString(doc, aaObject, "motionVectorFormat",
            snapshotFormatName(aa.motionVectorFormat));
  yyjson_mut_obj_add_uint(doc, aaObject, "motionVectorWidth",
                          aa.motionVectorWidth);
  yyjson_mut_obj_add_uint(doc, aaObject, "motionVectorHeight",
                          aa.motionVectorHeight);
  yyjson_mut_obj_add_uint(doc, aaObject, "motionVectorTextureCount",
                          aa.motionVectorTextureCount);
  yyjson_mut_obj_add_uint(doc, aaObject, "motionVectorClearPassCount",
                          aa.motionVectorClearPassCount);
  yyjson_mut_obj_add_uint(doc, aaObject,
                          "motionVectorDepthReprojectionPassCount",
                          aa.motionVectorDepthReprojectionPassCount);
  yyjson_mut_obj_add_bool(doc, aaObject, "motionVectorGraphPublished",
                          aa.motionVectorGraphPublished);
  yyjson_mut_obj_add_bool(doc, aaObject,
                          "motionVectorDepthReprojectionGenerated",
                          aa.motionVectorDepthReprojectionGenerated);
  yyjson_mut_obj_add_uint(doc, aaObject, "velocityPassCount",
                          aa.velocityPassCount);
  yyjson_mut_obj_add_uint(doc, aaObject, "velocityDrawCount",
                          aa.velocityDrawCount);
  yyjson_mut_obj_add_uint(doc, aaObject, "velocityInstanceCount",
                          aa.velocityInstanceCount);
  yyjson_mut_obj_add_uint(doc, aaObject, "velocityPreviousTransformValidCount",
                          aa.velocityPreviousTransformValidCount);
  yyjson_mut_obj_add_uint(doc, aaObject,
                          "velocityMissingPreviousTransformCount",
                          aa.velocityMissingPreviousTransformCount);
  yyjson_mut_obj_add_uint(doc, aaObject, "velocityTessellatedSkippedDrawCount",
                          aa.velocityTessellatedSkippedDrawCount);
  yyjson_mut_obj_add_bool(doc, aaObject, "previousTransformCacheValid",
                          aa.previousTransformCacheValid);
  yyjson_mut_obj_add_bool(doc, aaObject, "opaqueVelocityGenerated",
                          aa.opaqueVelocityGenerated);
  yyjson_mut_obj_add_real(doc, aaObject, "velocityAverageObjectMotion",
                          aa.velocityAverageObjectMotion);
  yyjson_mut_obj_add_real(doc, aaObject, "velocityMaxObjectMotion",
                          aa.velocityMaxObjectMotion);
  yyjson_mut_obj_add_real(doc, aaObject, "velocityCameraMatrixDelta",
                          aa.velocityCameraMatrixDelta);
  yyjson_mut_obj_add_real(doc, aaObject, "velocityStaticResidualEstimate",
                          aa.velocityStaticResidualEstimate);
  yyjson_mut_obj_add_real(doc, aaObject, "velocityEstimatedAverageMagnitude",
                          aa.velocityEstimatedAverageMagnitude);
  yyjson_mut_obj_add_real(doc, aaObject, "velocityEstimatedMaxMagnitude",
                          aa.velocityEstimatedMaxMagnitude);
  yyjson_mut_obj_add_uint(doc, aaObject, "reactiveMaskPassCount",
                          aa.reactiveMaskPassCount);
  yyjson_mut_obj_add_uint(doc, aaObject, "reactiveMaskDrawCount",
                          aa.reactiveMaskDrawCount);
  yyjson_mut_obj_add_uint(doc, aaObject,
                          "transparentTransmissionFeedbackRefreshCount",
                          aa.transparentTransmissionFeedbackRefreshCount);
  yyjson_mut_obj_add_uint(doc, aaObject,
                          "transparentTransmissionBlendDrawCount",
                          aa.transparentTransmissionBlendDrawCount);
  yyjson_mut_obj_add_uint(doc, aaObject,
                          "transparentTransmissionFeedbackSourceAvailable",
                          aa.transparentTransmissionFeedbackSourceAvailable);
  yyjson_mut_obj_add_uint(doc, aaObject, "taaTransparentPostTaaDrawCount",
                          aa.taaTransparentPostTaaDrawCount);
  yyjson_mut_obj_add_uint(doc, aaObject, "taaTransparentPostSpatialAAPassCount",
                          aa.taaTransparentPostSpatialAAPassCount);
  yyjson_mut_obj_add_uint(doc, aaObject, "taaResolvePassCount",
                          aa.taaResolvePassCount);
  yyjson_mut_obj_add_val(doc, object, "antiAliasing", aaObject);

  const AmbientOcclusionFrameMetrics &ao = metrics.ambientOcclusion;
  yyjson_mut_val *aoObject = yyjson_mut_obj(doc);
  yyjson_mut_obj_add_bool(doc, aoObject, "enabled", ao.enabled);
  yyjson_mut_obj_add_bool(doc, aoObject, "active", ao.active);
  yyjson_mut_obj_add_uint(doc, aoObject, "mainPassCount", ao.mainPassCount);
  yyjson_mut_obj_add_uint(doc, aoObject, "temporalPassCount",
                          ao.temporalPassCount);
  yyjson_mut_obj_add_bool(doc, aoObject, "temporalAccumulationActive",
                          ao.temporalAccumulationActive);
  yyjson_mut_obj_add_bool(doc, aoObject, "temporalMotionVectorsConsumed",
                          ao.temporalMotionVectorsConsumed);
  yyjson_mut_obj_add_val(doc, object, "ambientOcclusion", aoObject);

  const HDRPostProcessFrameMetrics &hdr = metrics.hdrPostProcess;
  yyjson_mut_val *hdrObject = yyjson_mut_obj(doc);
  yyjson_mut_obj_add_bool(doc, hdrObject, "bloomEnabled", hdr.bloomEnabled);
  yyjson_mut_obj_add_bool(doc, hdrObject, "bloomActive", hdr.bloomActive);
  yyjson_mut_obj_add_uint(doc, hdrObject, "bloomPassCount", hdr.bloomPassCount);
  yyjson_mut_obj_add_bool(doc, hdrObject, "adaptationEnabled",
                          hdr.adaptationEnabled);
  yyjson_mut_obj_add_bool(doc, hdrObject, "adaptationActive",
                          hdr.adaptationActive);
  yyjson_mut_obj_add_val(doc, object, "hdrPostProcess", hdrObject);

  const RenderFrameMetrics::TransparentFrameMetrics &transparent =
      metrics.transparent;
  yyjson_mut_val *transparentObject = yyjson_mut_obj(doc);
  yyjson_mut_obj_add_uint(doc, transparentObject, "meshDraws",
                          transparent.meshDraws);
  yyjson_mut_obj_add_uint(doc, transparentObject, "contributorSortableDraws",
                          transparent.contributorSortableDraws);
  yyjson_mut_obj_add_uint(doc, transparentObject, "contributorFixedDraws",
                          transparent.contributorFixedDraws);
  yyjson_mut_obj_add_uint(doc, transparentObject, "pickDraws",
                          transparent.pickDraws);
  yyjson_mut_obj_add_val(doc, object, "transparent", transparentObject);

  return object;
}

yyjson_mut_val *makeCaptureObject(yyjson_mut_doc *doc,
                                  const SnapshotCaptureReport &capture) {
  yyjson_mut_val *object = yyjson_mut_obj(doc);
  addString(doc, object, "target", capture.target);
  addString(doc, object, "artifactStem", capture.artifactStem);
  addString(doc, object, "profile", capture.profile);
  yyjson_mut_obj_add_bool(doc, object, "required", capture.required);
  yyjson_mut_obj_add_bool(doc, object, "available", capture.available);
  yyjson_mut_obj_add_uint(doc, object, "capturePointVersion",
                          capture.capturePointVersion);
  yyjson_mut_obj_add_uint(doc, object, "captureFrameIndex",
                          capture.captureFrameIndex);
  addString(doc, object, "lifetime", capture.lifetime);
  addString(doc, object, "format", capture.format);
  addString(doc, object, "colorSpace", capture.colorSpace);
  addString(doc, object, "origin", capture.origin);
  yyjson_mut_obj_add_uint(doc, object, "width", capture.width);
  yyjson_mut_obj_add_uint(doc, object, "height", capture.height);
  yyjson_mut_obj_add_uint(doc, object, "mip", capture.mip);
  yyjson_mut_obj_add_uint(doc, object, "layer", capture.layer);
  addString(doc, object, "actualHash", capture.actualHash);
  addString(doc, object, "expectedHash", capture.expectedHash);
  addPath(doc, object, "actual", capture.actual);
  addPath(doc, object, "actualMetadata", capture.actualMetadata);
  addPath(doc, object, "preview", capture.preview);
  addPath(doc, object, "expected", capture.expected);
  addPath(doc, object, "diff", capture.diff);
  yyjson_mut_obj_add_val(doc, object, "metrics",
                         makeMetricsObject(doc, capture.metrics));
  yyjson_mut_obj_add_val(doc, object, "failedThresholds",
                         makeStringArray(doc, capture.failedThresholds));
  addString(doc, object, "producerPassLabel", capture.producerPassLabel);
  addString(doc, object, "readbackError", capture.readbackError);
  addString(doc, object, "status", capture.status);
  addString(doc, object, "statusReason", capture.statusReason);
  return object;
}

[[nodiscard]] std::string readString(yyjson_val *object, const char *key,
                                     std::string defaultValue = {}) {
  yyjson_val *value = yyjson_obj_get(object, key);
  if (!yyjson_is_str(value)) {
    return defaultValue;
  }
  return std::string(yyjson_get_str(value), yyjson_get_len(value));
}

[[nodiscard]] uint32_t readU32(yyjson_val *object, const char *key,
                               uint32_t defaultValue = 0u) {
  yyjson_val *value = yyjson_obj_get(object, key);
  return yyjson_is_uint(value) ? static_cast<uint32_t>(yyjson_get_uint(value))
                               : defaultValue;
}

[[nodiscard]] uint64_t readU64(yyjson_val *object, const char *key,
                               uint64_t defaultValue = 0u) {
  yyjson_val *value = yyjson_obj_get(object, key);
  return yyjson_is_uint(value) ? yyjson_get_uint(value) : defaultValue;
}

[[nodiscard]] float readF32(yyjson_val *object, const char *key,
                            float defaultValue = 0.0f) {
  yyjson_val *value = yyjson_obj_get(object, key);
  return yyjson_is_num(value) ? static_cast<float>(yyjson_get_num(value))
                              : defaultValue;
}

[[nodiscard]] bool readBool(yyjson_val *object, const char *key,
                            bool defaultValue) {
  yyjson_val *value = yyjson_obj_get(object, key);
  return yyjson_is_bool(value) ? yyjson_get_bool(value) : defaultValue;
}

[[nodiscard]] Format readFormatName(std::string_view value) {
  if (value == "R32_UINT") {
    return Format::R32_UINT;
  }
  if (value == "RGBA8_UNORM") {
    return Format::RGBA8_UNORM;
  }
  if (value == "RGBA8_SRGB") {
    return Format::RGBA8_SRGB;
  }
  if (value == "RGBA8_UINT") {
    return Format::RGBA8_UINT;
  }
  if (value == "RGBA16_FLOAT") {
    return Format::RGBA16_FLOAT;
  }
  if (value == "RGBA32_FLOAT") {
    return Format::RGBA32_FLOAT;
  }
  if (value == "D32_FLOAT") {
    return Format::D32_FLOAT;
  }
  if (value == "R32_FLOAT") {
    return Format::R32_FLOAT;
  }
  if (value == "RG32_FLOAT") {
    return Format::RG32_FLOAT;
  }
  if (value == "D16_UNORM") {
    return Format::D16_UNORM;
  }
  if (value == "RG16_FLOAT") {
    return Format::RG16_FLOAT;
  }
  if (value == "R8_UNORM") {
    return Format::R8_UNORM;
  }
  if (value == "R16_UNORM") {
    return Format::R16_UNORM;
  }
  if (value == "BC7_RGBA_UNORM") {
    return Format::BC7_RGBA_UNORM;
  }
  if (value == "BC7_RGBA_SRGB") {
    return Format::BC7_RGBA_SRGB;
  }
  if (value == "ETC2_RGB8_UNORM") {
    return Format::ETC2_RGB8_UNORM;
  }
  if (value == "ETC2_RGB8_SRGB") {
    return Format::ETC2_RGB8_SRGB;
  }
  return Format::Count;
}

[[nodiscard]] TemporalHistoryResetReason
readTemporalHistoryResetReason(std::string_view value) {
  if (value == "First Frame") {
    return TemporalHistoryResetReason::FirstFrame;
  }
  if (value == "History Reset Requested") {
    return TemporalHistoryResetReason::HistoryResetRequested;
  }
  if (value == "AA Mode Changed") {
    return TemporalHistoryResetReason::AntiAliasingModeChanged;
  }
  if (value == "Resize") {
    return TemporalHistoryResetReason::Resize;
  }
  if (value == "Projection Changed") {
    return TemporalHistoryResetReason::ProjectionChanged;
  }
  if (value == "Render Scale Changed") {
    return TemporalHistoryResetReason::RenderScaleChanged;
  }
  if (value == "Camera Cut") {
    return TemporalHistoryResetReason::CameraCut;
  }
  if (value == "Invalid History Texture") {
    return TemporalHistoryResetReason::InvalidHistoryTexture;
  }
  if (value == "Scene Content Changed") {
    return TemporalHistoryResetReason::SceneContentChanged;
  }
  return TemporalHistoryResetReason::None;
}

void readRendererMetrics(yyjson_val *object, RenderFrameMetrics &metrics) {
  if (!yyjson_is_obj(object)) {
    return;
  }

  metrics.frameIndex = readU64(object, "frameIndex", metrics.frameIndex);

  yyjson_val *opaqueObject = yyjson_obj_get(object, "opaque");
  if (yyjson_is_obj(opaqueObject)) {
    OpaqueFrameMetrics &opaque = metrics.opaque;
    opaque.totalInstances =
        readU32(opaqueObject, "totalInstances", opaque.totalInstances);
    opaque.visibleInstances =
        readU32(opaqueObject, "visibleInstances", opaque.visibleInstances);
    opaque.instancedDraws =
        readU32(opaqueObject, "instancedDraws", opaque.instancedDraws);
    opaque.indirectDrawCalls =
        readU32(opaqueObject, "indirectDrawCalls", opaque.indirectDrawCalls);
    opaque.depthPrepassDraws =
        readU32(opaqueObject, "depthPrepassDraws", opaque.depthPrepassDraws);
    opaque.computeDispatches =
        readU32(opaqueObject, "computeDispatches", opaque.computeDispatches);
    opaque.meshletDispatches =
        readU32(opaqueObject, "meshletDispatches", opaque.meshletDispatches);
    opaque.meshletTaskGroups =
        readU32(opaqueObject, "meshletTaskGroups", opaque.meshletTaskGroups);
    opaque.meshletCandidateCount = readU32(
        opaqueObject, "meshletCandidateCount", opaque.meshletCandidateCount);
    opaque.meshletModeRequired = readU32(opaqueObject, "meshletModeRequired",
                                         opaque.meshletModeRequired);
    opaque.meshletModeActive =
        readU32(opaqueObject, "meshletModeActive", opaque.meshletModeActive);
    opaque.meshletRejectedMissingFeature =
        readU32(opaqueObject, "meshletRejectedMissingFeature",
                opaque.meshletRejectedMissingFeature);
    opaque.meshletRejectedMissingAssetData =
        readU32(opaqueObject, "meshletRejectedMissingAssetData",
                opaque.meshletRejectedMissingAssetData);
    opaque.meshletRejectedIncompatibleFrame =
        readU32(opaqueObject, "meshletRejectedIncompatibleFrame",
                opaque.meshletRejectedIncompatibleFrame);
  }

  yyjson_val *visibilityObject = yyjson_obj_get(object, "visibility");
  if (yyjson_is_obj(visibilityObject)) {
    VisibilityFrameMetrics &visibility = metrics.visibility;
    visibility.cpuMainCandidates = readU32(
        visibilityObject, "cpuMainCandidates", visibility.cpuMainCandidates);
    visibility.cpuMainVisibleCandidates =
        readU32(visibilityObject, "cpuMainVisibleCandidates",
                visibility.cpuMainVisibleCandidates);
    visibility.cpuMainRejected = readU32(visibilityObject, "cpuMainRejected",
                                         visibility.cpuMainRejected);
    visibility.gpuMainCandidates = readU32(
        visibilityObject, "gpuMainCandidates", visibility.gpuMainCandidates);
    visibility.gpuMainVisibleCandidates =
        readU32(visibilityObject, "gpuMainVisibleCandidates",
                visibility.gpuMainVisibleCandidates);
    visibility.gpuMainRejectedFrustum =
        readU32(visibilityObject, "gpuMainRejectedFrustum",
                visibility.gpuMainRejectedFrustum);
    visibility.gpuMainRejectedOcclusion =
        readU32(visibilityObject, "gpuMainRejectedOcclusion",
                visibility.gpuMainRejectedOcclusion);
    visibility.gpuOutputOverflowCount =
        readU32(visibilityObject, "gpuOutputOverflowCount",
                visibility.gpuOutputOverflowCount);
    visibility.gpuMainReadbackAvailable =
        readU32(visibilityObject, "gpuMainReadbackAvailable",
                visibility.gpuMainReadbackAvailable);
    visibility.gpuMainReadbackSourceFrame =
        readU32(visibilityObject, "gpuMainReadbackSourceFrame",
                visibility.gpuMainReadbackSourceFrame);
    visibility.gpuMainReadbackStaleFrameCount =
        readU32(visibilityObject, "gpuMainReadbackStaleFrameCount",
                visibility.gpuMainReadbackStaleFrameCount);
    visibility.gpuMainReadbackErrorCount =
        readU32(visibilityObject, "gpuMainReadbackErrorCount",
                visibility.gpuMainReadbackErrorCount);
    visibility.gpuMainReadbackVisibleCandidates =
        readU32(visibilityObject, "gpuMainReadbackVisibleCandidates",
                visibility.gpuMainReadbackVisibleCandidates);
    visibility.gpuMainVisibleListMismatches =
        readU32(visibilityObject, "gpuMainVisibleListMismatches",
                visibility.gpuMainVisibleListMismatches);
    visibility.gpuIndirectDrawUsed =
        readU32(visibilityObject, "gpuIndirectDrawUsed",
                visibility.gpuIndirectDrawUsed);
    visibility.gpuIndirectDrawFallback =
        readU32(visibilityObject, "gpuIndirectDrawFallback",
                visibility.gpuIndirectDrawFallback);
    visibility.gpuIndirectDrawCommands =
        readU32(visibilityObject, "gpuIndirectDrawCommands",
                visibility.gpuIndirectDrawCommands);
    visibility.gpuIndirectDrawReadbackCommands =
        readU32(visibilityObject, "gpuIndirectDrawReadbackCommands",
                visibility.gpuIndirectDrawReadbackCommands);
    visibility.gpuIndirectDrawReadbackTombstoned =
        readU32(visibilityObject, "gpuIndirectDrawReadbackTombstoned",
                visibility.gpuIndirectDrawReadbackTombstoned);
    visibility.gpuIndirectDrawReadbackVisible =
        readU32(visibilityObject, "gpuIndirectDrawReadbackVisible",
                visibility.gpuIndirectDrawReadbackVisible);
    visibility.indirectMeshDispatchCount =
        readU32(visibilityObject, "indirectMeshDispatchCount",
                visibility.indirectMeshDispatchCount);
    visibility.meshletRejectedFrustum =
        readU32(visibilityObject, "meshletRejectedFrustum",
                visibility.meshletRejectedFrustum);
    visibility.meshletRejectedCone =
        readU32(visibilityObject, "meshletRejectedCone",
                visibility.meshletRejectedCone);
    visibility.meshletRejectedOcclusion =
        readU32(visibilityObject, "meshletRejectedOcclusion",
                visibility.meshletRejectedOcclusion);
    visibility.meshletOcclusionAvailable =
        readU32(visibilityObject, "meshletOcclusionAvailable",
                visibility.meshletOcclusionAvailable);
    visibility.meshletPayloadOverflowCount =
        readU32(visibilityObject, "meshletPayloadOverflowCount",
                visibility.meshletPayloadOverflowCount);
    visibility.meshletReadbackAvailable =
        readU32(visibilityObject, "meshletReadbackAvailable",
                visibility.meshletReadbackAvailable);
    visibility.meshletReadbackSourceFrame =
        readU32(visibilityObject, "meshletReadbackSourceFrame",
                visibility.meshletReadbackSourceFrame);
    visibility.meshletReadbackStaleFrameCount =
        readU32(visibilityObject, "meshletReadbackStaleFrameCount",
                visibility.meshletReadbackStaleFrameCount);
    visibility.meshletReadbackErrorCount =
        readU32(visibilityObject, "meshletReadbackErrorCount",
                visibility.meshletReadbackErrorCount);
    visibility.meshletEmitted =
        readU32(visibilityObject, "meshletEmitted", visibility.meshletEmitted);
    visibility.meshletTaskGroupsExecuted =
        readU32(visibilityObject, "meshletTaskGroupsExecuted",
                visibility.meshletTaskGroupsExecuted);
    visibility.uncertainVisible = readU32(visibilityObject, "uncertainVisible",
                                          visibility.uncertainVisible);
    visibility.shadowCpuCandidates =
        readU32(visibilityObject, "shadowCpuCandidates",
                visibility.shadowCpuCandidates);
    visibility.shadowCpuRejected = readU32(
        visibilityObject, "shadowCpuRejected", visibility.shadowCpuRejected);
    visibility.shadowMeshletCandidates =
        readU32(visibilityObject, "shadowMeshletCandidates",
                visibility.shadowMeshletCandidates);
    visibility.shadowMeshletReadbackAvailable =
        readU32(visibilityObject, "shadowMeshletReadbackAvailable",
                visibility.shadowMeshletReadbackAvailable);
    visibility.shadowMeshletReadbackSourceFrame =
        readU32(visibilityObject, "shadowMeshletReadbackSourceFrame",
                visibility.shadowMeshletReadbackSourceFrame);
    visibility.shadowMeshletReadbackStaleFrameCount =
        readU32(visibilityObject, "shadowMeshletReadbackStaleFrameCount",
                visibility.shadowMeshletReadbackStaleFrameCount);
    visibility.shadowMeshletReadbackErrorCount =
        readU32(visibilityObject, "shadowMeshletReadbackErrorCount",
                visibility.shadowMeshletReadbackErrorCount);
    visibility.shadowMeshletRejectedBounds =
        readU32(visibilityObject, "shadowMeshletRejectedBounds",
                visibility.shadowMeshletRejectedBounds);
    visibility.occlusionAvailable = readU32(
        visibilityObject, "occlusionAvailable", visibility.occlusionAvailable);
  }

  yyjson_val *shadowObject = yyjson_obj_get(object, "shadow");
  if (yyjson_is_obj(shadowObject)) {
    ShadowFrameMetrics &shadow = metrics.shadow;
    shadow.cascadeCount =
        readU32(shadowObject, "cascadeCount", shadow.cascadeCount);
    shadow.shadowMapSize =
        readU32(shadowObject, "shadowMapSize", shadow.shadowMapSize);
    shadow.totalDraws = readU32(shadowObject, "totalDraws", shadow.totalDraws);
    shadow.totalCulledDraws =
        readU32(shadowObject, "totalCulledDraws", shadow.totalCulledDraws);
    shadow.shadowMeshletDispatchCount =
        readU32(shadowObject, "meshletDispatchCount",
                shadow.shadowMeshletDispatchCount);
    shadow.shadowMeshletTaskGroupCount =
        readU32(shadowObject, "meshletTaskGroupCount",
                shadow.shadowMeshletTaskGroupCount);
    shadow.staticCasterEntries = readU32(shadowObject, "staticCasterEntries",
                                         shadow.staticCasterEntries);
    shadow.dynamicCasterEntries = readU32(shadowObject, "dynamicCasterEntries",
                                          shadow.dynamicCasterEntries);
    shadow.staticCacheReused =
        readU32(shadowObject, "staticCacheReused", shadow.staticCacheReused);
    shadow.minCascadeTexelWorldSize =
        readF32(shadowObject, "minCascadeTexelWorldSize",
                shadow.minCascadeTexelWorldSize);
    shadow.averageCascadeTexelWorldSize =
        readF32(shadowObject, "averageCascadeTexelWorldSize",
                shadow.averageCascadeTexelWorldSize);
    shadow.maxCascadeTexelWorldSize =
        readF32(shadowObject, "maxCascadeTexelWorldSize",
                shadow.maxCascadeTexelWorldSize);
  }

  yyjson_val *aaObject = yyjson_obj_get(object, "antiAliasing");
  if (yyjson_is_obj(aaObject)) {
    AntiAliasingFrameMetrics &aa = metrics.antiAliasing;
    aa.historyValid = readBool(aaObject, "historyValid", aa.historyValid);
    aa.temporalDataValid =
        readBool(aaObject, "temporalDataValid", aa.temporalDataValid);
    aa.historyResetReason = readTemporalHistoryResetReason(readString(
        aaObject, "historyResetReason",
        std::string(temporalHistoryResetReasonName(aa.historyResetReason))));
    aa.historyResetCount =
        readU32(aaObject, "historyResetCount", aa.historyResetCount);
    aa.framesSinceHistoryReset = readU32(aaObject, "framesSinceHistoryReset",
                                         aa.framesSinceHistoryReset);
    aa.cameraPositionDelta =
        readF32(aaObject, "cameraPositionDelta", aa.cameraPositionDelta);
    aa.jitterDeltaMagnitude =
        readF32(aaObject, "jitterDeltaMagnitude", aa.jitterDeltaMagnitude);
    aa.motionVectorFormat =
        readFormatName(readString(aaObject, "motionVectorFormat",
                                  snapshotFormatName(aa.motionVectorFormat)));
    aa.motionVectorWidth =
        readU32(aaObject, "motionVectorWidth", aa.motionVectorWidth);
    aa.motionVectorHeight =
        readU32(aaObject, "motionVectorHeight", aa.motionVectorHeight);
    aa.motionVectorTextureCount = readU32(aaObject, "motionVectorTextureCount",
                                          aa.motionVectorTextureCount);
    aa.motionVectorClearPassCount = readU32(
        aaObject, "motionVectorClearPassCount", aa.motionVectorClearPassCount);
    aa.motionVectorDepthReprojectionPassCount =
        readU32(aaObject, "motionVectorDepthReprojectionPassCount",
                aa.motionVectorDepthReprojectionPassCount);
    aa.motionVectorGraphPublished = readBool(
        aaObject, "motionVectorGraphPublished", aa.motionVectorGraphPublished);
    aa.motionVectorDepthReprojectionGenerated =
        readBool(aaObject, "motionVectorDepthReprojectionGenerated",
                 aa.motionVectorDepthReprojectionGenerated);
    aa.velocityPassCount =
        readU32(aaObject, "velocityPassCount", aa.velocityPassCount);
    aa.velocityDrawCount =
        readU32(aaObject, "velocityDrawCount", aa.velocityDrawCount);
    aa.velocityInstanceCount =
        readU32(aaObject, "velocityInstanceCount", aa.velocityInstanceCount);
    aa.velocityPreviousTransformValidCount =
        readU32(aaObject, "velocityPreviousTransformValidCount",
                aa.velocityPreviousTransformValidCount);
    aa.velocityMissingPreviousTransformCount =
        readU32(aaObject, "velocityMissingPreviousTransformCount",
                aa.velocityMissingPreviousTransformCount);
    aa.velocityTessellatedSkippedDrawCount =
        readU32(aaObject, "velocityTessellatedSkippedDrawCount",
                aa.velocityTessellatedSkippedDrawCount);
    aa.previousTransformCacheValid =
        readBool(aaObject, "previousTransformCacheValid",
                 aa.previousTransformCacheValid);
    aa.opaqueVelocityGenerated = readBool(aaObject, "opaqueVelocityGenerated",
                                          aa.opaqueVelocityGenerated);
    aa.velocityAverageObjectMotion =
        readF32(aaObject, "velocityAverageObjectMotion",
                aa.velocityAverageObjectMotion);
    aa.velocityMaxObjectMotion = readF32(aaObject, "velocityMaxObjectMotion",
                                         aa.velocityMaxObjectMotion);
    aa.velocityCameraMatrixDelta = readF32(
        aaObject, "velocityCameraMatrixDelta", aa.velocityCameraMatrixDelta);
    aa.velocityStaticResidualEstimate =
        readF32(aaObject, "velocityStaticResidualEstimate",
                aa.velocityStaticResidualEstimate);
    aa.velocityEstimatedAverageMagnitude =
        readF32(aaObject, "velocityEstimatedAverageMagnitude",
                aa.velocityEstimatedAverageMagnitude);
    aa.velocityEstimatedMaxMagnitude =
        readF32(aaObject, "velocityEstimatedMaxMagnitude",
                aa.velocityEstimatedMaxMagnitude);
    aa.reactiveMaskPassCount =
        readU32(aaObject, "reactiveMaskPassCount", aa.reactiveMaskPassCount);
    aa.reactiveMaskDrawCount =
        readU32(aaObject, "reactiveMaskDrawCount", aa.reactiveMaskDrawCount);
    aa.transparentTransmissionFeedbackRefreshCount =
        readU32(aaObject, "transparentTransmissionFeedbackRefreshCount",
                aa.transparentTransmissionFeedbackRefreshCount);
    aa.transparentTransmissionBlendDrawCount =
        readU32(aaObject, "transparentTransmissionBlendDrawCount",
                aa.transparentTransmissionBlendDrawCount);
    aa.transparentTransmissionFeedbackSourceAvailable =
        readU32(aaObject, "transparentTransmissionFeedbackSourceAvailable",
                aa.transparentTransmissionFeedbackSourceAvailable);
    aa.taaTransparentPostTaaDrawCount =
        readU32(aaObject, "taaTransparentPostTaaDrawCount",
                aa.taaTransparentPostTaaDrawCount);
    aa.taaTransparentPostSpatialAAPassCount =
        readU32(aaObject, "taaTransparentPostSpatialAAPassCount",
                aa.taaTransparentPostSpatialAAPassCount);
    aa.taaResolvePassCount =
        readU32(aaObject, "taaResolvePassCount", aa.taaResolvePassCount);
  }

  yyjson_val *aoObject = yyjson_obj_get(object, "ambientOcclusion");
  if (yyjson_is_obj(aoObject)) {
    AmbientOcclusionFrameMetrics &ao = metrics.ambientOcclusion;
    ao.enabled = readBool(aoObject, "enabled", ao.enabled);
    ao.active = readBool(aoObject, "active", ao.active);
    ao.mainPassCount = readU32(aoObject, "mainPassCount", ao.mainPassCount);
    ao.temporalPassCount =
        readU32(aoObject, "temporalPassCount", ao.temporalPassCount);
    ao.temporalAccumulationActive = readBool(
        aoObject, "temporalAccumulationActive", ao.temporalAccumulationActive);
    ao.temporalMotionVectorsConsumed =
        readBool(aoObject, "temporalMotionVectorsConsumed",
                 ao.temporalMotionVectorsConsumed);
  }

  yyjson_val *hdrObject = yyjson_obj_get(object, "hdrPostProcess");
  if (yyjson_is_obj(hdrObject)) {
    HDRPostProcessFrameMetrics &hdr = metrics.hdrPostProcess;
    hdr.bloomEnabled = readBool(hdrObject, "bloomEnabled", hdr.bloomEnabled);
    hdr.bloomActive = readBool(hdrObject, "bloomActive", hdr.bloomActive);
    hdr.bloomPassCount =
        readU32(hdrObject, "bloomPassCount", hdr.bloomPassCount);
    hdr.adaptationEnabled =
        readBool(hdrObject, "adaptationEnabled", hdr.adaptationEnabled);
    hdr.adaptationActive =
        readBool(hdrObject, "adaptationActive", hdr.adaptationActive);
  }

  yyjson_val *transparentObject = yyjson_obj_get(object, "transparent");
  if (yyjson_is_obj(transparentObject)) {
    RenderFrameMetrics::TransparentFrameMetrics &transparent =
        metrics.transparent;
    transparent.meshDraws =
        readU32(transparentObject, "meshDraws", transparent.meshDraws);
    transparent.contributorSortableDraws =
        readU32(transparentObject, "contributorSortableDraws",
                transparent.contributorSortableDraws);
    transparent.contributorFixedDraws =
        readU32(transparentObject, "contributorFixedDraws",
                transparent.contributorFixedDraws);
    transparent.pickDraws =
        readU32(transparentObject, "pickDraws", transparent.pickDraws);
  }
}

} // namespace

std::string snapshotFormatName(Format format) {
  switch (format) {
  case Format::R32_UINT:
    return "R32_UINT";
  case Format::RGBA8_UNORM:
    return "RGBA8_UNORM";
  case Format::RGBA8_SRGB:
    return "RGBA8_SRGB";
  case Format::RGBA8_UINT:
    return "RGBA8_UINT";
  case Format::RGBA16_FLOAT:
    return "RGBA16_FLOAT";
  case Format::RGBA32_FLOAT:
    return "RGBA32_FLOAT";
  case Format::D32_FLOAT:
    return "D32_FLOAT";
  case Format::R32_FLOAT:
    return "R32_FLOAT";
  case Format::RG32_FLOAT:
    return "RG32_FLOAT";
  case Format::D16_UNORM:
    return "D16_UNORM";
  case Format::RG16_FLOAT:
    return "RG16_FLOAT";
  case Format::R8_UNORM:
    return "R8_UNORM";
  case Format::R16_UNORM:
    return "R16_UNORM";
  case Format::BC7_RGBA_UNORM:
    return "BC7_RGBA_UNORM";
  case Format::BC7_RGBA_SRGB:
    return "BC7_RGBA_SRGB";
  case Format::ETC2_RGB8_UNORM:
    return "ETC2_RGB8_UNORM";
  case Format::ETC2_RGB8_SRGB:
    return "ETC2_RGB8_SRGB";
  case Format::Count:
    break;
  }
  return "Count";
}

Result<std::string, std::string>
writeSnapshotReportJson(const SnapshotReport &report) {
  JsonMutDocPtr doc(yyjson_mut_doc_new(nullptr), &yyjson_mut_doc_free);
  if (!doc) {
    return Result<std::string, std::string>::makeError(
        "writeSnapshotReportJson: failed to allocate JSON document");
  }
  yyjson_mut_val *root = yyjson_mut_obj(doc.get());
  yyjson_mut_doc_set_root(doc.get(), root);
  yyjson_mut_obj_add_uint(doc.get(), root, "schemaVersion",
                          report.schemaVersion);
  addString(doc.get(), root, "kind", report.kind);
  addString(doc.get(), root, "generatedAtUtc", report.generatedAtUtc);
  addString(doc.get(), root, "command", report.command);
  yyjson_mut_obj_add_val(doc.get(), root, "environment",
                         makeEnvironmentObject(doc.get(), report.environment));
  yyjson_mut_obj_add_val(doc.get(), root, "case",
                         makeCaseObject(doc.get(), report.snapshotCase));

  yyjson_mut_val *artifacts = yyjson_mut_obj(doc.get());
  addPath(doc.get(), artifacts, "artifactDir", report.artifacts.artifactDir);
  addPath(doc.get(), artifacts, "rootHtml", report.artifacts.rootHtml);
  addPath(doc.get(), artifacts, "caseDir", report.artifacts.caseDir);
  addPath(doc.get(), artifacts, "caseHtml", report.artifacts.caseHtml);
  yyjson_mut_obj_add_val(doc.get(), root, "artifacts", artifacts);

  yyjson_mut_val *captures = yyjson_mut_arr(doc.get());
  for (const SnapshotCaptureReport &capture : report.captures) {
    yyjson_mut_arr_add_val(captures, makeCaptureObject(doc.get(), capture));
  }
  yyjson_mut_obj_add_val(doc.get(), root, "captures", captures);
  yyjson_mut_obj_add_val(
      doc.get(), root, "availableCapturePoints",
      makeStringArray(doc.get(), report.availableCapturePoints));
  yyjson_mut_obj_add_strcpy(doc.get(), root, "captureSynchronization",
                            report.captureSynchronization.c_str());
  yyjson_mut_obj_add_val(
      doc.get(), root, "rendererMetrics",
      makeRendererMetricsObject(doc.get(), report.rendererMetrics));
  yyjson_mut_obj_add_val(doc.get(), root, "renderGraph",
                         yyjson_mut_obj(doc.get()));
  addString(doc.get(), root, "reproduceCommand", report.reproduceCommand);
  yyjson_mut_obj_add_val(doc.get(), root, "warnings",
                         makeStringArray(doc.get(), report.warnings));
  yyjson_mut_obj_add_val(doc.get(), root, "errors",
                         makeStringArray(doc.get(), report.errors));

  size_t length = 0u;
  char *json = yyjson_mut_write_opts(doc.get(), YYJSON_WRITE_PRETTY, nullptr,
                                     &length, nullptr);
  if (json == nullptr) {
    return Result<std::string, std::string>::makeError(
        "writeSnapshotReportJson: failed to serialize JSON");
  }
  std::string out(json, length);
  std::free(json);
  return Result<std::string, std::string>::makeResult(std::move(out));
}

Result<bool, std::string>
writeSnapshotReportFile(const SnapshotReport &report,
                        const std::filesystem::path &path) {
  auto json = writeSnapshotReportJson(report);
  if (json.hasError()) {
    return Result<bool, std::string>::makeError(json.error());
  }
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    return Result<bool, std::string>::makeError(
        "writeSnapshotReportFile: failed to open " + path.string());
  }
  file << json.value();
  return Result<bool, std::string>::makeResult(true);
}

Result<SnapshotReport, std::string>
readSnapshotReportFile(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return Result<SnapshotReport, std::string>::makeError(
        "readSnapshotReportFile: failed to open " + path.string());
  }
  std::string json((std::istreambuf_iterator<char>(file)),
                   std::istreambuf_iterator<char>());
  yyjson_read_err error{};
  JsonDocPtr doc(yyjson_read_opts(json.data(), json.size(), 0, nullptr, &error),
                 &yyjson_doc_free);
  if (!doc) {
    return Result<SnapshotReport, std::string>::makeError(
        "readSnapshotReportFile: JSON parse failed at byte " +
        std::to_string(error.pos) + ": " + error.msg);
  }
  yyjson_val *root = yyjson_doc_get_root(doc.get());
  if (!yyjson_is_obj(root)) {
    return Result<SnapshotReport, std::string>::makeError(
        "readSnapshotReportFile: root must be an object");
  }
  SnapshotReport report{};
  report.kind = readString(root, "kind", report.kind);
  report.generatedAtUtc = readString(root, "generatedAtUtc");
  report.command = readString(root, "command");
  yyjson_val *environment = yyjson_obj_get(root, "environment");
  if (yyjson_is_obj(environment)) {
    report.environment.gpuBackend = readString(environment, "gpuBackend");
    report.environment.gpuBackendSource =
        readString(environment, "gpuBackendSource");
    report.environment.gpuDeviceName = readString(environment, "gpuDeviceName");
    report.environment.resolvedPresentMode =
        readString(environment, "resolvedPresentMode");
    report.environment.resolvedWindowMode =
        readString(environment, "resolvedWindowMode");
  }
  yyjson_val *caseObject = yyjson_obj_get(root, "case");
  if (yyjson_is_obj(caseObject)) {
    report.snapshotCase.id = readString(caseObject, "id");
    report.snapshotCase.suite = readString(caseObject, "suite");
  }
  yyjson_val *artifacts = yyjson_obj_get(root, "artifacts");
  if (yyjson_is_obj(artifacts)) {
    report.artifacts.artifactDir = readString(artifacts, "artifactDir");
    report.artifacts.rootHtml = readString(artifacts, "rootHtml");
    report.artifacts.caseDir = readString(artifacts, "caseDir");
    report.artifacts.caseHtml = readString(artifacts, "caseHtml");
  }
  yyjson_val *captures = yyjson_obj_get(root, "captures");
  if (yyjson_is_arr(captures)) {
    yyjson_arr_iter iter;
    yyjson_arr_iter_init(captures, &iter);
    yyjson_val *entry = nullptr;
    while ((entry = yyjson_arr_iter_next(&iter)) != nullptr) {
      if (!yyjson_is_obj(entry)) {
        continue;
      }
      SnapshotCaptureReport capture{};
      capture.target = readString(entry, "target");
      capture.artifactStem = readString(entry, "artifactStem");
      capture.profile = readString(entry, "profile");
      capture.required = readBool(entry, "required", capture.required);
      capture.available = readBool(entry, "available", capture.available);
      capture.capturePointVersion =
          readU32(entry, "capturePointVersion", capture.capturePointVersion);
      capture.captureFrameIndex =
          readU64(entry, "captureFrameIndex", capture.captureFrameIndex);
      capture.lifetime = readString(entry, "lifetime");
      capture.format = readString(entry, "format");
      capture.colorSpace = readString(entry, "colorSpace");
      capture.origin = readString(entry, "origin", capture.origin);
      capture.mip = readU32(entry, "mip", capture.mip);
      capture.layer = readU32(entry, "layer", capture.layer);
      capture.status = readString(entry, "status");
      capture.statusReason = readString(entry, "statusReason");
      capture.actual = readString(entry, "actual");
      capture.actualMetadata = readString(entry, "actualMetadata");
      capture.preview = readString(entry, "preview");
      capture.expected = readString(entry, "expected");
      capture.diff = readString(entry, "diff");
      capture.actualHash = readString(entry, "actualHash");
      capture.expectedHash = readString(entry, "expectedHash");
      capture.width = readU32(entry, "width");
      capture.height = readU32(entry, "height");
      capture.producerPassLabel = readString(entry, "producerPassLabel");
      capture.readbackError = readString(entry, "readbackError");
      report.captures.push_back(std::move(capture));
    }
  }
  readRendererMetrics(yyjson_obj_get(root, "rendererMetrics"),
                      report.rendererMetrics);
  return Result<SnapshotReport, std::string>::makeResult(std::move(report));
}

} // namespace nuri::tools::snapshot
