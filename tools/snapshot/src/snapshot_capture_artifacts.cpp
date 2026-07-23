#include "nuri/tools/snapshot/snapshot_capture_artifacts.h"

#include "nuri/tools/snapshot/snapshot_capture_point.h"
#include "nuri/tools/snapshot/snapshot_image.h"

#include <algorithm>
#include <system_error>

namespace nuri::tools::snapshot {
namespace {

[[nodiscard]] std::filesystem::path
relativeToCaseDir(const std::filesystem::path &caseDir,
                  const std::filesystem::path &path) {
  std::error_code ec;
  std::filesystem::path relative = std::filesystem::relative(path, caseDir, ec);
  return ec ? path : relative;
}

[[nodiscard]] SnapshotCaptureReport
makeInitialCaptureReport(const SnapshotCaptureTarget &target) {
  SnapshotCaptureReport capture{};
  capture.target = target.name;
  capture.artifactStem = target.name;
  capture.profile = target.profile;
  capture.required = target.required;
  capture.status = "missing_capture_point";
  capture.statusReason = "not_rendered";
  return capture;
}

} // namespace

Result<SnapshotCaptureArtifactResult, std::string>
writeSnapshotCaptureArtifacts(GPUDevice &gpu,
                              const RenderFrameContext &frameContext,
                              std::span<const SnapshotCaptureTarget> targets,
                              const std::filesystem::path &caseDir,
                              const std::filesystem::path &artifactStemDir) {
  SnapshotCaptureArtifactResult result{};
  result.captures.reserve(targets.size());
  for (const SnapshotCaptureTarget &target : targets) {
    result.captures.push_back(makeInitialCaptureReport(target));
  }
  for (const RenderCapturePoint &point :
       frameContext.captureRegistry.points()) {
    result.availableCapturePoints.emplace_back(point.name);
  }

  for (SnapshotCaptureReport &capture : result.captures) {
    const RenderCapturePoint *point =
        frameContext.captureRegistry.find(capture.target);
    if (point == nullptr) {
      capture.status = "missing_capture_point";
      capture.statusReason = "capture_point_not_published";
      result.missingRequiredCapture =
          result.missingRequiredCapture || capture.required;
      continue;
    }
    capture.available = true;
    capture.capturePointVersion = point->version;
    capture.captureFrameIndex = point->frameIndex;
    capture.kind = std::string(renderCaptureValueKindName(point->kind));
    capture.lifetime = std::string(renderCaptureLifetimeName(point->lifetime));
    capture.format = snapshotFormatName(point->format);
    capture.colorSpace = std::string(point->colorSpace);
    capture.width = point->mip >= 32u
                        ? 1u
                        : std::max(point->dimensions.width >> point->mip, 1u);
    capture.height = point->mip >= 32u
                         ? 1u
                         : std::max(point->dimensions.height >> point->mip, 1u);
    capture.mip = point->mip;
    capture.layer = point->layer;
    capture.producerPassLabel = std::string(point->producerPassLabel);
    capture.ddgiMetadata = point->ddgiMetadata;
    const SnapshotCaptureCatalogEntry *catalog =
        findSnapshotCaptureCatalogEntry(capture.target);
    if (catalog == nullptr || point->version != catalog->version ||
        point->kind != catalog->kind ||
        !snapshotCompareProfileSupportsKind(capture.profile, point->kind)) {
      capture.status = "invalid_descriptor";
      capture.statusReason = "published_capture_descriptor_mismatch";
      result.incompatibleRequiredCapture =
          result.incompatibleRequiredCapture || capture.required;
      continue;
    }
    if (snapshotFormatBytesPerPixel(point->format) == 0u) {
      capture.status = "unsupported_format";
      capture.statusReason = "unsupported_readback_format";
      result.unsupportedRequiredCapture =
          result.unsupportedRequiredCapture || capture.required;
      continue;
    }
    auto readback = readSnapshotCapture(gpu, *point);
    if (readback.hasError()) {
      capture.status = "readback_error";
      capture.statusReason = "gpu_readback_failed";
      capture.readbackError = readback.error();
      result.readbackFailedRequiredCapture =
          result.readbackFailedRequiredCapture || capture.required;
      continue;
    }
    SnapshotArtifactPaths paths{};
    auto written = writeSnapshotArtifacts(readback.value(),
                                          artifactStemDir / capture.target,
                                          paths, capture.profile);
    if (written.hasError()) {
      return Result<SnapshotCaptureArtifactResult, std::string>::makeError(
          written.error());
    }
    capture.actualHash = readback.value().hash;
    capture.actual = relativeToCaseDir(caseDir, paths.raw);
    capture.actualMetadata = relativeToCaseDir(caseDir, paths.metadata);
    capture.preview = relativeToCaseDir(caseDir, paths.preview);
    capture.status = "captured";
    capture.statusReason = "capture_written";
  }

  return Result<SnapshotCaptureArtifactResult, std::string>::makeResult(
      std::move(result));
}

} // namespace nuri::tools::snapshot
