#pragma once

#include "nuri/core/result.h"
#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/tools/snapshot/snapshot_case.h"
#include "nuri/tools/snapshot/snapshot_report.h"

#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace nuri::tools::snapshot {

struct SnapshotCaptureArtifactResult {
  std::vector<SnapshotCaptureReport> captures{};
  std::vector<std::string> availableCapturePoints{};
  bool missingRequiredCapture = false;
  bool unsupportedRequiredCapture = false;
  bool readbackFailedRequiredCapture = false;
};

[[nodiscard]] Result<SnapshotCaptureArtifactResult, std::string>
writeSnapshotCaptureArtifacts(GPUDevice &gpu,
                              const RenderFrameContext &frameContext,
                              std::span<const SnapshotCaptureTarget> targets,
                              const std::filesystem::path &caseDir,
                              const std::filesystem::path &artifactStemDir);

} // namespace nuri::tools::snapshot
