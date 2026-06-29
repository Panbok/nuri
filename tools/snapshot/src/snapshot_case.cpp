#include "nuri/tools/snapshot/snapshot_case.h"

namespace nuri::tools::snapshot {

std::string snapshotExitCodeName(SnapshotExitCode code) {
  switch (code) {
  case SnapshotExitCode::Success:
    return "success";
  case SnapshotExitCode::VisualMismatch:
    return "visual_mismatch";
  case SnapshotExitCode::InvalidInput:
    return "invalid_input";
  case SnapshotExitCode::EnvironmentUnavailable:
    return "environment_unavailable";
  case SnapshotExitCode::RuntimeError:
    return "runtime_error";
  case SnapshotExitCode::MissingBaseline:
    return "missing_baseline";
  }
  return "unknown";
}

void sanitizeSnapshotRenderSettings(RenderSettings &settings) {
  settings.debug.enabled = false;
  settings.debug.modelBounds = false;
  settings.debug.grid = false;
  settings.debug.lightIcons = false;
  settings.opaque.debugVisualization = OpaqueDebugVisualization::None;
  settings.antiAliasing.mode =
      sanitizeAntiAliasingMode(settings.antiAliasing.mode);
  settings.antiAliasing.qualityPreset =
      sanitizeTemporalAAQualityPreset(settings.antiAliasing.qualityPreset);
  settings.ambientOcclusion.mode =
      sanitizeAmbientOcclusionMode(settings.ambientOcclusion.mode);
  settings.ambientOcclusion.preset =
      sanitizeAmbientOcclusionPreset(settings.ambientOcclusion.preset);
  settings.textureFiltering.mode =
      sanitizeTextureFilterMode(settings.textureFiltering.mode);
  settings.textureFiltering.anisotropy =
      sanitizeTextureFilterAnisotropy(settings.textureFiltering.anisotropy);
  sanitizeToneMapSettings(settings.toneMap);
  sanitizeHDRPostProcessSettings(settings.hdrPostProcess);
}

} // namespace nuri::tools::snapshot
