#include "nuri/tools/benchmark/benchmark_case.h"

namespace nuri::tools::benchmark {

std::string benchmarkExitCodeName(BenchmarkExitCode code) {
  switch (code) {
  case BenchmarkExitCode::Success:
    return "success";
  case BenchmarkExitCode::Regression:
    return "benchmark regression";
  case BenchmarkExitCode::InvalidInput:
    return "invalid input";
  case BenchmarkExitCode::EnvironmentUnavailable:
    return "environment unavailable";
  case BenchmarkExitCode::RuntimeError:
    return "runtime error";
  case BenchmarkExitCode::MissingBaseline:
    return "baseline/report missing";
  }
  return "unknown";
}

void sanitizeBenchmarkRenderSettings(RenderSettings &settings) {
  sanitizeTextureFilteringSettings(settings.textureFiltering);
  sanitizeToneMapSettings(settings.toneMap);
  sanitizeHDRPostProcessSettings(settings.hdrPostProcess);
  sanitizeTransmissionSettings(settings.transmission);
  sanitizeAntiAliasingSettings(settings.antiAliasing);
  sanitizeAmbientOcclusionSettings(settings.ambientOcclusion, settings.opaque,
                                   settings.antiAliasing);
  sanitizeShadowSettings(settings.shadow);
  sanitizeVisibilitySettings(settings.visibility);
  sanitizeDDGISettings(settings.ddgi);
}

} // namespace nuri::tools::benchmark
