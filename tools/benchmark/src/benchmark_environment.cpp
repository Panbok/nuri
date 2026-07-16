#include "nuri/tools/benchmark/benchmark_environment.h"

#include "nuri/tools/benchmark/build_config.h"
#include "nuri/tools/core/environment_probe.h"

namespace nuri::tools::benchmark {
namespace {

void applyCommonEnvironment(
    BenchmarkEnvironment &env,
    const nuri::tools::core::EnvironmentProbeResult &common) {
  env.repoRoot = common.host.repoRoot;
  env.commitHash = common.host.commitHash;
  env.branchName = common.host.branchName;
  env.dirty = common.host.dirty;
  env.osName = common.host.osName;
  env.osVersion = common.host.osVersion;
  env.cpuName = common.host.cpuName;
  env.cpuLogicalThreadCount = common.host.cpuLogicalThreadCount;
  env.buildType = common.build.buildType;
  env.cmakeToolProfile = common.build.cmakeToolProfile;
  env.vcpkgManifestFeatures = common.build.vcpkgManifestFeatures;
  env.buildShared = common.build.buildShared;
  env.loggingEnabled = common.build.loggingEnabled;
  env.assertsEnabled = common.build.assertsEnabled;
  env.tracyEnabled = common.build.tracyEnabled;
  env.devChecks = common.build.devChecks;
}

} // namespace

std::filesystem::path benchmarkRepoRoot() {
#if defined(PROJECT_SOURCE_DIR)
  return std::filesystem::path(PROJECT_SOURCE_DIR).lexically_normal();
#else
  return std::filesystem::current_path();
#endif
}

std::string readProcessEnvironment(std::string_view name) {
  return nuri::tools::core::readEnvironmentVariable(name);
}

BenchmarkEnvironment collectBenchmarkEnvironment(
    std::string_view backend, std::string_view backendSource,
    std::string_view requestedPresentMode, std::string_view presentModeSource,
    bool tracyDiagnostic) {
  BenchmarkEnvironment env{};
  const auto common = nuri::tools::core::collectEnvironmentProbe(
      benchmarkRepoRoot(),
      nuri::tools::core::EnvironmentBuildFacts{
          .buildType = NURI_BENCHMARK_BUILD_TYPE,
          .cmakeToolProfile = NURI_BENCHMARK_TOOL_PROFILE,
          .vcpkgManifestFeatures = NURI_BENCHMARK_VCPKG_MANIFEST_FEATURES,
          .buildShared = NURI_BENCHMARK_BUILD_SHARED != 0,
          .loggingEnabled = NURI_BENCHMARK_WITH_LOGGING != 0,
          .assertsEnabled = NURI_BENCHMARK_WITH_ASSERTS != 0,
          .tracyEnabled = NURI_BENCHMARK_WITH_TRACY != 0,
          .devChecks = NURI_BENCHMARK_DEV_CHECKS != 0,
      });
  applyCommonEnvironment(env, common);

  // Runtime-resolved GPU, present, window, and render-graph facts remain owned
  // by the benchmark adapter and are updated from the live session later.
  env.gpuBackend = std::string(backend);
  env.gpuBackendSource = std::string(backendSource);
  env.requestedPresentMode = std::string(requestedPresentMode);
  env.resolvedPresentMode = std::string(requestedPresentMode);
  env.presentModeSource = std::string(presentModeSource);
  env.tracyDiagnostic = tracyDiagnostic;
  return env;
}

std::string joinCommandLine(int argc, char **argv) {
  return nuri::tools::core::joinCommandLine(argc, argv);
}

std::string utcTimestampIso8601() {
  return nuri::tools::core::utcTimestampIso8601();
}

std::string utcTimestampForPath() {
  return nuri::tools::core::utcTimestampForPath();
}

} // namespace nuri::tools::benchmark
