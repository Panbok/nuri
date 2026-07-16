#include "nuri/tools/core/baseline_profile.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#else
#include <unistd.h>
#endif

namespace nuri::tools::core {
namespace {

class ProfileDirectory {
public:
  ProfileDirectory() {
    static std::atomic_uint64_t sequence{0u};
    const auto tick =
        std::chrono::steady_clock::now().time_since_epoch().count();
#if defined(_WIN32)
    const auto process = static_cast<uint64_t>(GetCurrentProcessId());
#else
    const auto process = static_cast<uint64_t>(getpid());
#endif
    path_ =
        std::filesystem::temp_directory_path() /
        ("nuri-baseline-profile-tests-" + std::to_string(process) + "-" +
         std::to_string(tick) + "-" + std::to_string(sequence.fetch_add(1u)));
    std::filesystem::create_directories(path_);
  }

  ~ProfileDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  ProfileDirectory(const ProfileDirectory &) = delete;
  ProfileDirectory &operator=(const ProfileDirectory &) = delete;

  void write(std::string_view id, std::string_view json) const {
    std::ofstream file(path_ / (std::string(id) + ".json"), std::ios::binary);
    file << json;
  }

  [[nodiscard]] const std::filesystem::path &path() const noexcept {
    return path_;
  }

private:
  std::filesystem::path path_{};
};

[[nodiscard]] std::string validProfile(std::string_view id) {
  return R"json({
    "schemaVersion": 1,
    "kind": "nuri.baseline.profile",
    "id": ")json" +
         std::string(id) + R"json(",
    "description": "test profile",
    "authority": {
      "authoritative": false,
      "allowDirtyTree": true,
      "reason": "not pinned"
    },
    "environment": {
      "os": "windows",
      "backend": "nvrhi-vulkan",
      "windowMode": "visible",
      "gpuVendorId": 4318,
      "gpuDeviceId": 9860,
      "driver": "test-driver"
    },
    "execution": {
      "presentMode": "immediate",
      "profiling": "off",
      "devChecks": false
    },
    "benchmarkPolicy": {
      "minimumRepetitions": 3,
      "warmupStability": "investigative",
      "warmupWindowFrames": 2,
      "warmupMaxDriftPercent": 15.0,
      "requiredMetrics": ["cpu.render_submit_ms", "gpu.scopes_sum_ms"],
      "thresholdOwnership": "baseline"
    }
  })json";
}

[[nodiscard]] std::string authoritativeProfile(std::string_view id) {
  std::string json = validProfile(id);
  json.replace(json.find("\"authoritative\": false"), 22u,
               "\"authoritative\": true");
  json.replace(json.find("\"allowDirtyTree\": true"), 22u,
               "\"allowDirtyTree\": false");
  json.replace(json.find("\"warmupStability\": \"investigative\""), 34u,
               "\"warmupStability\": \"required\"");
  return json;
}

TEST(BaselineProfile, LoadsStrictV1Contract) {
  ProfileDirectory profiles;
  profiles.write("test-profile", validProfile("test-profile"));

  auto loaded = loadBaselineProfile(profiles.path(), "test-profile");

  ASSERT_TRUE(loaded.hasValue())
      << (loaded.hasError() ? loaded.error() : std::string{});
  EXPECT_EQ(loaded.value().schemaVersion, 1u);
  EXPECT_EQ(loaded.value().kind, "nuri.baseline.profile");
  EXPECT_EQ(loaded.value().id, "test-profile");
  EXPECT_FALSE(loaded.value().authority.authoritative);
  EXPECT_TRUE(loaded.value().authority.allowDirtyTree);
  EXPECT_EQ(loaded.value().environment.windowMode, "visible");
  ASSERT_TRUE(loaded.value().environment.gpuVendorId.has_value());
  EXPECT_EQ(*loaded.value().environment.gpuVendorId, 4318u);
  ASSERT_TRUE(loaded.value().environment.driver.has_value());
  EXPECT_EQ(*loaded.value().environment.driver, "test-driver");
  EXPECT_EQ(loaded.value().execution.profiling, "off");
  EXPECT_EQ(loaded.value().benchmarkPolicy.minimumRepetitions, 3u);
  EXPECT_EQ(loaded.value().benchmarkPolicy.warmupWindowFrames, 2u);
  EXPECT_DOUBLE_EQ(loaded.value().benchmarkPolicy.warmupMaxDriftPercent, 15.0);
  EXPECT_EQ(loaded.value().benchmarkPolicy.requiredMetrics.size(), 2u);
  EXPECT_EQ(loaded.value().sourcePath.filename(), "test-profile.json");
}

TEST(BaselineProfile, RejectsRemovedGpuProfilingMode) {
  ProfileDirectory profiles;
  std::string json = validProfile("removed-gpu-profiling");
  json.replace(json.find("\"profiling\": \"off\""), 18u,
               "\"profiling\": \"unsupported\"");
  profiles.write("removed-gpu-profiling", json);

  auto loaded = loadBaselineProfile(profiles.path(), "removed-gpu-profiling");
  ASSERT_TRUE(loaded.hasError());
  EXPECT_NE(loaded.error().find("unsupported value"), std::string::npos);
}

TEST(BaselineProfile, AcceptsNullableHardwareSelectors) {
  ProfileDirectory profiles;
  std::string json = validProfile("nullable");
  json.replace(json.find("4318"), 4u, "null");
  json.replace(json.find("9860"), 4u, "null");
  json.replace(json.find("\"test-driver\""), 13u, "null");
  profiles.write("nullable", json);

  auto loaded = loadBaselineProfile(profiles.path(), "nullable");

  ASSERT_TRUE(loaded.hasValue())
      << (loaded.hasError() ? loaded.error() : std::string{});
  EXPECT_FALSE(loaded.value().environment.gpuVendorId.has_value());
  EXPECT_FALSE(loaded.value().environment.gpuDeviceId.has_value());
  EXPECT_FALSE(loaded.value().environment.driver.has_value());
}

TEST(BaselineProfile, RejectsUnsafeAndMissingProfileIds) {
  ProfileDirectory profiles;

  auto unsafe = loadBaselineProfile(profiles.path(), "../escape");
  auto missing = loadBaselineProfile(profiles.path(), "missing");

  ASSERT_TRUE(unsafe.hasError());
  EXPECT_NE(unsafe.error().find("invalid"), std::string::npos);
  ASSERT_TRUE(missing.hasError());
  EXPECT_NE(missing.error().find("does not exist"), std::string::npos);
}

TEST(BaselineProfile, RejectsMismatchedIdentity) {
  ProfileDirectory profiles;
  profiles.write("requested", validProfile("different"));

  auto loaded = loadBaselineProfile(profiles.path(), "requested");

  ASSERT_TRUE(loaded.hasError());
  EXPECT_NE(loaded.error().find("does not match"), std::string::npos);
}

TEST(BaselineProfile, RejectsUnknownFieldsAndUnsupportedSchema) {
  ProfileDirectory profiles;
  std::string unknown = validProfile("unknown-field");
  unknown.replace(unknown.rfind("}"), 1u, ", \"unexpected\": true }");
  profiles.write("unknown-field", unknown);

  std::string unsupported = validProfile("unsupported");
  unsupported.replace(unsupported.find("\"schemaVersion\": 1"), 18u,
                      "\"schemaVersion\": 2");
  profiles.write("unsupported", unsupported);

  auto unknownResult = loadBaselineProfile(profiles.path(), "unknown-field");
  auto unsupportedResult = loadBaselineProfile(profiles.path(), "unsupported");

  ASSERT_TRUE(unknownResult.hasError());
  EXPECT_NE(unknownResult.error().find("unknown field"), std::string::npos);
  ASSERT_TRUE(unsupportedResult.hasError());
  EXPECT_NE(unsupportedResult.error().find("must equal 1"), std::string::npos);
}

TEST(BaselineProfile, RejectsDuplicateOrUnsafeRequiredMetrics) {
  ProfileDirectory profiles;
  std::string duplicate = validProfile("duplicate");
  duplicate.replace(duplicate.find("\"gpu.scopes_sum_ms\""), 19u,
                    "\"cpu.render_submit_ms\"");
  profiles.write("duplicate", duplicate);

  std::string unsafe = validProfile("unsafe-metric");
  unsafe.replace(unsafe.find("\"gpu.scopes_sum_ms\""), 19u, "\"gpu/escape\"");
  profiles.write("unsafe-metric", unsafe);

  auto duplicateResult = loadBaselineProfile(profiles.path(), "duplicate");
  auto unsafeResult = loadBaselineProfile(profiles.path(), "unsafe-metric");

  ASSERT_TRUE(duplicateResult.hasError());
  EXPECT_NE(duplicateResult.error().find("unique"), std::string::npos);
  ASSERT_TRUE(unsafeResult.hasError());
  EXPECT_NE(unsafeResult.error().find("invalid character"), std::string::npos);
}

TEST(BaselineProfile, AuthoritativeProfilesRequirePinnedStablePolicy) {
  ProfileDirectory profiles;
  profiles.write("authoritative", authoritativeProfile("authoritative"));
  auto loaded = loadBaselineProfile(profiles.path(), "authoritative");
  ASSERT_TRUE(loaded.hasValue())
      << (loaded.hasError() ? loaded.error() : std::string{});

  std::string unpinned = authoritativeProfile("unpinned");
  unpinned.replace(unpinned.find("4318"), 4u, "null");
  profiles.write("unpinned", unpinned);
  auto rejected = loadBaselineProfile(profiles.path(), "unpinned");
  ASSERT_TRUE(rejected.hasError());
  EXPECT_NE(rejected.error().find("pinned"), std::string::npos);
}

TEST(BaselineProfile, CompatibilityExplainsEveryProfileMismatch) {
  ProfileDirectory profiles;
  profiles.write("authoritative", authoritativeProfile("authoritative"));
  auto loaded = loadBaselineProfile(profiles.path(), "authoritative");
  ASSERT_TRUE(loaded.hasValue())
      << (loaded.hasError() ? loaded.error() : std::string{});
  BaselineProfileObservedEnvironment observed{
      .os = "Windows",
      .backend = "nvrhi-vulkan",
      .backendSource = "manifest",
      .windowMode = "visible",
      .windowVisible = true,
      .gpuVendorId = 4318u,
      .gpuDeviceId = 9860u,
      .driver = "test-driver",
      .presentMode = "immediate",
      .profiling = "off",
      .devChecks = false,
      .dirtyTree = false,
  };
  auto compatible = evaluateBaselineProfile(loaded.value(), observed);
  EXPECT_TRUE(compatible.compatible);
  EXPECT_TRUE(compatible.reasons.empty());

  observed.driver = "other-driver";
  observed.dirtyTree = true;
  compatible = evaluateBaselineProfile(loaded.value(), observed);
  EXPECT_FALSE(compatible.compatible);
  EXPECT_EQ(compatible.reasons.size(), 2u);
}

} // namespace
} // namespace nuri::tools::core
