#include "tests_pch.h"

#include "nuri/core/event_manager.h"
#include "nuri/core/log.h"
#include "nuri/core/result.h"
#include "nuri/math/utils.h"
#include "nuri/utils/frame_time_display.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory_resource>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

struct ThrowingMove {
  ThrowingMove() = default;
  ThrowingMove(const ThrowingMove &) = default;
  ThrowingMove(ThrowingMove &&) noexcept(false) {}
};

using NothrowResult = nuri::Result<int, int>;
using ThrowingValueResult = nuri::Result<ThrowingMove, int>;
using ThrowingErrorResult = nuri::Result<void, ThrowingMove>;

static_assert(noexcept(
    std::declval<NothrowResult &>().swap(std::declval<NothrowResult &>())));
static_assert(!noexcept(std::declval<ThrowingValueResult &>().swap(
    std::declval<ThrowingValueResult &>())));
static_assert(!noexcept(std::declval<ThrowingErrorResult &>().swap(
    std::declval<ThrowingErrorResult &>())));
static_assert(noexcept(std::declval<NothrowResult &>().value()));
static_assert(noexcept(std::declval<NothrowResult &>().error()));
static_assert(noexcept(std::declval<nuri::Result<void, int> &>().error()));

TEST(LogTests, WritesConfiguredFileWithoutExternalLoggingDependency) {
  const auto tick =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      ("nuri_stdio_log_" + std::to_string(tick) + ".log");

  nuri::Log::shutdown();
  nuri::Log::initialize({
      .filePath = path.string(),
      .logLevel = nuri::LogLevel::Info,
      .consoleLevel = nuri::LogLevel::Fatal,
  });
  nuri::logMessage(nuri::LogLevel::Info, "stdio logger probe");
  nuri::Log::shutdown();

  std::ifstream input(path, std::ios::binary);
  const std::string contents((std::istreambuf_iterator<char>(input)),
                             std::istreambuf_iterator<char>());
  EXPECT_NE(contents.find("info"), std::string::npos);
  EXPECT_NE(contents.find("stdio logger probe"), std::string::npos);
  EXPECT_EQ(contents.find("\x1b["), std::string::npos);

  std::error_code error;
  std::filesystem::remove(path, error);
}

TEST(LogTests, EnablesColoredConsoleByDefault) {
  EXPECT_TRUE(nuri::LogConfig{}.coloredConsole);
}

TEST(FrameTimeDisplaySamplerTests, PublishesIntervalAveragesAtReadableCadence) {
  nuri::FrameTimeDisplaySampler sampler(0.25);

  EXPECT_FALSE(sampler.tick(0.10, nuri::GpuFrameTimeSample{1u, 8.0f}));
  EXPECT_FALSE(sampler.tick(0.10, nuri::GpuFrameTimeSample{2u, 12.0f}));
  EXPECT_TRUE(sampler.tick(0.05, nuri::GpuFrameTimeSample{3u, 10.0f}));

  const nuri::FrameTimeDisplayValues values = sampler.values();
  EXPECT_TRUE(values.cpuAvailable);
  EXPECT_TRUE(values.gpuAvailable);
  EXPECT_NEAR(values.cpuMilliseconds, 250.0f / 3.0f, 1.0e-4f);
  EXPECT_NEAR(values.gpuMilliseconds, 10.0f, 1.0e-4f);
}

TEST(FrameTimeDisplaySamplerTests,
     DeduplicatesGpuReportsAndRetainsTheLastPublishedValue) {
  nuri::FrameTimeDisplaySampler sampler(0.20);

  EXPECT_FALSE(sampler.tick(0.05, nuri::GpuFrameTimeSample{7u, 10.0f}));
  EXPECT_FALSE(sampler.tick(0.05, nuri::GpuFrameTimeSample{7u, 10.0f}));
  EXPECT_TRUE(sampler.tick(0.10, nuri::GpuFrameTimeSample{8u, 20.0f}));
  EXPECT_NEAR(sampler.values().gpuMilliseconds, 15.0f, 1.0e-4f);

  EXPECT_FALSE(sampler.tick(0.10, std::nullopt));
  EXPECT_TRUE(sampler.tick(0.10, std::nullopt));
  EXPECT_TRUE(sampler.values().gpuAvailable);
  EXPECT_NEAR(sampler.values().gpuMilliseconds, 15.0f, 1.0e-4f);
}

TEST(FrameTimeDisplaySamplerTests, IgnoresInvalidGpuSamples) {
  nuri::FrameTimeDisplaySampler sampler(0.10);

  EXPECT_TRUE(sampler.tick(
      0.10,
      nuri::GpuFrameTimeSample{1u, std::numeric_limits<float>::quiet_NaN()}));
  EXPECT_TRUE(sampler.values().cpuAvailable);
  EXPECT_FALSE(sampler.values().gpuAvailable);
}

TEST(AlignUpU64Tests, SupportsZeroOnePowerOfTwoAndArbitraryAlignments) {
  uint64_t aligned = 99u;

  EXPECT_TRUE(nuri::alignUpU64(7u, 0u, aligned));
  EXPECT_EQ(aligned, 7u);
  EXPECT_TRUE(nuri::alignUpU64(7u, 1u, aligned));
  EXPECT_EQ(aligned, 7u);
  EXPECT_TRUE(nuri::alignUpU64(16u, 8u, aligned));
  EXPECT_EQ(aligned, 16u);
  EXPECT_TRUE(nuri::alignUpU64(17u, 8u, aligned));
  EXPECT_EQ(aligned, 24u);
  EXPECT_TRUE(nuri::alignUpU64(7u, 3u, aligned));
  EXPECT_EQ(aligned, 9u);
  EXPECT_TRUE(nuri::alignUpU64(12u, 6u, aligned));
  EXPECT_EQ(aligned, 12u);
}

TEST(AlignUpU64Tests, HandlesMaximumValueBoundariesWithoutChangingOnFailure) {
  constexpr uint64_t maximum = std::numeric_limits<uint64_t>::max();
  uint64_t aligned = 41u;

  EXPECT_TRUE(nuri::alignUpU64(1u, maximum, aligned));
  EXPECT_EQ(aligned, maximum);
  EXPECT_TRUE(nuri::alignUpU64(maximum, maximum, aligned));
  EXPECT_EQ(aligned, maximum);

  aligned = 41u;
  EXPECT_FALSE(nuri::alignUpU64(maximum, 2u, aligned));
  EXPECT_EQ(aligned, 41u);
  EXPECT_FALSE(nuri::alignUpU64(maximum - 1u, 4u, aligned));
  EXPECT_EQ(aligned, 41u);
}

struct RetryEvent {
  uint32_t id = 0u;
};

struct RetryState {
  uint32_t throwOn = 0u;
  bool didThrow = false;
  std::vector<uint32_t> calls;
};

bool throwOnce(const RetryEvent &event, void *user) {
  auto &state = *static_cast<RetryState *>(user);
  state.calls.push_back(event.id);
  if (!state.didThrow && event.id == state.throwOn) {
    state.didThrow = true;
    throw std::runtime_error("event callback failure");
  }
  return false;
}

std::vector<uint32_t>
dispatchWithOneRetry(std::initializer_list<uint32_t> eventIds,
                     uint32_t throwOn) {
  std::pmr::memory_resource *memory = std::pmr::new_delete_resource();
  nuri::EventManager events(*memory);
  RetryState state{.throwOn = throwOn};
  events.subscribe<RetryEvent>(nuri::EventChannel::Generic, &throwOnce, &state);
  for (const uint32_t id : eventIds) {
    events.emit(RetryEvent{.id = id});
  }

  EXPECT_THROW(events.dispatch(nuri::EventChannel::Generic),
               std::runtime_error);
  EXPECT_NO_THROW(events.dispatch(nuri::EventChannel::Generic));
  return state.calls;
}

TEST(EventManagerTests, RetriesOnlyThrowingQueuedEvent) {
  EXPECT_EQ(dispatchWithOneRetry({7u}, 7u), (std::vector<uint32_t>{7u, 7u}));
}

TEST(EventManagerTests, RetriesFirstThrowingQueuedEventAndLaterEvents) {
  EXPECT_EQ(dispatchWithOneRetry({1u, 2u}, 1u),
            (std::vector<uint32_t>{1u, 1u, 2u}));
}

TEST(EventManagerTests, RetriesLastThrowingQueuedEventWithoutEarlierEvents) {
  EXPECT_EQ(dispatchWithOneRetry({1u, 2u}, 2u),
            (std::vector<uint32_t>{1u, 2u, 2u}));
}

TEST(ResultTests, SwapExchangesSameAndDifferentAlternatives) {
  auto first = nuri::Result<int, int>::makeResult(3);
  auto second = nuri::Result<int, int>::makeResult(9);
  first.swap(second);
  EXPECT_EQ(first.value(), 9);
  EXPECT_EQ(second.value(), 3);

  auto error = nuri::Result<int, int>::makeError(17);
  first.swap(error);
  ASSERT_TRUE(first.hasError());
  EXPECT_EQ(first.error(), 17);
  ASSERT_TRUE(error.hasValue());
  EXPECT_EQ(error.value(), 9);

  auto voidValue = nuri::Result<void, int>::makeResult();
  auto voidError = nuri::Result<void, int>::makeError(23);
  voidValue.swap(voidError);
  EXPECT_TRUE(voidValue.hasError());
  EXPECT_EQ(voidValue.error(), 23);
  EXPECT_TRUE(voidError.hasValue());
}

void accessErrorFromValueResult() {
  auto value = nuri::Result<int, int>::makeResult(1);
  (void)value.error();
}

void accessValueFromErrorResult() {
  auto error = nuri::Result<int, int>::makeError(1);
  (void)error.value();
}

void accessErrorFromVoidValueResult() {
  auto value = nuri::Result<void, int>::makeResult();
  (void)value.error();
}

TEST(ResultTests, InvalidAlternativeAccessTerminatesInsteadOfThrowing) {
  EXPECT_DEATH_IF_SUPPORTED(accessErrorFromValueResult(), "");
  EXPECT_DEATH_IF_SUPPORTED(accessValueFromErrorResult(), "");
  EXPECT_DEATH_IF_SUPPORTED(accessErrorFromVoidValueResult(), "");
}

} // namespace
