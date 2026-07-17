#include "nuri/editor_pch.h"

#include "nuri/app/editor_application.h"

#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
#include "nuri/core/runtime_config.h"
#include "nuri/utils/env_utils.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#elif defined(__linux__)
#include <time.h>
#endif

namespace nuri {
Result<void, std::string> registerBuiltInScenes(EditorSceneCatalog &catalog,
                                                const RuntimeConfig &config);

namespace {

[[nodiscard]] std::string jsonEscape(std::string_view value) {
  std::string escaped{};
  escaped.reserve(value.size() + 8u);
  for (const char character : value) {
    switch (character) {
    case '\\':
      escaped += "\\\\";
      break;
    case '"':
      escaped += "\\\"";
      break;
    case '\n':
      escaped += "\\n";
      break;
    case '\r':
      escaped += "\\r";
      break;
    case '\t':
      escaped += "\\t";
      break;
    default:
      escaped += character;
      break;
    }
  }
  return escaped;
}

[[nodiscard]] std::string_view
transitionPhaseName(EditorSceneTransitionPhase phase) noexcept {
  switch (phase) {
  case EditorSceneTransitionPhase::Idle:
    return "idle";
  case EditorSceneTransitionPhase::Preparing:
    return "preparing";
  case EditorSceneTransitionPhase::LoadingAssets:
    return "loading_assets";
  case EditorSceneTransitionPhase::Finalizing:
    return "finalizing";
  case EditorSceneTransitionPhase::Failed:
    return "failed";
  }
  return "unknown";
}

[[nodiscard]] double percentile(std::vector<double> values, double fraction) {
  if (values.empty()) {
    return 0.0;
  }
  std::ranges::sort(values);
  const size_t index =
      std::min(values.size() - 1u,
               static_cast<size_t>(
                   std::ceil(fraction * static_cast<double>(values.size()))) -
                   1u);
  return values[index];
}

[[nodiscard]] double maximum(const std::vector<double> &values) noexcept {
  return values.empty() ? 0.0 : *std::ranges::max_element(values);
}

[[nodiscard]] uint64_t currentThreadCpuTicks() noexcept {
#if defined(_WIN32)
  ULONG64 cycles = 0u;
  return ::QueryThreadCycleTime(::GetCurrentThread(), &cycles) != FALSE
             ? static_cast<uint64_t>(cycles)
             : 0u;
#elif defined(__linux__)
  timespec value{};
  return ::clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value) == 0
             ? static_cast<uint64_t>(value.tv_sec) * 1'000'000'000ull +
                   static_cast<uint64_t>(value.tv_nsec)
             : 0u;
#else
  return 0u;
#endif
}

[[nodiscard]] double calibrateThreadCpuTicksPerMillisecond() {
#if defined(_WIN32)
  using Clock = std::chrono::steady_clock;
  constexpr auto kCalibrationDuration = std::chrono::milliseconds(25);
  const uint64_t tickStart = currentThreadCpuTicks();
  const auto wallStart = Clock::now();
  auto wallNow = wallStart;
  do {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    wallNow = Clock::now();
  } while (wallNow - wallStart < kCalibrationDuration);
  const uint64_t tickEnd = currentThreadCpuTicks();
  const double elapsedMs =
      std::chrono::duration<double, std::milli>(wallNow - wallStart).count();
  return tickEnd > tickStart && elapsedMs > 0.0
             ? static_cast<double>(tickEnd - tickStart) / elapsedMs
             : 0.0;
#elif defined(__linux__)
  return 1'000'000.0;
#else
  return 0.0;
#endif
}

} // namespace

struct EditorApplication::TransitionProbe {
  using Clock = std::chrono::steady_clock;

  struct FrameSample {
    uint64_t frame = 0u;
    std::string phase{};
    float progress = 0.0f;
    double frameIntervalMs = 0.0;
    double coordinatorMs = 0.0;
    double coordinatorWallMs = 0.0;
    double sceneUpdateMs = 0.0;
    double runtimeUpdateMs = 0.0;
    double drawMs = 0.0;
    double publicationMs = 0.0;
    double maximumPublicationOperationMs = 0.0;
    uint32_t queuedCpuJobs = 0u;
    uint32_t deferredCpuCompletions = 0u;
    bool publicationDeadlineExceeded = false;
  };

  static std::unique_ptr<TransitionProbe> fromEnvironment() {
    const std::optional<std::string> target =
        readEnvVar("NURI_EDITOR_TRANSITION_TARGET");
    if (!target.has_value() || target->empty()) {
      return nullptr;
    }
    const std::optional<std::string> report =
        readEnvVar("NURI_EDITOR_TRANSITION_REPORT");
    const std::optional<std::string> source =
        readEnvVar("NURI_EDITOR_TRANSITION_SOURCE");
    auto probe = std::make_unique<TransitionProbe>();
    probe->targetScene = *target;
    probe->sourceScene =
        source.has_value() && !source->empty() ? *source : "single_duck";
    probe->reportPath =
        report.has_value() && !report->empty()
            ? std::filesystem::path(*report)
            : std::filesystem::path(
                  "artifacts/editor-transition/latest/report.json");
    probe->threadCpuTicksPerMillisecond =
        calibrateThreadCpuTicksPerMillisecond();
    return probe;
  }

  void beginFrame(uint64_t frame, Clock::time_point now) {
    currentFrame = frame;
    if (!requestIssued || completed) {
      return;
    }
    if (lastFrameStart.has_value()) {
      currentFrameIntervalMs =
          std::chrono::duration<double, std::milli>(now - *lastFrameStart)
              .count();
      frameIntervalsMs.push_back(currentFrameIntervalMs);
    }
    lastFrameStart = now;
    if (!pendingUiAcknowledgement.has_value()) {
      pendingUiAcknowledgement = now;
      pendingUiAcknowledgementPresentedFrame = presentedFrameCounter;
    }
  }

  void markRequested(uint64_t frame, Clock::time_point now) {
    requestIssued = true;
    requestFrame = frame;
    requestTime = now;
    lastFrameStart = now;
  }

  void recordUpdate(const EditorSceneTransitionSnapshot &snapshot,
                    double coordinatorMs, double coordinatorWallMs,
                    double sceneUpdateMs, double runtimeUpdateMs) {
    if (!requestIssued || completed) {
      return;
    }
    coordinatorTimesMs.push_back(coordinatorMs);
    coordinatorWallTimesMs.push_back(coordinatorWallMs);
    sceneUpdateTimesMs.push_back(sceneUpdateMs);
    runtimeUpdateTimesMs.push_back(runtimeUpdateMs);
    samples.push_back(FrameSample{
        .frame = currentFrame,
        .phase = std::string(transitionPhaseName(snapshot.phase)),
        .progress = snapshot.progress,
        .frameIntervalMs = currentFrameIntervalMs,
        .coordinatorMs = coordinatorMs,
        .coordinatorWallMs = coordinatorWallMs,
        .sceneUpdateMs = sceneUpdateMs,
        .runtimeUpdateMs = runtimeUpdateMs,
    });
  }

  void recordDraw(const RenderFrameMetrics::AssetStreamingFrameMetrics &assets,
                  double drawMs, Clock::time_point now,
                  bool frameOutputAvailable) {
    if (!requestIssued || completed) {
      return;
    }
    publicationTimesMs.push_back(assets.publicationMainThreadMilliseconds);
    drawTimesMs.push_back(drawMs);
    publicationOperationTimesMs.push_back(
        assets.publicationMaxOperationMilliseconds);
    deadlineExceededFrames +=
        assets.publicationDeadlineExceeded != 0u ? 1u : 0u;
    if (!samples.empty() && samples.back().frame == currentFrame) {
      FrameSample &sample = samples.back();
      sample.drawMs = drawMs;
      sample.publicationMs = assets.publicationMainThreadMilliseconds;
      sample.maximumPublicationOperationMs =
          assets.publicationMaxOperationMilliseconds;
      sample.queuedCpuJobs = assets.cpuQueuedJobs;
      sample.deferredCpuCompletions = assets.deferredCpuCompletions;
      sample.publicationDeadlineExceeded =
          assets.publicationDeadlineExceeded != 0u;
    }
    if (frameOutputAvailable) {
      ++presentedFrameCounter;
    }
    if (frameOutputAvailable && pendingUiAcknowledgement.has_value()) {
      uiAcknowledgementTimesMs.push_back(
          std::chrono::duration<double, std::milli>(now -
                                                    *pendingUiAcknowledgement)
              .count());
      uiAcknowledgementFrameDeltas.push_back(
          presentedFrameCounter - pendingUiAcknowledgementPresentedFrame);
      pendingUiAcknowledgement.reset();
    }
  }

  void markActivated(uint64_t frame, Clock::time_point now) {
    if (activated || completed) {
      return;
    }
    activated = true;
    activationFrame = frame;
    activationTime = now;
  }

  void recordActivationCommit(double coordinatorMs) noexcept {
    activationCommitMs = coordinatorMs;
  }

  void fail(std::string message) {
    if (!failure.empty()) {
      return;
    }
    failure = std::move(message);
    completed = true;
  }

  [[nodiscard]] bool advanceStableFrame() {
    if (!activated || completed) {
      return false;
    }
    ++stableFrames;
    if (stableFrames >= 5u) {
      completed = true;
      return true;
    }
    return false;
  }

  [[nodiscard]] bool timedOut() const noexcept {
    return requestIssued && !completed && currentFrame > requestFrame + 18000u;
  }

  void writeReport() {
    if (!requestIssued && failure.empty()) {
      fail("editor closed before the transition probe could issue its request");
    } else if (!completed) {
      fail("editor closed before the target scene reached a stable active "
           "state");
    }

    const double frameP99 = percentile(frameIntervalsMs, 0.99);
    const double frameMaximum = maximum(frameIntervalsMs);
    const double coordinatorP99 = percentile(coordinatorTimesMs, 0.99);
    const double coordinatorMaximum = maximum(coordinatorTimesMs);
    const double coordinatorWallP99 = percentile(coordinatorWallTimesMs, 0.99);
    const double coordinatorWallMaximum = maximum(coordinatorWallTimesMs);
    const double sceneUpdateMaximum = maximum(sceneUpdateTimesMs);
    const double runtimeUpdateMaximum = maximum(runtimeUpdateTimesMs);
    const double drawMaximum = maximum(drawTimesMs);
    const double uiP99 = percentile(uiAcknowledgementTimesMs, 0.99);
    const double uiMaximum = maximum(uiAcknowledgementTimesMs);
    const uint64_t uiFrameMaximum =
        uiAcknowledgementFrameDeltas.empty()
            ? 0u
            : *std::ranges::max_element(uiAcknowledgementFrameDeltas);
    const double publicationP99 = percentile(publicationTimesMs, 0.99);
    const double publicationMaximum = maximum(publicationTimesMs);
    const double operationP99 = percentile(publicationOperationTimesMs, 0.99);
    const double operationMaximum = maximum(publicationOperationTimesMs);
    const double transitionDurationMs =
        activated ? std::chrono::duration<double, std::milli>(activationTime -
                                                              requestTime)
                        .count()
                  : 0.0;

    std::vector<std::string> gateFailures{};
    if (!activated) {
      gateFailures.emplace_back("target scene was not activated");
    }
    if (uiP99 > 50.0 || uiFrameMaximum > 2u) {
      gateFailures.emplace_back(
          "input-to-visible acknowledgement exceeded 50 ms or two frames");
    }
    if (frameMaximum > 100.0) {
      gateFailures.emplace_back("load-phase frame exceeded 100 ms");
    }
    if (coordinatorP99 > 2.0 || coordinatorMaximum > 4.0) {
      gateFailures.emplace_back(
          "load-attributable coordinator overhead exceeded its p99 or "
          "maximum budget");
    }
    if (coordinatorWallMaximum > 50.0) {
      gateFailures.emplace_back("coordinator wall-clock stall exceeded 50 ms");
    }
    if (activationCommitMs > 2.0) {
      gateFailures.emplace_back("active-document commit exceeded 2 ms");
    }
    if (publicationP99 > 2.0 || publicationMaximum > 4.0) {
      gateFailures.emplace_back(
          "asset publication exceeded its p99 or maximum frame budget");
    }
    if (operationP99 > 0.5 || operationMaximum > 2.0) {
      gateFailures.emplace_back(
          "a publication work unit exceeded its p99 or maximum budget");
    }
    if (deadlineExceededFrames != 0u) {
      gateFailures.emplace_back("asset publication deadline was exceeded");
    }
    if (!failure.empty()) {
      gateFailures.push_back(failure);
    }
    exitStatus = gateFailures.empty() ? 0 : 1;

    std::error_code filesystemError{};
    const std::filesystem::path parent = reportPath.parent_path();
    if (!parent.empty()) {
      std::filesystem::create_directories(parent, filesystemError);
    }
    std::ofstream output(reportPath, std::ios::binary | std::ios::trunc);
    if (!output) {
      NURI_LOG_ERROR("Editor transition probe: failed to open report '%s'",
                     reportPath.string().c_str());
      exitStatus = 1;
      return;
    }

    output << std::fixed << std::setprecision(6);
    output << "{\n"
           << "  \"schemaVersion\": 2,\n"
           << "  \"kind\": \"nuri.editor-transition.report\",\n"
           << "  \"status\": \"" << (exitStatus == 0 ? "pass" : "fail")
           << "\",\n"
           << "  \"targetScene\": \"" << jsonEscape(targetScene) << "\",\n"
           << "  \"sourceScene\": \"" << jsonEscape(sourceScene) << "\",\n"
           << "  \"requestFrame\": " << requestFrame << ",\n"
           << "  \"activationFrame\": " << activationFrame << ",\n"
           << "  \"transitionDurationMs\": " << transitionDurationMs << ",\n"
           << "  \"metrics\": {\n"
           << "    \"frameIntervalP99Ms\": " << frameP99 << ",\n"
           << "    \"frameIntervalMaximumMs\": " << frameMaximum << ",\n"
           << "    \"coordinatorP99Ms\": " << coordinatorP99 << ",\n"
           << "    \"coordinatorMaximumMs\": " << coordinatorMaximum << ",\n"
           << "    \"coordinatorCpuP99Ms\": " << coordinatorP99 << ",\n"
           << "    \"coordinatorCpuMaximumMs\": " << coordinatorMaximum << ",\n"
           << "    \"coordinatorWallP99Ms\": " << coordinatorWallP99 << ",\n"
           << "    \"coordinatorWallMaximumMs\": " << coordinatorWallMaximum
           << ",\n"
           << "    \"threadCpuTicksPerMillisecond\": "
           << threadCpuTicksPerMillisecond << ",\n"
           << "    \"activationCommitMs\": " << activationCommitMs << ",\n"
           << "    \"sceneUpdateMaximumMs\": " << sceneUpdateMaximum << ",\n"
           << "    \"runtimeUpdateMaximumMs\": " << runtimeUpdateMaximum
           << ",\n"
           << "    \"drawMaximumMs\": " << drawMaximum << ",\n"
           << "    \"uiAcknowledgementP99Ms\": " << uiP99 << ",\n"
           << "    \"uiAcknowledgementMaximumMs\": " << uiMaximum << ",\n"
           << "    \"uiAcknowledgementMaximumFrames\": " << uiFrameMaximum
           << ",\n"
           << "    \"publicationP99Ms\": " << publicationP99 << ",\n"
           << "    \"publicationMaximumMs\": " << publicationMaximum << ",\n"
           << "    \"publicationOperationP99Ms\": " << operationP99 << ",\n"
           << "    \"publicationOperationMaximumMs\": " << operationMaximum
           << ",\n"
           << "    \"publicationDeadlineExceededFrames\": "
           << deadlineExceededFrames << "\n"
           << "  },\n"
           << "  \"gateFailures\": [";
    for (size_t index = 0u; index < gateFailures.size(); ++index) {
      output << (index == 0u ? "\n" : ",\n") << "    \""
             << jsonEscape(gateFailures[index]) << "\"";
    }
    if (!gateFailures.empty()) {
      output << '\n';
    }
    output << "  ],\n  \"frames\": [";
    for (size_t index = 0u; index < samples.size(); ++index) {
      const FrameSample &sample = samples[index];
      output << (index == 0u ? "\n" : ",\n")
             << "    {\"frame\": " << sample.frame << ", \"phase\": \""
             << jsonEscape(sample.phase)
             << "\", \"progress\": " << sample.progress
             << ", \"frameIntervalMs\": " << sample.frameIntervalMs
             << ", \"coordinatorMs\": " << sample.coordinatorMs
             << ", \"coordinatorCpuMs\": " << sample.coordinatorMs
             << ", \"coordinatorWallMs\": " << sample.coordinatorWallMs
             << ", \"sceneUpdateMs\": " << sample.sceneUpdateMs
             << ", \"runtimeUpdateMs\": " << sample.runtimeUpdateMs
             << ", \"drawMs\": " << sample.drawMs
             << ", \"publicationMs\": " << sample.publicationMs
             << ", \"maximumPublicationOperationMs\": "
             << sample.maximumPublicationOperationMs
             << ", \"queuedCpuJobs\": " << sample.queuedCpuJobs
             << ", \"deferredCpuCompletions\": "
             << sample.deferredCpuCompletions
             << ", \"publicationDeadlineExceeded\": "
             << (sample.publicationDeadlineExceeded ? "true" : "false") << '}';
    }
    if (!samples.empty()) {
      output << '\n';
    }
    output << "  ]\n}\n";
    NURI_LOG_INFO("Editor transition probe: %s report='%s'",
                  exitStatus == 0 ? "pass" : "fail",
                  reportPath.string().c_str());
  }

  std::string targetScene{};
  std::string sourceScene{};
  std::filesystem::path reportPath{};
  std::vector<FrameSample> samples{};
  std::vector<double> frameIntervalsMs{};
  std::vector<double> coordinatorTimesMs{};
  std::vector<double> coordinatorWallTimesMs{};
  std::vector<double> sceneUpdateTimesMs{};
  std::vector<double> runtimeUpdateTimesMs{};
  std::vector<double> drawTimesMs{};
  std::vector<double> uiAcknowledgementTimesMs{};
  std::vector<uint64_t> uiAcknowledgementFrameDeltas{};
  std::vector<double> publicationTimesMs{};
  std::vector<double> publicationOperationTimesMs{};
  std::optional<Clock::time_point> lastFrameStart{};
  std::optional<Clock::time_point> pendingUiAcknowledgement{};
  Clock::time_point requestTime{};
  Clock::time_point activationTime{};
  uint64_t currentFrame = 0u;
  uint64_t requestFrame = 0u;
  uint64_t activationFrame = 0u;
  uint64_t pendingUiAcknowledgementPresentedFrame = 0u;
  uint64_t presentedFrameCounter = 0u;
  double currentFrameIntervalMs = 0.0;
  double activationCommitMs = 0.0;
  double threadCpuTicksPerMillisecond = 0.0;
  uint32_t deadlineExceededFrames = 0u;
  uint32_t initialStableFrames = 0u;
  uint32_t stableFrames = 0u;
  bool requestIssued = false;
  bool sourceSelectionIssued = false;
  bool activated = false;
  bool completed = false;
  int exitStatus = 0;
  std::string failure{};
};

ApplicationConfig
EditorApplication::makeEditorApplicationConfig(const RuntimeConfig &config) {
  ApplicationConfig appConfig = makeApplicationConfig(config);
  appConfig.renderComposition = RenderCompositionMode::PipelineOnly;
  return appConfig;
}

EditorApplication::EditorApplication(RuntimeConfig config)
    : Application(makeEditorApplicationConfig(config)), config_(config),
      runtime_(*this, config), scenes_(),
      transitionProbe_(TransitionProbe::fromEnvironment()) {}

EditorApplication::~EditorApplication() = default;

void EditorApplication::onInit() {
  auto registerResult = registerBuiltInScenes(scenes_, config_);
  NURI_ASSERT(!registerResult.hasError(),
              "Failed to register built-in editor scenes: %s",
              registerResult.error().c_str());
  runtime_.initialize();
  const std::string_view initialSceneId = scenes_.initialSceneId();
  NURI_ASSERT(!initialSceneId.empty(),
              "Failed to determine initial editor scene");
  NURI_ASSERT(scenes_.requestActivate(initialSceneId),
              "Failed to request initial scene activation");
  NURI_LOG_INFO("Editor application initialized");
}

void EditorApplication::onUpdate(double deltaTime) {
  static uint64_t editorFrame = 0u;
  ++editorFrame;
  const auto frameStart = TransitionProbe::Clock::now();
  if (transitionProbe_ != nullptr) {
    transitionProbe_->beginFrame(editorFrame, frameStart);
  }
  const auto coordinatorStart = TransitionProbe::Clock::now();
  const uint64_t coordinatorCpuStart =
      transitionProbe_ != nullptr ? currentThreadCpuTicks() : 0u;
  auto transitionResult = scenes_.advanceTransition(runtime_);
  const uint64_t coordinatorCpuEnd =
      transitionProbe_ != nullptr ? currentThreadCpuTicks() : 0u;
  const double coordinatorWallMs =
      std::chrono::duration<double, std::milli>(TransitionProbe::Clock::now() -
                                                coordinatorStart)
          .count();
  const double coordinatorMs =
      transitionProbe_ != nullptr &&
              transitionProbe_->threadCpuTicksPerMillisecond > 0.0 &&
              coordinatorCpuEnd >= coordinatorCpuStart
          ? static_cast<double>(coordinatorCpuEnd - coordinatorCpuStart) /
                transitionProbe_->threadCpuTicksPerMillisecond
          : coordinatorWallMs;
  if (transitionResult.hasError()) {
    NURI_LOG_ERROR("Editor scene transition failed: %s",
                   transitionResult.error().c_str());
    if (transitionProbe_ != nullptr) {
      transitionProbe_->fail(transitionResult.error());
    }
  } else if (transitionProbe_ != nullptr && transitionResult.value()) {
    transitionProbe_->recordActivationCommit(coordinatorWallMs);
  }
  const auto sceneUpdateStart = TransitionProbe::Clock::now();
  scenes_.updateActive(runtime_, deltaTime);
  const double sceneUpdateMs =
      std::chrono::duration<double, std::milli>(TransitionProbe::Clock::now() -
                                                sceneUpdateStart)
          .count();
  const auto runtimeUpdateStart = TransitionProbe::Clock::now();
  runtime_.update(deltaTime);
  const double runtimeUpdateMs =
      std::chrono::duration<double, std::milli>(TransitionProbe::Clock::now() -
                                                runtimeUpdateStart)
          .count();

  if (transitionProbe_ == nullptr) {
    return;
  }
  const EditorSceneTransitionSnapshot snapshot = scenes_.transitionSnapshot();
  transitionProbe_->recordUpdate(snapshot, coordinatorMs, coordinatorWallMs,
                                 sceneUpdateMs, runtimeUpdateMs);
  const AssetCpuSchedulerStats initialCpuStats = runtime_.assets().cpuStats();
  const RenderFrameMetrics::AssetStreamingFrameMetrics &initialAssetMetrics =
      runtime_.frameContext().metrics.assets;
  const bool initialAssetsIdle =
      initialCpuStats.queuedJobs == 0u && initialCpuStats.runningJobs == 0u &&
      initialAssetMetrics.cpuCompletions == 0u &&
      initialAssetMetrics.gpuMaterialized == 0u &&
      initialAssetMetrics.published == 0u &&
      initialAssetMetrics.deferredCpuCompletions == 0u &&
      initialAssetMetrics.uploadBytes == 0u;
  if (!transitionProbe_->sourceSelectionIssued) {
    if (transitionProbe_->sourceScene == transitionProbe_->targetScene) {
      transitionProbe_->fail(
          "transition probe source and target scenes must be distinct");
    } else if (scenes_.find(transitionProbe_->sourceScene) == nullptr) {
      transitionProbe_->fail("unknown source scene: " +
                             transitionProbe_->sourceScene);
    } else if (scenes_.activeSceneId() == transitionProbe_->sourceScene &&
               snapshot.phase == EditorSceneTransitionPhase::Idle) {
      transitionProbe_->sourceSelectionIssued = true;
    } else if (!scenes_.requestActivate(transitionProbe_->sourceScene)) {
      transitionProbe_->fail("failed to request source scene: " +
                             transitionProbe_->sourceScene);
    } else {
      transitionProbe_->sourceSelectionIssued = true;
      transitionProbe_->initialStableFrames = 0u;
    }
  }
  if (!transitionProbe_->requestIssued &&
      transitionProbe_->sourceSelectionIssued &&
      scenes_.activeSceneId() == transitionProbe_->sourceScene &&
      snapshot.phase == EditorSceneTransitionPhase::Idle && initialAssetsIdle) {
    ++transitionProbe_->initialStableFrames;
  } else if (!transitionProbe_->requestIssued) {
    transitionProbe_->initialStableFrames = 0u;
  }
  if (!transitionProbe_->requestIssued &&
      transitionProbe_->sourceSelectionIssued &&
      scenes_.activeSceneId() == transitionProbe_->sourceScene &&
      snapshot.phase == EditorSceneTransitionPhase::Idle && initialAssetsIdle &&
      transitionProbe_->initialStableFrames >= 60u) {
    if (scenes_.find(transitionProbe_->targetScene) == nullptr) {
      transitionProbe_->fail("unknown target scene: " +
                             transitionProbe_->targetScene);
    } else if (scenes_.activeSceneId() == transitionProbe_->targetScene) {
      transitionProbe_->fail(
          "target scene is already active; choose a distinct target");
    } else if (!scenes_.requestActivate(transitionProbe_->targetScene)) {
      transitionProbe_->fail("failed to request target scene: " +
                             transitionProbe_->targetScene);
    } else {
      transitionProbe_->markRequested(editorFrame,
                                      TransitionProbe::Clock::now());
    }
  }
  if (transitionProbe_->requestIssued && !transitionProbe_->activated &&
      scenes_.activeSceneId() == transitionProbe_->targetScene &&
      snapshot.phase == EditorSceneTransitionPhase::Idle) {
    transitionProbe_->markActivated(editorFrame, TransitionProbe::Clock::now());
  }
  if (snapshot.phase == EditorSceneTransitionPhase::Failed) {
    transitionProbe_->fail(snapshot.error.empty()
                               ? "scene transition failed"
                               : std::string(snapshot.error));
  }
  if (transitionProbe_->timedOut()) {
    transitionProbe_->fail("scene transition exceeded 18000 editor frames");
  }
  if (transitionProbe_->completed || transitionProbe_->advanceStableFrame()) {
    getWindow().requestClose();
  }
}

void EditorApplication::onDraw() {
  const auto drawStart = TransitionProbe::Clock::now();
  runtime_.syncSceneSelectionUi(scenes_);
  runtime_.draw();
  if (transitionProbe_ != nullptr) {
    const auto drawEnd = TransitionProbe::Clock::now();
    transitionProbe_->recordDraw(
        runtime_.frameContext().metrics.assets,
        std::chrono::duration<double, std::milli>(drawEnd - drawStart).count(),
        drawEnd, runtime_.lastFrameOutputAvailable());
  }
  if (const auto requestedScene = runtime_.takeSceneSelectionRequest();
      requestedScene.has_value()) {
    if (!scenes_.requestActivate(*requestedScene)) {
      NURI_LOG_ERROR("EditorApplication::onDraw: invalid scene selection "
                     "request '%s'",
                     requestedScene->c_str());
      NURI_ASSERT(false, "Failed to request scene activation for '%s'",
                  requestedScene->c_str());
    }
  }
  if (runtime_.takeSceneCancelRequest()) {
    scenes_.requestCancelPending();
  }
}

void EditorApplication::onResize(std::int32_t width, std::int32_t height) {
  runtime_.resize(width, height);
}

bool EditorApplication::onInput(const InputEvent &event) {
  return runtime_.onInput(event) || Application::onInput(event);
}

void EditorApplication::onShutdown() {
  scenes_.shutdown(runtime_);
  runtime_.shutdown();
  if (transitionProbe_ != nullptr) {
    transitionProbe_->writeReport();
  }
  NURI_LOG_INFO("Editor application shutdown");
}

int EditorApplication::exitCode() const noexcept {
  return transitionProbe_ != nullptr ? transitionProbe_->exitStatus : 0;
}

} // namespace nuri
