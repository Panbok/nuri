#include "nuri/editor_pch.h"

#include "nuri/app/editor_runtime.h"
#include "nuri/app/editor_scene_catalog.h"
#include "nuri/core/profiling.h"

namespace nuri {
namespace {

template <typename Callback>
void measureLegacySceneCallback(std::string_view sceneLabel,
                                std::string_view phase, Callback &&callback) {
#if defined(NURI_DEBUG)
  using Clock = std::chrono::steady_clock;
  const auto start = Clock::now();
  callback();
  const double elapsedMs =
      std::chrono::duration<double, std::milli>(Clock::now() - start).count();
  if (elapsedMs > 2.0) {
    NURI_LOG_WARNING(
        "Editor scene '%.*s' %.*s callback exceeded the interactive "
        "operation limit: %.3f ms",
        static_cast<int>(sceneLabel.size()), sceneLabel.data(),
        static_cast<int>(phase.size()), phase.data(), elapsedMs);
  }
#else
  (void)sceneLabel;
  (void)phase;
  callback();
#endif
}

} // namespace

[[nodiscard]] Result<void, std::string>
EditorSceneCatalog::append(EditorSceneSpec spec) {
  if (spec.info.id.empty()) {
    return Result<void, std::string>::makeError(
        "EditorSceneCatalog::append: scene id must not be empty");
  }
  if (findIndex(spec.info.id).has_value()) {
    return Result<void, std::string>::makeError(
        std::string("EditorSceneCatalog::append: duplicate scene id '") +
        spec.info.id + "'");
  }
  if (spec.info.initiallySelected && initialIndex_.has_value()) {
    return Result<void, std::string>::makeError(
        std::string("EditorSceneCatalog::append: multiple initial scenes: '") +
        entries_[*initialIndex_].info.id + "' and '" + spec.info.id + "'");
  }

  const size_t index = entries_.size();
  entries_.push_back(EditorSceneEntry{.info = spec.info});
  specs_.push_back(std::move(spec));
  indexById_.emplace(entries_.back().info.id, index);
  if (entries_.back().info.initiallySelected) {
    initialIndex_ = index;
  }
  ++version_;
  return Result<void, std::string>::makeResult();
}

const EditorSceneSpec *EditorSceneCatalog::find(std::string_view id) const {
  const auto index = findIndex(id);
  return index.has_value() ? &specs_[*index] : nullptr;
}

std::span<const EditorSceneEntry> EditorSceneCatalog::entries() const noexcept {
  return entries_;
}

bool EditorSceneCatalog::requestActivate(std::string_view id) {
  const auto index = findIndex(id);
  if (!index.has_value()) {
    return false;
  }
  if (transitionIndex_ == *index && !pendingIndex_.has_value()) {
    return true;
  }
  pendingIndex_ = *index;
  cancelRequested_ = false;
  ++requestGeneration_;
  return true;
}

void EditorSceneCatalog::requestCancelPending() noexcept {
  cancelRequested_ = true;
  pendingIndex_.reset();
  ++requestGeneration_;
}

Result<bool, std::string>
EditorSceneCatalog::advanceTransition(EditorRuntime &runtime) {
  NURI_PROFILER_FUNCTION();
  const auto retireTransition = [&] {
    if (!transitionIndex_.has_value()) {
      return;
    }
    if (transitionActivated_) {
      EditorSceneSpec &spec = specs_[*transitionIndex_];
      if (spec.deactivate) {
        auto scope = runtime.useStagingSceneDocument();
        EditorSceneDeactivateContext context{.runtime = runtime};
        spec.deactivate(context);
      }
    }
    runtime.retireStagingSceneDocument();
    transitionIndex_.reset();
    transitionActivated_ = false;
    transitionReadyToCommit_ = false;
  };
  const auto failTransition = [&](size_t targetIndex, std::string error) {
    retireTransition();
    statusIndex_ = targetIndex;
    transitionPhase_ = EditorSceneTransitionPhase::Failed;
    transitionProgress_ = 1.0f;
    transitionError_ = std::move(error);
  };

  if (cancelRequested_) {
    cancelRequested_ = false;
    retireTransition();
    statusIndex_.reset();
    transitionPhase_ = EditorSceneTransitionPhase::Idle;
    transitionProgress_ = 0.0f;
    transitionError_.clear();
  }

  if (pendingIndex_.has_value()) {
    const size_t targetIndex = *pendingIndex_;
    pendingIndex_.reset();
    retireTransition();
    statusIndex_.reset();
    transitionError_.clear();
    transitionProgress_ = 0.0f;

    if (activeIndex_ == targetIndex) {
      transitionPhase_ = EditorSceneTransitionPhase::Idle;
      return Result<bool, std::string>::makeResult(false);
    }

    auto beginResult = runtime.beginStagingSceneDocument();
    if (beginResult.hasError()) {
      return Result<bool, std::string>::makeError(beginResult.error());
    }
    transitionIndex_ = targetIndex;
    statusIndex_ = targetIndex;
    transitionGeneration_ = requestGeneration_;
    transitionPhase_ = EditorSceneTransitionPhase::Preparing;

    EditorSceneEntry &targetEntry = entries_[targetIndex];
    EditorSceneSpec &targetSpec = specs_[targetIndex];
    std::string activationError{};
    {
      auto scope = runtime.useStagingSceneDocument();
      if (!targetEntry.prepared && targetSpec.prepare) {
        EditorScenePrepareContext prepareContext{.runtime = runtime};
        Result<void, std::string> prepareResult =
            Result<void, std::string>::makeResult();
        measureLegacySceneCallback(targetSpec.info.label, "prepare", [&] {
          prepareResult = targetSpec.prepare(prepareContext);
        });
        if (prepareResult.hasError()) {
          activationError = prepareResult.error();
        } else {
          targetEntry.prepared = true;
        }
      }

      if (activationError.empty()) {
        transitionActivated_ = true;
        if (targetSpec.activate) {
          EditorSceneActivateContext activateContext{.runtime = runtime};
          Result<void, std::string> activateResult =
              Result<void, std::string>::makeResult();
          measureLegacySceneCallback(targetSpec.info.label, "activate", [&] {
            activateResult = targetSpec.activate(activateContext);
          });
          if (activateResult.hasError()) {
            activationError = activateResult.error();
          }
        }
      }
    }
    if (!activationError.empty()) {
      failTransition(targetIndex, std::move(activationError));
      return Result<bool, std::string>::makeResult(false);
    }
    transitionPhase_ = EditorSceneTransitionPhase::LoadingAssets;
    transitionProgress_ = 0.05f;
  }

  if (!transitionIndex_.has_value()) {
    return Result<bool, std::string>::makeResult(false);
  }

  const size_t targetIndex = *transitionIndex_;
  EditorSceneSpec &targetSpec = specs_[targetIndex];
  if (!transitionReadyToCommit_) {
    {
      auto scope = runtime.useStagingSceneDocument();
      if (targetSpec.update) {
        EditorSceneUpdateContext updateContext{
            .runtime = runtime,
            .deltaTime = 0.0,
        };
        measureLegacySceneCallback(targetSpec.info.label, "staging update",
                                   [&] { targetSpec.update(updateContext); });
      }
    }

    const EditorSceneStagingStatus stagingStatus =
        targetSpec.stagingStatus ? targetSpec.stagingStatus()
                                 : EditorSceneStagingStatus{};
    if (stagingStatus.failed) {
      failTransition(targetIndex, stagingStatus.error.empty()
                                      ? "scene-local staging work failed"
                                      : stagingStatus.error);
      return Result<bool, std::string>::makeResult(false);
    }

    const ScenePublicationTargetSnapshot publication =
        runtime.stagingScenePublicationSnapshot();
    transitionProgress_ =
        std::max(transitionProgress_,
                 0.05f + publication.progress * 0.75f +
                     std::clamp(stagingStatus.progress, 0.0f, 1.0f) * 0.15f);
    if (publication.failedCount != 0u || publication.cancelledCount != 0u) {
      failTransition(targetIndex,
                     publication.failedCount != 0u
                         ? "scene asset publication failed"
                         : "scene asset publication was cancelled");
      return Result<bool, std::string>::makeResult(false);
    }
    if (!publication.ready() || !stagingStatus.ready) {
      return Result<bool, std::string>::makeResult(false);
    }

    transitionPhase_ = EditorSceneTransitionPhase::Finalizing;
    transitionProgress_ = std::max(transitionProgress_, 0.95f);
    auto finalizeResult = runtime.finalizeStagingSceneDocument();
    if (finalizeResult.hasError()) {
      failTransition(targetIndex, finalizeResult.error());
      return Result<bool, std::string>::makeResult(false);
    }
    if (!finalizeResult.value()) {
      return Result<bool, std::string>::makeResult(false);
    }
    transitionReadyToCommit_ = true;
    return Result<bool, std::string>::makeResult(false);
  }

  if (activeIndex_.has_value()) {
    EditorSceneEntry &activeEntry = entries_[*activeIndex_];
    EditorSceneSpec &activeSpec = specs_[*activeIndex_];
    if (activeSpec.deactivate) {
      auto scope = runtime.useActiveSceneDocument();
      EditorSceneDeactivateContext deactivateContext{.runtime = runtime};
      activeSpec.deactivate(deactivateContext);
    }
    activeEntry.active = false;
  }
  if (!runtime.activateStagingSceneDocument()) {
    return Result<bool, std::string>::makeError(
        "EditorSceneCatalog: staging document disappeared before activation");
  }

  activeIndex_ = targetIndex;
  entries_[targetIndex].active = true;
  transitionIndex_.reset();
  statusIndex_.reset();
  transitionActivated_ = false;
  transitionReadyToCommit_ = false;
  transitionPhase_ = EditorSceneTransitionPhase::Idle;
  transitionProgress_ = 0.0f;
  transitionError_.clear();
  return Result<bool, std::string>::makeResult(true);
}

std::string_view EditorSceneCatalog::initialSceneId() const noexcept {
  if (initialIndex_.has_value()) {
    return entries_[*initialIndex_].info.id;
  }
  return !entries_.empty() ? std::string_view(entries_.front().info.id)
                           : std::string_view{};
}

std::string_view EditorSceneCatalog::activeSceneId() const noexcept {
  return activeIndex_.has_value() ? entries_[*activeIndex_].info.id
                                  : std::string_view{};
}

EditorSceneTransitionSnapshot EditorSceneCatalog::transitionSnapshot() const {
  return EditorSceneTransitionSnapshot{
      .phase = transitionPhase_,
      .activeSceneId = activeSceneId(),
      .pendingSceneId = statusIndex_.has_value()
                            ? std::string_view(entries_[*statusIndex_].info.id)
                            : std::string_view{},
      .generation = transitionGeneration_,
      .progress = transitionProgress_,
      .cancellable = transitionIndex_.has_value(),
      .error = transitionError_,
  };
}

void EditorSceneCatalog::updateActive(EditorRuntime &runtime,
                                      double deltaTime) {
  NURI_PROFILER_FUNCTION();
  if (!activeIndex_.has_value()) {
    return;
  }
  EditorSceneSpec &activeSpec = specs_[*activeIndex_];
  if (activeSpec.update) {
    auto scope = runtime.useActiveSceneDocument();
    EditorSceneUpdateContext updateContext{
        .runtime = runtime,
        .deltaTime = deltaTime,
    };
    activeSpec.update(updateContext);
  }
}

void EditorSceneCatalog::shutdown(EditorRuntime &runtime) {
  NURI_PROFILER_FUNCTION();
  requestCancelPending();
  (void)advanceTransition(runtime);
  if (!activeIndex_.has_value()) {
    return;
  }
  EditorSceneEntry &activeEntry = entries_[*activeIndex_];
  EditorSceneSpec &activeSpec = specs_[*activeIndex_];
  if (activeSpec.deactivate) {
    auto scope = runtime.useActiveSceneDocument();
    EditorSceneDeactivateContext deactivateContext{.runtime = runtime};
    activeSpec.deactivate(deactivateContext);
  }
  activeEntry.active = false;
  activeIndex_.reset();
}

std::optional<size_t> EditorSceneCatalog::findIndex(std::string_view id) const {
  const auto it = indexById_.find(id);
  return it != indexById_.end() ? std::optional<size_t>(it->second)
                                : std::nullopt;
}

} // namespace nuri
